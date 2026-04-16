import argparse
import json
import math
import os
import re
import subprocess
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
from torchvision.ops import StochasticDepth
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from Bio import SeqIO
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


class CharacterTokenizer(PreTrainedTokenizer):
    """Посимвольный токенайзер для ДНК-последовательностей."""

    def __init__(self, characters: Sequence[str], model_max_length: int,
                 padding_side: str = "left", **kwargs):
        self.characters = list(characters)
        self.model_max_length = model_max_length
        self._vocab_str_to_int = {
            "[CLS]": 0, "[SEP]": 1, "[BOS]": 2, "[MASK]": 3,
            "[PAD]": 4, "[RESERVED]": 5, "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(self.characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        super().__init__(
            bos_token=AddedToken("[BOS]"), eos_token=AddedToken("[SEP]"),
            sep_token=AddedToken("[SEP]"), cls_token=AddedToken("[CLS]"),
            pad_token=AddedToken("[PAD]"), unk_token=AddedToken("[UNK]"),
            mask_token=AddedToken("[MASK]", lstrip=True),
            add_prefix_space=False, model_max_length=model_max_length,
            padding_side=padding_side, **kwargs,
        )

    def get_vocab(self): return dict(self._vocab_str_to_int)

    @property
    def vocab_size(self): return len(self._vocab_str_to_int)

    def __len__(self): return self.vocab_size

    def _tokenize(self, text): return list(text)

    def _convert_token_to_id(self, token):
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index):
        return self._vocab_int_to_str.get(index, "[UNK]")

    def convert_tokens_to_string(self, tokens): return "".join(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        result = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        if token_ids_1 is not None:
            result += token_ids_1 + [self.sep_token_id]
        return result


def fftconv(u, k, D):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]
    return (y + u * D.unsqueeze(-1)).to(dtype=u.dtype)


@torch.jit.script
def mul_sum(q, y):
    return (q * y).sum(dim=1)


class OptimModule(nn.Module):
    def register(self, name, tensor, lr=None, wd=0.0):
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {}
            if lr is not None: optim["lr"] = lr
            if wd is not None: optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)


class Sin(nn.Module):
    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim)) if train_freq else w * torch.ones(1, dim)

    def forward(self, x):
        return torch.sin(self.freq * x)


class PositionalEmbedding(OptimModule):
    def __init__(self, emb_dim, seq_len, lr_pos_emb=1e-5, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]
        bands = (emb_dim - 1) // 2
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len
        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.register("z", z, lr=lr_pos_emb)
        self.register("t", t, lr=0.0)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(OptimModule):
    def __init__(self, d_model, fast_decay_pct=0.3, slow_decay_pct=1.5,
                 target=1e-2, modulation_lr=0.0, modulate=True, shift=0.05, **kwargs):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register("deltas", deltas, lr=modulation_lr)

    def forward(self, t, x):
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs())
            x = x * (decay + self.shift)
        return x


class HyenaFilter(OptimModule):
    def __init__(self, d_model, emb_dim=3, order=16, fused_fft_conv=False,
                 seq_len=1024, lr=1e-3, lr_pos_emb=1e-5, dropout=0.0,
                 w=1, wd=0, bias=True, num_inner_mlps=2, normalized=False, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.use_bias = bias
        self.fused_fft_conv = fused_fft_conv
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)
        act = Sin(dim=order, w=w)
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)
        self.implicit_filter = nn.Sequential(nn.Linear(emb_dim, order), act)
        for _ in range(num_inner_mlps):
            self.implicit_filter.append(nn.Linear(order, order))
            self.implicit_filter.append(act)
        self.implicit_filter.append(nn.Linear(order, d_model, bias=False))
        self.modulation = ExponentialModulation(d_model, **kwargs)
        self.normalized = normalized
        for c in self.implicit_filter.children():
            for name, v in c.state_dict().items():
                setattr(getattr(c, name), "_optim", {"weight_decay": wd, "lr": lr})

    def filter(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        return self.modulation(t, h)

    def forward(self, x, L, k=None, bias=None, *args, **kwargs):
        if k is None: k = self.filter(L)
        k = k[0] if type(k) is tuple else k
        return fftconv(x, k, bias)


class HyenaOperator(nn.Module):
    def __init__(self, d_model, l_max, order=2, filter_order=64,
                 dropout=0.0, filter_dropout=0.0, **filter_args):
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        inner_width = d_model * (order + 1)
        self.dropout = nn.Dropout(dropout)
        self.in_proj = nn.Linear(d_model, inner_width)
        self.out_proj = nn.Linear(d_model, d_model)
        self.short_filter = nn.Conv1d(inner_width, inner_width, 3, padding=2, groups=inner_width)
        self.filter_fn = HyenaFilter(
            d_model * (order - 1), order=filter_order, seq_len=l_max,
            channels=1, dropout=filter_dropout, **filter_args,
        )

    def forward(self, u, *args, **kwargs):
        l = u.size(-2)
        l_filter = min(l, self.l_max)
        u = self.in_proj(u)
        u = rearrange(u, "b l d -> b d l")
        uc = self.short_filter(u)[..., :l_filter]
        *x, v = uc.split(self.d_model, dim=1)
        k = self.filter_fn.filter(l_filter)[0]
        k = rearrange(k, "l (o d) -> o d l", o=self.order - 1)
        bias = rearrange(self.filter_fn.bias, "(o d) -> o d", o=self.order - 1)
        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o])
        y = rearrange(v * x[0], "b d l -> b l d")
        return self.out_proj(y)


class SelfAttention(nn.Module):
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, causal=None, key_padding_mask=None):
        batch_size, seqlen = qkv.shape[0], qkv.shape[1]
        causal = self.causal if causal is None else causal
        q, k, v = qkv.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        if key_padding_mask is not None:
            padding_mask = torch.full((batch_size, seqlen), -10000.0,
                                      dtype=scores.dtype, device=scores.device)
            padding_mask.masked_fill_(key_padding_mask, 0.0)
            scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")
        if causal:
            causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
            scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = F.dropout(attention, self.dropout_p if self.training else 0.0)
        return torch.einsum("bhts,bshd->bthd", attention_drop, v)


class LinearResidual(nn.Linear):
    def forward(self, input):
        return super().forward(input), input


class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True, dropout=0.0,
                 softmax_scale=None, causal=False, layer_idx=None,
                 dwconv=False, return_residual=False, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.layer_idx = layer_idx
        self.dwconv = dwconv
        self.return_residual = return_residual
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if not self.return_residual:
            self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        else:
            self.Wqkv = LinearResidual(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        if self.dwconv:
            self.dwconv_qkv = nn.Conv1d(3 * embed_dim, 3 * embed_dim,
                                        kernel_size=3, padding=2, groups=3 * embed_dim)
        self.inner_attn = SelfAttention(causal=causal, softmax_scale=softmax_scale,
                                        attention_dropout=dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)

    def forward(self, x, key_padding_mask=None, **kwargs):
        if not self.return_residual:
            qkv = self.Wqkv(x)
        else:
            qkv, x = self.Wqkv(x)
        if self.dwconv:
            qkv = rearrange(
                self.dwconv_qkv(rearrange(qkv, "b s d -> b d s"))[..., :-2], "b d s -> b s d"
            ).contiguous()
        qkv = rearrange(qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim)
        context = self.inner_attn(qkv, key_padding_mask=key_padding_mask)
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out if not self.return_residual else (out, x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 activation=F.gelu, return_residual=False, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, **factory_kwargs)

    def forward(self, x):
        y = self.activation(self.fc1(x))
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)


class Block(nn.Module):
    def __init__(self, dim, mixer_cls=None, mlp_cls=None, norm_cls=nn.LayerNorm,
                 dropout_cls=nn.Dropout, prenorm=True, resid_dropout1=0.,
                 resid_dropout2=0., drop_path1=0., drop_path2=0.,
                 return_residual=False, residual_in_fp32=False):
        super().__init__()
        self.prenorm = prenorm
        self.return_residual = return_residual
        self.residual_in_fp32 = residual_in_fp32
        if mixer_cls is None:
            mixer_cls = partial(MHA, num_heads=dim // 64)
        if mlp_cls is None:
            mlp_cls = partial(Mlp, hidden_features=4 * dim)
        self.mixer = mixer_cls()
        self.dropout1 = dropout_cls(resid_dropout1)
        self.drop_path1 = StochasticDepth(drop_path1, mode="row")
        self.norm1 = norm_cls(dim)
        self.mlp = mlp_cls(dim)
        if not isinstance(self.mlp, nn.Identity):
            self.dropout2 = dropout_cls(resid_dropout2)
            self.drop_path2 = StochasticDepth(drop_path2, mode="row")
            self.norm2 = norm_cls(dim)

    def forward(self, hidden_states, residual=None, mixer_subset=None, mixer_kwargs=None):
        if self.prenorm:
            dropped = self.drop_path1(self.dropout1(hidden_states))
            residual = (dropped + residual) if residual is not None else dropped
            hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
            hidden_states = self.mixer(hidden_states, **(mixer_kwargs or {}))
            if not isinstance(self.mlp, nn.Identity):
                dropped = self.drop_path2(self.dropout2(hidden_states))
                residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
                hidden_states = self.mlp(hidden_states)
            return hidden_states, residual
        else:
            mixer_out = self.mixer(hidden_states, **(mixer_kwargs or {}))
            if self.return_residual:
                mixer_out, hidden_states = mixer_out
            hidden_states = self.norm1(
                (self.drop_path1(self.dropout1(mixer_out)) + hidden_states).to(
                    dtype=self.norm1.weight.dtype
                )
            )
            if not isinstance(self.mlp, nn.Identity):
                mlp_out = self.mlp(hidden_states)
                if self.return_residual:
                    mlp_out, hidden_states = mlp_out
                hidden_states = self.norm2(
                    (self.drop_path2(self.dropout2(mlp_out)) + hidden_states).to(
                        dtype=self.norm2.weight.dtype
                    )
                )
            return hidden_states


def create_mixer_cls(layer=None, attn_layer_idx=None, attn_cfg=None,
                     layer_idx=None, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    if attn_layer_idx is not None and layer_idx in attn_layer_idx:
        causal = True if attn_cfg is None else attn_cfg.pop("causal", True)
        mixer_cls = partial(MHA, causal=causal, layer_idx=layer_idx,
                            **(attn_cfg or {}), **factory_kwargs)
    else:
        mixer_cls = partial(HyenaOperator, **layer)
    return mixer_cls


def create_mlp_cls(d_model, d_inner=None, device=None, dtype=None):
    inner_dim = d_inner if d_inner is not None else 4 * d_model
    return partial(Mlp, hidden_features=inner_dim,
                   activation=partial(F.gelu, approximate="tanh"),
                   device=device, dtype=dtype)


def create_block(d_model, d_inner=None, layer=None, attn_layer_idx=None,
                 attn_cfg=None, layer_norm_epsilon=1e-5, resid_dropout1=0.0,
                 resid_dropout2=0.0, residual_in_fp32=False, layer_idx=None,
                 device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = create_mixer_cls(layer=layer, attn_layer_idx=attn_layer_idx,
                                 attn_cfg=attn_cfg, layer_idx=layer_idx, **factory_kwargs)
    mlp_cls = create_mlp_cls(d_model, d_inner=d_inner, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm, eps=layer_norm_epsilon, **factory_kwargs)
    block = Block(d_model, mixer_cls, mlp_cls, norm_cls=norm_cls,
                  prenorm=True, resid_dropout1=resid_dropout1,
                  resid_dropout2=resid_dropout2, residual_in_fp32=residual_in_fp32)
    block.layer_idx = layer_idx
    return block


def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True,
                  glu_act=False):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))


class GPT2Embeddings(nn.Module):
    def __init__(self, embed_dim, vocab_size, max_position_embeddings,
                 padding_idx=None, word_embed_proj_dim=None, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if word_embed_proj_dim is None:
            self.word_embeddings = nn.Embedding(vocab_size, embed_dim,
                                                padding_idx=padding_idx, **factory_kwargs)
            self.project_in = None
        else:
            self.word_embeddings = nn.Embedding(vocab_size, word_embed_proj_dim,
                                                padding_idx=padding_idx, **factory_kwargs)
            self.project_in = nn.Linear(word_embed_proj_dim, embed_dim,
                                        bias=False, **factory_kwargs)
        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(max_position_embeddings, embed_dim,
                                                    **factory_kwargs)

    def forward(self, input_ids, position_ids=None):
        batch_size, seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        if self.project_in is not None:
            embeddings = self.project_in(embeddings)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            embeddings = embeddings + self.position_embeddings(position_ids)
        return embeddings


class LMBackbone(nn.Module):
    def __init__(self, d_model, n_layer, d_inner, vocab_size, process_group=None,
                 layer=None, attn_layer_idx=None, attn_cfg=None, max_position_embeddings=0,
                 resid_dropout=0.0, embed_dropout=0.1, layer_norm_epsilon=1e-5,
                 initializer_cfg=None, residual_in_fp32=False, device=None, dtype=None, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.embeddings = GPT2Embeddings(d_model, vocab_size, max_position_embeddings,
                                         **factory_kwargs)
        self.layers = nn.ModuleList([
            create_block(d_model, d_inner=d_inner, layer=layer,
                         attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg,
                         layer_norm_epsilon=layer_norm_epsilon,
                         resid_dropout1=embed_dropout if i == 0 else resid_dropout,
                         resid_dropout2=resid_dropout, residual_in_fp32=residual_in_fp32,
                         layer_idx=i, **factory_kwargs)
            for i in range(n_layer)
        ])
        self.drop_f = nn.Dropout(resid_dropout)
        self.ln_f = nn.LayerNorm(d_model, eps=layer_norm_epsilon, **factory_kwargs)
        self.apply(partial(_init_weights, n_layer=n_layer,
                           **(initializer_cfg or {})))

    def forward(self, input_ids, position_ids=None):
        hidden_states = self.embeddings(input_ids, position_ids=position_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        dropped = self.drop_f(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        return self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))


class HyenaDNAClassifier(nn.Module):
    def __init__(self, d_model, n_layer, d_inner, vocab_size, n_classes, tokenizer,
                 layer=None, attn_layer_idx=None, attn_cfg=None,
                 max_position_embeddings=0, resid_dropout=0.0, embed_dropout=0.1,
                 layer_norm_epsilon=1e-5, initializer_cfg=None, residual_in_fp32=False,
                 pad_vocab_size_multiple=1, clf_dropout=0.3,
                 device=None, dtype=None, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)

        if layer and "d_model" not in layer:
            layer["d_model"] = d_model

        self.token_ids = tokenizer.get_vocab()

        self.backbone = LMBackbone(
            d_model=d_model, n_layer=n_layer, d_inner=d_inner, vocab_size=vocab_size,
            layer=layer, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg,
            max_position_embeddings=max_position_embeddings,
            resid_dropout=resid_dropout, embed_dropout=embed_dropout,
            layer_norm_epsilon=layer_norm_epsilon, initializer_cfg=initializer_cfg,
            residual_in_fp32=residual_in_fp32, **factory_kwargs, **kwargs,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4,
        )

        self.classifier = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(clf_dropout),
            nn.Linear(d_model, n_classes),
        )

        self.apply(partial(_init_weights, n_layer=n_layer, **(initializer_cfg or {})))

    def forward(self, input_ids, position_ids=None):
        hidden_states = self.backbone(input_ids, position_ids=position_ids)

        # убираем [CLS] и [SEP]
        hidden_states = hidden_states[:, 1:-1, :]
        input_ids = input_ids[:, 1:-1]

        B, L, D = hidden_states.shape

        A_id = self.token_ids["A"]
        C_id = self.token_ids["C"]
        G_id = self.token_ids["G"]
        T_id = self.token_ids["T"]

        pooled_list = []

        for token_id in [A_id, C_id, G_id, T_id]:
            mask = (input_ids == token_id).unsqueeze(-1)  # (B, L, 1)

            count = mask.sum(dim=1).clamp(min=1)

            summed = (hidden_states * mask).sum(dim=1)  # (B, D)
            mean_vec = summed / count  # (B, D)

            pooled_list.append(mean_vec)

        # (B, 4 * d_model)
        # pooled = torch.cat(pooled_list, dim=-1)
        #
        # return self.classifier(pooled)

        pooled = torch.stack(pooled_list, dim=1)  # (B, 4, d_model)
        x = self.transformer_encoder(pooled)  # (B, 4, d_model)
        x = x.reshape(x.size(0), -1)  # (B, 4 * d_model)
        return self.classifier(x)


def inject_substring(orig_str):
    modified = re.sub(r"\.mixer", ".mixer.layer", orig_str)
    modified = re.sub(r"\.mlp", ".mlp.layer", modified)
    return modified


def load_pretrained_backbone(model: HyenaDNAClassifier, pretrained_dict: dict,
                             checkpointing: bool = False) -> HyenaDNAClassifier:
    """Загружает веса backbone из pretrained checkpoint, голова остаётся случайной."""
    scratch_dict = model.state_dict()
    for key in scratch_dict:
        if "backbone" in key:
            key_loaded = "model." + key
            if checkpointing:
                key_loaded = inject_substring(key_loaded)
            if key_loaded in pretrained_dict:
                scratch_dict[key] = pretrained_dict[key_loaded]
            else:
                print(f"[WARN] Key not found in checkpoint: {key_loaded}")
    model.load_state_dict(scratch_dict)
    return model


def download_and_load_model(checkpoints_dir: str, model_name: str,
                            model: HyenaDNAClassifier, device: str = "cpu") -> HyenaDNAClassifier:
    model_path = os.path.join(checkpoints_dir, model_name)
    if not os.path.isdir(model_path):
        print(f"Скачиваем {model_name} с HuggingFace...")
        hf_url = f"https://huggingface.co/LongSafari/{model_name}"
        subprocess.run(f"mkdir -p {checkpoints_dir} && cd {checkpoints_dir} "
                       f"&& git lfs install && git clone {hf_url}", shell=True)

    config = json.load(open(os.path.join(model_path, "config.json")))
    checkpointing = config.get("checkpoint_mixer", False)

    _orig_load = torch.load

    def _load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)

    torch.load = _load
    ckpt = torch.load(os.path.join(model_path, "weights.ckpt"), map_location=device)
    torch.load = _orig_load

    model = load_pretrained_backbone(model, ckpt["state_dict"], checkpointing=checkpointing)
    print("Претренированные веса backbone загружены.")
    return model


class DNASequenceDataset(Dataset):
    """
    Читает .fasta и соответствующий .json с метаинформацией.
    Токенизирует на лету (без кеширования на диск).

    Длинные последовательности обрезаются до max_length.
    allowed_ids — множество ID для фильтрации (train или test).
    """

    def __init__(self, fasta_path: str, hierarchy_root: str, meta: dict, label_field: str,
                 tokenizer: CharacterTokenizer, max_length: int,
                 label_encoder: Optional[LabelEncoder] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length

        records = {}
        for rec in SeqIO.parse(fasta_path, "fasta"):
            name = rec.description.split("\t")[0].split()[0]
            records[name] = str(rec.seq).upper()

        self.samples = []
        labels_raw = []

        hierarchy_root = hierarchy_root.split("\t")
        for path_part in hierarchy_root:
            if not path_part:
                break
            meta = meta[path_part]["subs"]

        for class_type in meta.keys():
            for item in meta[class_type]["sequences"]:
                if item in records:
                    self.samples.append((item, records[item]))
                    labels_raw.append(class_type)

        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(labels_raw)
        else:
            self.label_encoder = label_encoder

        self.labels = self.label_encoder.transform(labels_raw)
        self.n_classes = len(self.label_encoder.classes_)

        print(f"Датасет: {len(self.samples)} последовательностей, "
              f"{self.n_classes} классов: {list(self.label_encoder.classes_)}")

        meta_ids = set()
        for class_type in meta.keys():
            for item in meta[class_type]["sequences"]:
                meta_ids.add(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, seq = self.samples[idx]
        label = self.labels[idx]

        seq = seq[: self.max_length]

        token_ids = self.tokenizer(seq)["input_ids"]
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def collate_fn(batch, pad_token_id: int):
    """Паддинг до длины самой длинной последовательности в батче."""
    seqs, labels = zip(*batch)
    max_len = max(s.size(0) for s in seqs)
    padded = torch.full((len(seqs), max_len), pad_token_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : s.size(0)] = s
    return padded, torch.stack(labels)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for input_ids, labels in tqdm(loader, desc="train" if train else "eval", leave=False):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += len(labels)

    return total_loss / total, correct / total


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")

    # — Токенайзер —
    tokenizer = CharacterTokenizer(
        characters=["A", "C", "G", "T", "N"],
        model_max_length=args.max_length,
    )

    # — Метаинформация —
    with open(args.meta_train) as f:
        meta_train = json.load(f)

    with open(args.meta_test) as f:
        meta_test = json.load(f)

    # — Загрузка IDs из файлов —
    # with open(args.train_ids) as f:
    #     train_ids = set(line.strip() for line in f if line.strip())
    # with open(args.test_ids) as f:
    #     test_ids = set(line.strip() for line in f if line.strip())

    # print(f"Train IDs: {len(train_ids)}, Test IDs: {len(test_ids)}")

    # — Датасеты —
    train_ds = DNASequenceDataset(
        fasta_path=args.fasta,
        meta=meta_train,
        hierarchy_root=args.hierarchy_root,
        label_field=args.label_field,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    n_classes = train_ds.n_classes

    # — Распределение классов в train —
    import numpy as np

    class_counts = np.bincount(train_ds.labels)
    class_names = train_ds.label_encoder.classes_

    print("\n— Распределение классов (train) —")
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        print(f"{name:30s}: {count}")

    test_ds = DNASequenceDataset(
        fasta_path=args.fasta,
        meta=meta_test,
        hierarchy_root=args.hierarchy_root,
        label_field=args.label_field,
        tokenizer=tokenizer,
        max_length=args.max_length,
        label_encoder=train_ds.label_encoder,
    )

    train_sample_ids = {name for name, _ in train_ds.samples}
    test_sample_ids = {name for name, _ in test_ds.samples}

    overlap_ds = train_sample_ids & test_sample_ids
    print(f"Пересечение в датасетах: {len(overlap_ds)}")
    print("Примеры:", list(sorted(overlap_ds))[:20])

    pad_id = tokenizer._vocab_str_to_int["[PAD]"]
    _collate = partial(collate_fn, pad_token_id=pad_id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=_collate, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True,
                             collate_fn=_collate, num_workers=2, pin_memory=True)

    # — Конфиг модели —
    model_path = os.path.join(args.checkpoints_dir, args.model_name)
    if not os.path.isdir(model_path):
        subprocess.run(
            f"mkdir -p {args.checkpoints_dir} && cd {args.checkpoints_dir} "
            f"&& git lfs install && git clone https://huggingface.co/LongSafari/{args.model_name}",
            shell=True,
        )
    config = json.load(open(os.path.join(model_path, "config.json")))

    # — Модель —
    model = HyenaDNAClassifier(
        n_classes=n_classes,
        clf_dropout=0.3,
        tokenizer=tokenizer,
        **config,
    )
    model = download_and_load_model(args.checkpoints_dir, args.model_name, model, device)
    model = model.to(device)

    # — Опциональная заморозка backbone —
    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("Backbone заморожен. Обучается только голова классификатора.")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Обучаемых параметров: {trainable_params:,}")
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=1e-2,
        )
    else:
        # Дифференцированные learning rates: backbone медленнее, голова быстрее
        backbone_params = list(model.backbone.parameters())
        head_params = list(model.classifier.parameters())
        optimizer = AdamW([
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": head_params, "lr": args.lr},
        ], weight_decay=1e-2)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Взвешенный loss для дисбаланса классов
    class_counts = np.bincount(train_ds.labels)
    class_weights = torch.tensor(1.0 / (class_counts + 1e-6), dtype=torch.float32).to(device)
    class_weights = class_weights / class_weights.sum()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # — Цикл обучения —
    best_train_acc = 0.0
    best_model_path = "model.pt"

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f}")

        if tr_acc > best_train_acc:
            best_train_acc = tr_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "label_encoder_classes": train_ds.label_encoder.classes_.tolist(),
                "train_acc": tr_acc,
            }, best_model_path)
            print(f"  ✓ Лучшая модель сохранена (train_acc={tr_acc:.4f})")

    # — Финальная оценка на тесте —
    print("\n— Тест —")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for input_ids, labels in test_loader:
            preds = model(input_ids.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(labels.numpy())

    print(classification_report(
        all_true, all_preds,
        target_names=train_ds.label_encoder.classes_,
        zero_division=0,
    ))
    cm = confusion_matrix(all_true, all_preds)

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=train_ds.label_encoder.classes_,
        yticklabels=train_ds.label_encoder.classes_,
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test)")

    plt.tight_layout()
    plt.savefig("confusion.pdf")
    plt.close()

    print("Confusion matrix сохранена в confusion.pdf")

