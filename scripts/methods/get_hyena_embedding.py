import json
import math
import os
import pickle
import re
import subprocess
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torchvision.ops import StochasticDepth
from tqdm import tqdm
from transformers import PreTrainedModel
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def fftconv(u, k, D):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]
    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)


class OptimModule(nn.Module):
    def register(self, name, tensor, lr=None, wd=0.0):
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {}
            if lr is not None:
                optim["lr"] = lr
            if wd is not None:
                optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)


class Sin(nn.Module):
    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim)) if train_freq else w * torch.ones(1, dim)

    def forward(self, x):
        return torch.sin(self.freq * x)


class PositionalEmbedding(OptimModule):
    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float = 1e-5):
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
    def __init__(
        self,
        d_model,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        modulation_lr=0.0,
        modulate=True,
        shift=0.05,
        **kwargs,
    ):
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
    def __init__(
        self,
        d_model,
        emb_dim=3,
        order=16,
        fused_fft_conv=False,
        seq_len=1024,
        lr=1e-3,
        lr_pos_emb=1e-5,
        dropout=0.0,
        w=1,
        wd=0,
        bias=True,
        num_inner_mlps=2,
        normalized=False,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_bias = bias
        self.fused_fft_conv = fused_fft_conv
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)
        act = Sin(dim=order, w=w)
        self.emb_dim = emb_dim
        assert emb_dim % 2 != 0 and emb_dim >= 3
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
            for name, _ in c.state_dict().items():
                setattr(getattr(c, name), "_optim", {"weight_decay": wd, "lr": lr})

    def filter(self, L):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)
        return h

    def forward(self, x, L, k=None, bias=None):
        if k is None:
            k = self.filter(L)
        if isinstance(k, tuple):
            k = k[0]
        return fftconv(x, k, bias)


class HyenaOperator(nn.Module):
    def __init__(
        self,
        d_model,
        l_max,
        order=2,
        filter_order=64,
        dropout=0.0,
        filter_dropout=0.0,
        **filter_args,
    ):
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
            d_model * (order - 1),
            order=filter_order,
            seq_len=l_max,
            channels=1,
            dropout=filter_dropout,
            **filter_args,
        )

    def forward(self, u):
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


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, activation=F.gelu, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, **factory_kwargs)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        mlp_cls,
        norm_cls=nn.LayerNorm,
        dropout_cls=nn.Dropout,
        resid_dropout1=0.0,
        resid_dropout2=0.0,
        drop_path1=0.0,
        drop_path2=0.0,
        residual_in_fp32=False,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.mixer = mixer_cls()
        self.dropout1 = dropout_cls(resid_dropout1)
        self.drop_path1 = StochasticDepth(drop_path1, mode="row")
        self.norm1 = norm_cls(dim)
        self.mlp = mlp_cls(dim)
        self.dropout2 = dropout_cls(resid_dropout2)
        self.drop_path2 = StochasticDepth(drop_path2, mode="row")
        self.norm2 = norm_cls(dim)

    def forward(self, hidden_states, residual=None):
        dropped = self.drop_path1(self.dropout1(hidden_states))
        residual = dropped if residual is None else dropped + residual
        hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.mixer(hidden_states)
        dropped = self.drop_path2(self.dropout2(hidden_states))
        residual = dropped if residual is None else dropped + residual
        hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


def create_mlp_cls(d_model, d_inner=None, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    inner_dim = d_inner if d_inner is not None else 4 * d_model
    return partial(Mlp, hidden_features=inner_dim, activation=partial(F.gelu, approximate="tanh"), **factory_kwargs)


def create_block(
    d_model,
    d_inner=None,
    layer=None,
    layer_norm_epsilon=1e-5,
    resid_dropout1=0.0,
    resid_dropout2=0.0,
    residual_in_fp32=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(HyenaOperator, **layer)
    mlp_cls = create_mlp_cls(d_model, d_inner=d_inner, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm, eps=layer_norm_epsilon, **factory_kwargs)
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        resid_dropout1=resid_dropout1,
        resid_dropout2=resid_dropout2,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True, glu_act=False):
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
            elif name in ["output_linear.0.weight"]:
                if not glu_act:
                    nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))
                else:
                    out_features = p.shape[0]
                    nn.init.normal_(p[: out_features // 2], mean=0.0, std=initializer_range / math.sqrt(2 * n_layer) * 2)


class GPT2Embeddings(nn.Module):
    def __init__(self, embed_dim, vocab_size, max_position_embeddings, padding_idx=None, word_embed_proj_dim=None, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if word_embed_proj_dim is None:
            self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx, **factory_kwargs)
            self.project_in = None
        else:
            self.word_embeddings = nn.Embedding(vocab_size, word_embed_proj_dim, padding_idx=padding_idx, **factory_kwargs)
            self.project_in = nn.Linear(word_embed_proj_dim, embed_dim, bias=False, **factory_kwargs)
        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(max_position_embeddings, embed_dim, **factory_kwargs)

    def forward(self, input_ids, position_ids=None):
        _, seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        if self.project_in is not None:
            embeddings = self.project_in(embeddings)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            embeddings = embeddings + self.position_embeddings(position_ids)
        return embeddings


class LMBackbone(nn.Module):
    def __init__(
        self,
        d_model,
        n_layer,
        d_inner,
        vocab_size,
        layer=None,
        max_position_embeddings=0,
        resid_dropout=0.0,
        embed_dropout=0.1,
        layer_norm_epsilon=1e-5,
        initializer_cfg=None,
        residual_in_fp32=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.embeddings = GPT2Embeddings(d_model, vocab_size, max_position_embeddings, **factory_kwargs)
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_inner=d_inner,
                    layer=layer,
                    layer_norm_epsilon=layer_norm_epsilon,
                    resid_dropout1=embed_dropout if i == 0 else resid_dropout,
                    resid_dropout2=resid_dropout,
                    residual_in_fp32=residual_in_fp32,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )
        self.drop_f = nn.Dropout(resid_dropout)
        self.ln_f = nn.LayerNorm(d_model, eps=layer_norm_epsilon, **factory_kwargs)
        self.apply(partial(_init_weights, n_layer=n_layer, **(initializer_cfg if initializer_cfg is not None else {})))

    def forward(self, input_ids, position_ids=None):
        hidden_states = self.embeddings(input_ids, position_ids=position_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        dropped = self.drop_f(hidden_states)
        residual = dropped if residual is None else dropped + residual
        return self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))


class HyenaDNAModel(nn.Module):
    def __init__(
        self,
        d_model,
        n_layer,
        d_inner,
        vocab_size,
        layer=None,
        max_position_embeddings=0,
        resid_dropout=0.0,
        embed_dropout=0.1,
        layer_norm_epsilon=1e-5,
        initializer_cfg=None,
        residual_in_fp32=False,
        pad_vocab_size_multiple=1,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        if "d_model" not in layer:
            layer["d_model"] = d_model
        self.backbone = LMBackbone(
            d_model=d_model,
            n_layer=n_layer,
            d_inner=d_inner,
            vocab_size=vocab_size,
            layer=layer,
            max_position_embeddings=max_position_embeddings,
            resid_dropout=resid_dropout,
            embed_dropout=embed_dropout,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_cfg=initializer_cfg,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
            **kwargs,
        )
        self.apply(partial(_init_weights, n_layer=n_layer, **(initializer_cfg if initializer_cfg is not None else {})))

    def forward(self, input_ids, position_ids=None, state=None):
        return self.backbone(input_ids, position_ids=position_ids)


class CharacterTokenizer(PreTrainedTokenizer):
    def __init__(self, characters: Sequence[str], model_max_length: int, padding_side: str = "left", **kwargs):
        self.characters = list(characters)
        self.model_max_length = model_max_length
        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[RESERVED]": 5,
            "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(self.characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        eos_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)
        mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._vocab_str_to_int)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def __len__(self):
        return self.vocab_size

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str.get(index, "[UNK]")

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        result = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        if token_ids_1 is not None:
            result += token_ids_1 + [self.sep_token_id]
        return result

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )
        result = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        result = len([self.cls_token_id] + token_ids_0 + [self.sep_token_id]) * [0]
        if token_ids_1 is not None:
            result += len(token_ids_1 + [self.sep_token_id]) * [1]
        return result

    def get_config(self) -> Dict:
        return {"char_ords": [ord(ch) for ch in self.characters], "model_max_length": self.model_max_length}

    @classmethod
    def from_config(cls, config: Dict):
        return cls(characters=[chr(i) for i in config["char_ords"]], model_max_length=config["model_max_length"])

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file, "w") as f:
            json.dump(self.get_config(), f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file) as f:
            cfg = json.load(f)
        return cls.from_config(cfg)


def inject_substring(orig_str):
    modified_string = re.sub(r"\.mixer", ".mixer.layer", orig_str)
    modified_string = re.sub(r"\.mlp", ".mlp.layer", modified_string)
    return modified_string


def load_weights(scratch_dict, pretrained_dict, checkpointing=False):
    for key in scratch_dict:
        if "backbone" in key:
            key_loaded = "model." + key
            if checkpointing:
                key_loaded = inject_substring(key_loaded)
            if key_loaded not in pretrained_dict:
                raise KeyError(f"Missing key: {key_loaded}")
            scratch_dict[key] = pretrained_dict[key_loaded]
    return scratch_dict


class HyenaDNAPreTrainedModel(PreTrainedModel):
    base_model_prefix = "hyenadna"

    def __init__(self, config):
        pass

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        path,
        model_name,
        download=False,
        config=None,
        device="cpu",
    ):
        pretrained_model_name_or_path = os.path.join(path, model_name)
        if os.path.isdir(pretrained_model_name_or_path) and not download:
            if config is None:
                config = json.load(open(os.path.join(pretrained_model_name_or_path, "config.json")))
        else:
            hf_url = f"https://huggingface.co/LongSafari/{model_name}"
            subprocess.run(f"rm -rf {pretrained_model_name_or_path}", shell=True, check=False)
            subprocess.run(
                f"mkdir -p {path} && cd {path} && git lfs install && git clone {hf_url}",
                shell=True,
                check=True,
            )
            if config is None:
                config = json.load(open(os.path.join(pretrained_model_name_or_path, "config.json")))
        scratch_model = HyenaDNAModel(**config)
        loaded_ckpt = torch.load(
            os.path.join(pretrained_model_name_or_path, "weights.ckpt"),
            map_location=torch.device(device),
        )
        checkpointing = bool(config.get("checkpoint_mixer", False))
        state_dict = load_weights(scratch_model.state_dict(), loaded_ckpt["state_dict"], checkpointing=checkpointing)
        scratch_model.load_state_dict(state_dict)
        return scratch_model


def compute_base_embedding(sequence: str, tokenizer, model, device: str) -> Tensor:
    tokenized = tokenizer(sequence)["input_ids"]
    tok_tensor = torch.LongTensor(tokenized).unsqueeze(0).to(device)
    with torch.inference_mode():
        outputs = model(tok_tensor)[0][1:-1]
        if outputs.size(0) != len(sequence):
            raise ValueError("Token/output length mismatch")
        base_to_indices = {"A": [], "C": [], "G": [], "T": []}
        for i, nucl in enumerate(sequence):
            if nucl in base_to_indices:
                base_to_indices[nucl].append(outputs[i])
        for key in base_to_indices:
            if base_to_indices[key]:
                base_to_indices[key] = torch.stack(base_to_indices[key]).mean(dim=0)
            else:
                base_to_indices[key] = torch.zeros(outputs.size(1), device=outputs.device)
        return torch.cat(
            [base_to_indices["A"], base_to_indices["C"], base_to_indices["G"], base_to_indices["T"]],
            dim=0,
        )


def process_fasta(fasta_path: str, tokenizer, model, device: str):
    name_to_embedding = {}
    with open(fasta_path, "r") as f:
        name = None
        sequence_chunks = []
        for line in tqdm(f):
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    sequence = "".join(sequence_chunks).upper()
                    name_to_embedding[name] = compute_base_embedding(sequence, tokenizer, model, device)
                name = line[1:].split("\t")[0]
                sequence_chunks = []
            else:
                sequence_chunks.append(line)
        if name is not None:
            sequence = "".join(sequence_chunks).upper()
            name_to_embedding[name] = compute_base_embedding(sequence, tokenizer, model, device)
    return name_to_embedding


def get_hyena_embeddings(fasta_path: str, output_dir: str):
    _orig_load = torch.load
    def _load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)
    torch.load = _load

    pretrained_model_name = "hyenadna-small-32k-seqlen"
    max_length = 32_000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HyenaDNAPreTrainedModel.from_pretrained("checkpoints", pretrained_model_name, download=True, device=device)

    torch.load = _orig_load

    tokenizer = CharacterTokenizer(
        characters=["A", "C", "G", "T", "N"],
        model_max_length=max_length,
    )

    model.to(device)
    model.eval()

    name_to_embedding = process_fasta(fasta_path, tokenizer, model, device)

    path_to_hyena_embedding = f"{output_dir}/hyena_embeddings.pkl"
    os.makedirs(output_dir, exist_ok=True)
    with open(path_to_hyena_embedding, "wb") as f:
        pickle.dump({"embeddings": name_to_embedding}, f)
