import config

import os
from Bio import SeqIO
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import re
import warnings
import os
from hyena_dna.standalone_hyenadna import HyenaDNAModel, CharacterTokenizer
import torch
import pickle

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

from n02_scripts.n12_seq_from_fasta import seq_from_fasta


#@title Huggingface Pretrained Wrapper
# for Huggingface integration, we use a wrapper class around the model
# to load weights
import json
import os
import subprocess
import transformers
from transformers import PreTrainedModel, AutoModelForCausalLM, PretrainedConfig
import re

def inject_substring(orig_str):
    """Hack to handle matching keys between models trained with and without
    gradient checkpointing."""

    # modify for mixer keys
    pattern = r"\.mixer"
    injection = ".mixer.layer"

    modified_string = re.sub(pattern, injection, orig_str)

    # modify for mlp keys
    pattern = r"\.mlp"
    injection = ".mlp.layer"

    modified_string = re.sub(pattern, injection, modified_string)

    return modified_string

def load_weights(scratch_dict, pretrained_dict, checkpointing=False):
    """Loads pretrained (backbone only) weights into the scratch state dict.

    scratch_dict: dict, a state dict from a newly initialized HyenaDNA model
    pretrained_dict: dict, a state dict from the pretrained ckpt
    checkpointing: bool, whether the gradient checkpoint flag was used in the
    pretrained model ckpt. This slightly changes state dict keys, so we patch
    that if used.

    return:
    dict, a state dict with the pretrained weights loaded (head is scratch)

    # loop thru state dict of scratch
    # find the corresponding weights in the loaded model, and set it

    """

    # need to do some state dict "surgery"
    for key, value in scratch_dict.items():
        if 'backbone' in key:
            # the state dicts differ by one prefix, '.model', so we add that
            key_loaded = 'model.' + key
            # breakpoint()
            # need to add an extra ".layer" in key
            if checkpointing:
                key_loaded = inject_substring(key_loaded)
            try:
                scratch_dict[key] = pretrained_dict[key_loaded]
            except:
                raise Exception('key mismatch in the state dicts!')

    # scratch_dict has been updated
    return scratch_dict

class HyenaDNAPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    base_model_prefix = "hyenadna"

    def __init__(self, config):
        pass

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    @classmethod
    def from_pretrained(cls,
                        path,
                        model_name,
                        download=False,
                        config=None,
                        device='cpu',
                        use_head=False,
                        n_classes=2,
                      ):
        # first check if it is a local path
        pretrained_model_name_or_path = os.path.join(path, model_name)
        if os.path.isdir(pretrained_model_name_or_path) and download == False:
            if config is None:
                config = json.load(open(os.path.join(pretrained_model_name_or_path, 'config.json')))
        else:
            hf_url = f'https://huggingface.co/LongSafari/{model_name}'

            subprocess.run(f'rm -rf {pretrained_model_name_or_path}', shell=True)
            command = f'mkdir -p {path} && cd {path} && git lfs install && git clone {hf_url}'
            subprocess.run(command, shell=True)

            if config is None:
                config = json.load(open(os.path.join(pretrained_model_name_or_path, 'config.json')))

        scratch_model = HyenaDNAModel(**config, use_head=use_head, n_classes=n_classes)  # the new model format
        loaded_ckpt = torch.load(
            os.path.join(pretrained_model_name_or_path, 'weights.ckpt'),
            map_location=torch.device(device)
        )

        # need to load weights slightly different if using gradient checkpointing
        if config.get("checkpoint_mixer", False):
            checkpointing = config["checkpoint_mixer"] == True or config["checkpoint_mixer"] == True
        else:
            checkpointing = False

        # grab state dict from both and load weights
        state_dict = load_weights(scratch_model.state_dict(), loaded_ckpt['state_dict'], checkpointing=checkpointing)

        # scratch model has now been updated
        scratch_model.load_state_dict(state_dict)
        print("Loaded pretrained weights ok!")
        return scratch_model

# instantiate pretrained model
pretrained_model_name = 'hyenadna-small-32k-seqlen'
max_length = 32_000

model = HyenaDNAPreTrainedModel.from_pretrained(
    'checkpoints',
    pretrained_model_name,
    download=True
)

tokenizer = CharacterTokenizer(
    characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters
    model_max_length=max_length,
)



model.to("cpu")
model.eval()

name_to_embedding = {}
name_to_type = {}

path_to_df_info = os.path.join(config.DIR_INCEST_MANY, 'repbase_orf_type.txt')
df_info = pd.read_csv(path_to_df_info, sep='\t')
df_info = df_info[df_info['Good'] == 1]
fasta_path = os.path.join(config.DIR_INCEST_MANY, 'repbase.fasta')

# name_to_sequence = seq_from_fasta(fasta_path, list(df_info['name']))


# n = 3
names = ['Helitron-N2C_CGi_Helitron_Crassostrea_gigas', 'ATCOPIA97I_Copia_Arabidopsis_thaliana', 'L1-9_LCh_L1_Latimeria_chalumnae']

rows = df_info[df_info['name'].isin(names)]
types = rows['MainType'].tolist()
sequences = [name_to_sequence[name].upper() for name in names]

tokenized = [tokenizer(sequence)["input_ids"] for sequence in sequences]
tok_tensors = [torch.LongTensor(tokens).unsqueeze(0).to("cpu") for tokens in tokenized]

outputs_list = []

for i in range(len(names)):
    with torch.inference_mode():
        outputs = model(tok_tensors[i])  # (1, seq_len, hidden_dim)
        outputs_list.append(outputs)
        embedding1 = outputs.mean(dim=1).squeeze(0).cpu().numpy()  # среднее по токенам

path_to_save = os.path.join(config.DIR_INCEST_MANY, f'hyena_model_output_correct_examples.pkl')
to_save = {'names': names, 'types': types, 'sequences': sequences, 'outputs': outputs_list}
with open(path_to_save, "wb") as f:
    pickle.dump(to_save, f)

