#%%
"""
Generate train / test split for known entities
"""
# Imports
import os
import json
from collections import defaultdict
import torch as t
import numpy as np


# Global parameters
threshold_known_accuracy = 0.8
device = 'cuda'

# Dataset loading
HOME_DIR = "/share/u/can/sae_eval/ravel/"
DATA_DIR = os.path.join(HOME_DIR, "data/")
SAE_DIR = os.path.join(HOME_DIR, "saes/")
# MODEL_DIR = os.path.join(HOME_DIR, "models/")

entity_class = "city"
model_name = 'gemma-2-2b'

with open(os.path.join(DATA_DIR, f"base/ravel_{entity_class}_entity_attributes.json")) as f:
    entity_to_attributes = json.load(f)

with open(os.path.join(DATA_DIR, f"base/ravel_{entity_class}_entity_to_split.json")) as f:
    entity_to_split = json.load(f)

with open(os.path.join(DATA_DIR, f'{model_name}/known_entity_attribute_pairs.json')) as f:
    known_entity_attribute_accuriacies = json.load(f)

with open(os.path.join(DATA_DIR, f"base/ravel_{entity_class}_attribute_to_prompts.json")) as f:
    attribute_to_prompts = json.load(f)

# Per attribute class
attribute_class = "Country"
# split_classes = ['train', 'val', 'test']

split_known_entity_attribute_indices = defaultdict(list)

for entity, split in entity_to_split.items(): # could do enumerate here and only save the index
    if known_entity_attribute_accuriacies[entity][attribute_class] > threshold_known_accuracy:
        split_known_entity_attribute_indices[split].append(entity)

#%%
# Load GEMMA 2 2b SAE from neuronpedia

from huggingface_hub import hf_hub_download
path_to_params = hf_hub_download(
    repo_id="google/gemma-scope-2b-pt-res",
    filename="layer_20/width_16k/average_l0_71/params.npz",
    force_download=False,
    cache_dir= os.path.join(SAE_DIR, model_name),
)
params = np.load(path_to_params)
pt_params = {k: t.from_numpy(v).cuda() for k, v in params.items()}

#%%
from dictionary_learning.dictionary import JumpReluAutoEncoder
activation_dim = params['W_enc'].shape[0]
dict_size = params['W_enc'].shape[1]

sae = JumpReluAutoEncoder(
    activation_dim=activation_dim,
    dict_size=dict_size,
)
sae.load_state_dict(pt_params)
sae.to(device)

#%%
# Test the SAE
# Load model
from transformers import AutoModelForCausalLM, AutoTokenizer

with open('/share/u/can/src/hf.txt', 'r') as f:
    hf_token = f.read().strip()

t.set_grad_enabled(False) # avoid blowing up mem

from nnsight import LanguageModel
model = LanguageModel(
    "google/gemma-2-2b",
    device_map=device,
    cache_dir="/share/u/can/sae_eval/ravel/models/gemma-2-2b",
    use_auth_token=hf_token,
    dispatch=True,
)


tokenizer =  AutoTokenizer.from_pretrained(
    "google/gemma-2-2b",
    cache_dir="/share/u/can/sae_eval/ravel/models/gemma-2-2b",
    use_auth_token=hf_token,
)

#%% 

sample_text = "The capital of France is Paris."
tracer_kwargs = {'scan': False, 'validate': False}

layer_idx = 20
submodule = model.model.layers[layer_idx].post_feedforward_layernorm

with model.trace(sample_text):
    act = submodule.output.save()

#%%
latents = sae.encode(act)
print(f'shape: {latents.shape}')
nonzeros = (latents != 0).sum(dim=-1).flatten()
print(f'nonzeros: {nonzeros}')


#%%
base_prompt_templates = attribute_to_prompts[attribute_class]
X = defaultdict(list)
Y = defaultdict(list)

for entity in split_known_entity_attribute_indices['train'][:10]:
    base_prompt = [p % entity for p in base_prompt_templates[:10]]
    labels = [entity_to_attributes[entity][attribute_class]] * len(base_prompt)
    X['train'].extend(base_prompt)
    Y['train'].extend(labels)

with model.trace(X['train']):
    X_act = submodule.output.save() # only do one position here

# options
# mean
# single token
# sparse probing paper

# %%
# Run feature importances on test set
# Use LinearSVC from RAVEL paper

import numpy as np

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import torch


def select_features_with_classifier(featurizer, inputs, labels, coeff=None):
  if coeff is None:
    coeff = [0.1, 10, 100, 1000]
  coeff_to_select_features = {}
  for c in coeff:
    with torch.no_grad():
      X_transformed = featurizer(inputs).cpu().numpy()
      lsvc = LinearSVC(C=c, penalty="l1", dual=False, max_iter=5000,
                       tol=0.01).fit(X_transformed, labels)
      selector = SelectFromModel(lsvc, prefit=True)
      kept_dim = np.where(selector.get_support())[0]
      coeff_to_select_features[c] = kept_dim
  return coeff_to_select_features


coeff_to_kept_dims = select_features_with_classifier(
    sae.encode, X_act, Y['train'])
# %%
