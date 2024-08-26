import matplotlib.pyplot as plt
import json
import torch
import pickle
from typing import Optional
from matplotlib.colors import Normalize
import numpy as np
import os
import random
import datasets

import einops
import dictionary_learning.interp as interp
from circuitsvis.activations import text_neuron_activations
from collections import namedtuple
from nnsight import LanguageModel
from tqdm import tqdm

import experiments.utils as utils
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.buffer import ActivationBuffer

DEBUGGING = True

if DEBUGGING:
    tracer_kwargs = dict(scan=True, validate=True)
else:
    tracer_kwargs = dict(scan=False, validate=False)


def highlight_top_activations(
    decoded_tokens_KL: list[list[str]], activations_KL: torch.Tensor, top_n: int = 5, include_activations: bool = False
) -> list[list[str]]:
    assert len(decoded_tokens_KL) == activations_KL.shape[0], "Number of sequences must match"

    result = []
    for tokens, activations in zip(decoded_tokens_KL, activations_KL):
        # Get indices of top activations
        nonzero_activations = activations[activations != 0]
        top_indices = torch.argsort(activations, descending=True)[
            : min(top_n, len(nonzero_activations))
        ]

        # Create a new list of tokens with highlights
        highlighted_tokens = []
        for i, token in enumerate(tokens):
            if i in top_indices:
                highlighted_tokens.append(f" <<{token}>>")
                if include_activations:
                    highlighted_tokens.append(f"({activations[i].item():.0f})")
            else:
                highlighted_tokens.append(token)

        result.append("".join(highlighted_tokens))

    return result

def compute_dla(feat_indices: torch.Tensor, sae_decoder: torch.Tensor, unembed: torch.Tensor, return_topk_tokens: int) -> torch.Tensor:
    """
    Compute direct logit attribution for a given set of features
    """
    assert len(feat_indices.shape) == 1
    assert len(sae_decoder.shape) == 2
    assert len(unembed.shape) == 2
    assert sae_decoder.shape[0] == unembed.shape[1]

    # Select features from the decoder
    W_dec = sae_decoder[:, feat_indices]
    W_dec = W_dec.to(unembed.dtype)

    # Normalize the decoder and unembed matrices
    W_dec = W_dec / W_dec.norm(dim=0, keepdim=True)
    unembed = unembed / unembed.norm(dim=1, keepdim=True)

    dla = unembed @ W_dec
    _, topk_indices = torch.topk(dla, return_topk_tokens, dim=0)
    return topk_indices.T


def get_max_activating_prompts(
    model,
    submodule,
    tokenized_inputs_bL: list[list[dict]],
    dim_indices: torch.Tensor,
    batch_size: int,
    dictionary=None,
    n_inputs: int = 512,
    k: int = 30,
    context_length: int = 128,
):

    assert n_inputs % batch_size == 0
    assert n_inputs >= len(tokenized_inputs_bL)
    tokenized_inputs_bL = tokenized_inputs_bL[:n_inputs]

    feature_count = dim_indices.shape[0]

    device = model.device

    max_activating_indices_FK = torch.zeros((feature_count, k), device=device, dtype=torch.int)
    max_activations_FK = torch.zeros((feature_count, k), device=device, dtype=torch.float32)
    max_tokens_FKL = torch.zeros((feature_count, k, context_length), device=device, dtype=torch.int)
    max_activations_FKL = torch.zeros((feature_count, k, context_length), device=device, dtype=torch.float32)

    for i, inputs in tqdm(enumerate(tokenized_inputs_bL), total=len(tokenized_inputs_bL)):

        batch_offset = i * batch_size
        inputs_BL = inputs['input_ids']

        with torch.no_grad(), model.trace(inputs, **tracer_kwargs):
            activations_BLD = submodule.output
            if type(activations_BLD.shape) == tuple:
                activations_BLD = activations_BLD[0]
            activations_BLF = dictionary.encode(activations_BLD)
            activations_BLF = activations_BLF[:, :, dim_indices].save()

        activations_FBL = einops.rearrange(activations_BLF.value, 'B L F -> F B L')
        # Use einops to find the max activation per input
        activations_FB = einops.reduce(activations_FBL, 'F B L -> F B', 'max')
        tokens_FBL = einops.repeat(inputs_BL, 'B L -> F B L', F=feature_count)
        
        # Keep track of input indices
        indices_B = torch.arange(batch_offset, batch_offset + batch_size, device=device)
        indices_FB = einops.repeat(indices_B, 'B -> F B', F=feature_count)

        # Concatenate current batch activations and indices with the previous ones
        combined_activations_FB = torch.cat([max_activations_FK, activations_FB], dim=1)
        combined_indices_FB = torch.cat([max_activating_indices_FK, indices_FB], dim=1)
        combined_activations_FBL = torch.cat([max_activations_FKL, activations_FBL], dim=1)
        combined_tokens_FBL = torch.cat([max_tokens_FKL, tokens_FBL], dim=1)

        # Sort and keep top k activations for each dimension
        topk_activations_FK, topk_indices_FK = torch.topk(combined_activations_FB, k, dim=1)
        max_activations_FK = topk_activations_FK

        feature_indices_F1 = torch.arange(feature_count, device=device)[:, None]
        max_activating_indices_FK = combined_indices_FB[feature_indices_F1, topk_indices_FK]
        max_activations_FKL = combined_activations_FBL[feature_indices_F1, topk_indices_FK]
        max_tokens_FKL = combined_tokens_FBL[feature_indices_F1, topk_indices_FK]
            

    return max_tokens_FKL, max_activations_FKL

def evaluate_binary_llm_output(llm_outputs):
    decisions = torch.ones(len(llm_outputs), dtype=torch.int) * -2
    for i, llm_out in enumerate(llm_outputs):
        llm_out = llm_out[0].text.lower()[-10:]
        if 'yes' in llm_out and 'no' in llm_out:
            decisions[i] = -1
        elif 'yes' in llm_out:
            decisions[i] = 1
        elif 'no' in llm_out:
            decisions[i] = 0
        else:
            decisions[i] = -1
    return decisions

# loading model, data, dictionaries --> dict, feature: yes/no gender

# per feature
    # def collect_activating_inputs(dataset: list, dictionary: Any, feature_idxs: list, n_inputs: int)
    # store as a file

    # collect_direct_logit_attributions

# build prompts

# call LLM (batched)

# build a test set

# scale to classes



# if __name__ == "__main__":


#     trainer_ids = [2, 6, 10, 14, 18]
#     ae_sweep_paths = {
#         "pythia70m_sweep_standard_ctx128_0712": {
#             #     # "resid_post_layer_0": {"trainer_ids": None},
#             #     # "resid_post_layer_1": {"trainer_ids": None},
#             #     # "resid_post_layer_2": {"trainer_ids": None},
#             "resid_post_layer_3": {"trainer_ids": trainer_ids},
#             #     "resid_post_layer_4": {"trainer_ids": None},
#         },
#         "pythia70m_sweep_gated_ctx128_0730": {
#             # "resid_post_layer_0": {"trainer_ids": None},
#             # "resid_post_layer_1": {"trainer_ids": None},
#             # "resid_post_layer_2": {"trainer_ids": None},
#             "resid_post_layer_3": {"trainer_ids": trainer_ids},
#             # "resid_post_layer_4": {"trainer_ids": None},
#         },
#         # "pythia70m_sweep_panneal_ctx128_0730": {
#         #     # "resid_post_layer_0": {"trainer_ids": None},
#         #     # "resid_post_layer_1": {"trainer_ids": None},
#         #     # "resid_post_layer_2": {"trainer_ids": None},
#         #     "resid_post_layer_3": {"trainer_ids": trainer_ids},
#         #     # "resid_post_layer_4": {"trainer_ids": None},
#         # },
#         "pythia70m_sweep_topk_ctx128_0730": {
#             # "resid_post_layer_0": {"trainer_ids": None},
#             # "resid_post_layer_1": {"trainer_ids": None},
#             # "resid_post_layer_2": {"trainer_ids": None},
#             "resid_post_layer_3": {"trainer_ids": trainer_ids},
#             # "resid_post_layer_4": {"trainer_ids": None},
#         },
#     }


#     dictionaries_path = "../dictionary_learning/dictionaries"

#     for sweep_name, submodule_trainers in ae_sweep_paths.items():
#         ae_group_paths = utils.get_ae_group_paths(dictionaries_path, sweep_name, submodule_trainers)