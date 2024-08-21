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
    decoded_tokens_KL: list[list[str]], activations_KL: torch.Tensor, top_n: int = 5
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
            else:
                highlighted_tokens.append(token)

        result.append(highlighted_tokens)

    return result


# def collect_activating_inputs(sweep_name: str, submodule_trainers, dictionaries_path: str, n_inputs: int)

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
