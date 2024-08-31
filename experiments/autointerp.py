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
from transformers import AutoTokenizer

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

### BEGIN OF ACTIVATION COLLECTION FUNCTIONS ###


def get_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
    return 0  # Return 0 if CUDA is not available


MEMORY_DEBUG = False


@torch.no_grad()
def get_max_activating_prompts(
    model,
    submodule,
    tokenized_inputs_bL: list[list[dict]],
    dim_indices: torch.Tensor,
    batch_size: int,
    dictionary,
    context_length: int = 128,
    k: int = 30,
):
    # If encountering memory issues, we could try preallocate tensors instead of concatenating in the for loop
    # If speedup is needed, we could do multiple SAEs at once
    feature_count = dim_indices.shape[0]

    device = model.device

    if MEMORY_DEBUG:
        print(f"Initial memory usage: {get_memory_usage():.2f} MB")

    max_activating_indices_FK = torch.zeros((feature_count, k), device=device, dtype=torch.int32)
    max_activations_FK = torch.zeros((feature_count, k), device=device, dtype=torch.bfloat16)
    max_tokens_FKL = torch.zeros(
        (feature_count, k, context_length), device=device, dtype=torch.int32
    )
    max_activations_FKL = torch.zeros(
        (feature_count, k, context_length), device=device, dtype=torch.bfloat16
    )

    if MEMORY_DEBUG:
        print(f"Initial memory usage: {get_memory_usage():.2f} MB")

    for i, inputs in tqdm(enumerate(tokenized_inputs_bL), total=len(tokenized_inputs_bL)):
        batch_offset = i * batch_size
        inputs_BL = inputs["input_ids"].to(dtype=torch.int32)

        if MEMORY_DEBUG:
            print(f"Memory usage: {get_memory_usage():.2f} MB")

        with torch.no_grad(), model.trace(inputs, **tracer_kwargs):
            activations_BLD = submodule.output
            if type(activations_BLD.shape) == tuple:
                activations_BLD = activations_BLD[0].save()

        activations_BLF = dictionary.encode(activations_BLD.value)
        activations_BLF = activations_BLF[:, :, dim_indices]

        activations_FBL = einops.rearrange(activations_BLF, "B L F -> F B L")
        # Use einops to find the max activation per input
        activations_FB = einops.reduce(activations_FBL, "F B L -> F B", "max")
        tokens_FBL = einops.repeat(inputs_BL, "B L -> F B L", F=feature_count)

        # Keep track of input indices
        indices_B = torch.arange(batch_offset, batch_offset + batch_size, device=device)
        indices_FB = einops.repeat(indices_B, "B -> F B", F=feature_count)

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


def compute_dla(
    feat_indices: torch.Tensor,
    sae_decoder_DF: torch.Tensor,
    unembed_VD: torch.Tensor,
    return_topk_tokens: int,
) -> torch.Tensor:
    """
    Compute direct logit attribution for a given set of features
    """
    assert len(feat_indices.shape) == 1
    assert len(sae_decoder_DF.shape) == 2
    assert len(unembed_VD.shape) == 2
    assert sae_decoder_DF.shape[0] == unembed_VD.shape[1]

    # Select features from the decoder
    W_dec_DF = sae_decoder_DF[:, feat_indices]
    W_dec_DF = W_dec_DF.to(unembed_VD.dtype)

    # Normalize the decoder and unembed matrices
    W_dec_DF = W_dec_DF / W_dec_DF.norm(dim=0, keepdim=True)
    unembed_VD = unembed_VD / unembed_VD.norm(dim=1, keepdim=True)

    dla_VF = unembed_VD @ W_dec_DF
    _, topk_indices_KF = torch.topk(dla_VF, return_topk_tokens, dim=0)
    topk_indices_FK = topk_indices_KF.T
    return topk_indices_FK


def get_autointerp_inputs_for_one_sae(
    model: LanguageModel,
    batched_data: dict,
    autoencoder,
    autoencoder_config: dict,
    submodule,
    batch_size: int,
    top_k_prompts: int,
):
    """ "Design decisions: Don't pretokenize. Detokenizing uses a ton of space with many features. Just tokenize before constructing autointerp prompt."""
    assert batched_data[0]["input_ids"].shape[0] == batch_size, "Batch size must be consistent"

    dict_size = autoencoder_config["trainer"]["dict_size"]
    context_length = autoencoder_config["buffer"]["ctx_len"]

    all_feature_indices = torch.arange(dict_size, device=model.device)

    max_tokens_FKL, max_activations_FKL = get_max_activating_prompts(
        model,
        submodule,
        batched_data,
        all_feature_indices,
        batch_size,
        autoencoder,
        context_length=context_length,
        k=top_k_prompts,
    )

    unembed_VD = utils.get_submodule(model, "unembed", -1).weight

    dla_results_FK = compute_dla(
        all_feature_indices, autoencoder.decoder.weight, unembed_VD, top_k_prompts
    ).to(dtype=torch.int32)

    assert max_tokens_FKL.dtype == torch.int32
    assert max_activations_FKL.dtype == torch.bfloat16

    # TODO: We could maybe reduce max_activations_FKL to unsigned 8-bit integers
    # They are always positive, but maybe look into it first?
    results = {
        "max_tokens_FKL": max_tokens_FKL.to("cpu"),
        "max_activations_FKL": max_activations_FKL.to("cpu"),
        "dla_results_FK": dla_results_FK.to("cpu"),
    }

    return results


def get_autointerp_inputs_for_all_saes(
    model: LanguageModel,
    n_inputs: int,
    batch_size: int,
    context_length: int,
    top_k_inputs: int,
    ae_paths: list[str],
    force_rerun: bool = False,
):
    batched_data = datasets.load_dataset("georgeyw/dsir-pile-100k", streaming=False)

    data = (
        model.tokenizer(
            batched_data["train"]["contents"][:n_inputs],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=context_length,
        )
        .to(model.device)
        .data
    )

    batched_data = utils.batch_inputs(data, batch_size)

    for ae_path in ae_paths:
        output_file = os.path.join(ae_path, f"max_activating_inputs.pkl")

        if os.path.exists(output_file) and not force_rerun:
            print(f"Skipping {ae_path}")
            continue

        submodule, dictionary, config = utils.load_dictionary(model, ae_path, model.device)
        dictionary = dictionary.to(model.dtype)
        results = get_autointerp_inputs_for_one_sae(
            model, batched_data, dictionary, config, submodule, batch_size, top_k_inputs
        )

        with open(output_file, "wb") as f:
            pickle.dump(results, f)


### END OF ACTIVATION COLLECTION ###
### BEGIN OF AUTOINTERP PROMPT CONSTRUCTION ###


def highlight_top_activations(
    token_str_KL: list[list[str]],
    activations_KL: torch.Tensor,
    top_n: int = 5,
    include_activations: bool = False,
) -> list[list[str]]:
    assert len(token_str_KL) == activations_KL.shape[0], "Number of sequences must match"

    result = []
    for tokens, activations in zip(token_str_KL, activations_KL):
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

        result.append(highlighted_tokens)

    return result


def format_examples(
    tokenizer: AutoTokenizer,
    max_token_idxs_FKL: torch.Tensor,
    max_activations_FKL: torch.Tensor,
    num_top_emphasized_tokens: int,
    include_activations: bool = False,
) -> list[str]:
    # Move tensors to CPU if they're on GPU, seems to be 4x faster on BauLab Machine
    max_token_idxs_FKL = max_token_idxs_FKL.cpu()
    max_activations_FKL = max_activations_FKL.cpu()

    # Batch decode all tokens at once, maintaining individual tokens
    max_token_str_FKL = utils.list_decode(max_token_idxs_FKL, tokenizer)

    example_prompts = []
    for max_token_str_KL, max_activations_KL in tqdm(
        zip(max_token_str_FKL, max_activations_FKL),
        desc="Formatting examples",
        total=len(max_token_str_FKL),
    ):
        formatted_sequences_K = highlight_top_activations(
            max_token_str_KL,
            max_activations_KL,
            top_n=num_top_emphasized_tokens,
            include_activations=include_activations,
        )

        for tokens in formatted_sequences_K:
            if not tokens:
                raise ValueError("Empty sequence found")

        formatted_sequences = ["".join(tokens) for tokens in formatted_sequences_K] 

        example_prompt = ""
        for i, seq in enumerate(formatted_sequences):
            example_prompt += f"\n\n\nExample {i+1}: {seq}\n\n"
        example_prompts.append(example_prompt)

    return example_prompts


if __name__ == "__main__":
    import os

    # This flag is set because we are calling torch.cat in a for loop
    # At some point, we may want to preallocate tensors and fill them in but this is fine for now
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    trainer_ids = [2, 6, 10, 14, 18]
    ae_sweep_paths = {
        "pythia70m_sweep_standard_ctx128_0712": {
            #     # "resid_post_layer_0": {"trainer_ids": None},
            #     # "resid_post_layer_1": {"trainer_ids": None},
            #     # "resid_post_layer_2": {"trainer_ids": None},
            "resid_post_layer_3": {"trainer_ids": trainer_ids},
            #     "resid_post_layer_4": {"trainer_ids": None},
        },
        # "pythia70m_sweep_gated_ctx128_0730": {
        #     # "resid_post_layer_0": {"trainer_ids": None},
        #     # "resid_post_layer_1": {"trainer_ids": None},
        #     # "resid_post_layer_2": {"trainer_ids": None},
        #     "resid_post_layer_3": {"trainer_ids": trainer_ids},
        #     # "resid_post_layer_4": {"trainer_ids": None},
        # },
        # "pythia70m_sweep_panneal_ctx128_0730": {
        #     # "resid_post_layer_0": {"trainer_ids": None},
        #     # "resid_post_layer_1": {"trainer_ids": None},
        #     # "resid_post_layer_2": {"trainer_ids": None},
        #     "resid_post_layer_3": {"trainer_ids": trainer_ids},
        #     # "resid_post_layer_4": {"trainer_ids": None},
        # },
        "pythia70m_sweep_topk_ctx128_0730": {
            # "resid_post_layer_0": {"trainer_ids": None},
            # "resid_post_layer_1": {"trainer_ids": None},
            # "resid_post_layer_2": {"trainer_ids": None},
            "resid_post_layer_3": {"trainer_ids": trainer_ids},
            # "resid_post_layer_4": {"trainer_ids": None},
        },
    }

    # ae_sweep_paths = {
    #     "pythia70m_test_sae": {"resid_post_layer_3": {"trainer_ids": None}},
    # }

    ae_sweep_paths = {
        "gemma-2-2b_sweep_topk_ctx128_0817": {"resid_post_layer_12": {"trainer_ids": None}},
    }

    dictionaries_path = "../dictionary_learning/dictionaries"

    n_inputs = 10000
    top_k_prompts = 10
    model_dtype = torch.bfloat16
    device = "cuda"
    # device = "mps"

    for sweep_name, submodule_trainers in ae_sweep_paths.items():
        ae_group_paths = utils.get_ae_group_paths(dictionaries_path, sweep_name, submodule_trainers)
        ae_paths = utils.get_ae_paths(ae_group_paths)
        context_length = utils.get_ctx_length(ae_paths)

        model_eval_config = utils.ModelEvalConfig.from_sweep_name(sweep_name)
        model_name = model_eval_config.full_model_name

        llm_batch_size, patching_batch_size, eval_results_batch_size = utils.get_batch_sizes(
            model_eval_config,
            train_set_size=n_inputs,
            reduced_GPU_memory=False,
        )

        model = LanguageModel(
            model_name,
            device_map=device,
            dispatch=True,
            attn_implementation="eager",
            torch_dtype=model_dtype,
        )

        get_autointerp_inputs_for_all_saes(
            model,
            n_inputs,
            llm_batch_size,
            context_length,
            top_k_prompts,
            ae_paths,
            force_rerun=True,
        )
