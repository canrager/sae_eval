# %%
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gc
import pickle
import json
from typing import Optional
from datasets import load_dataset
import random
from nnsight import LanguageModel
import torch as t
from torch import nn
from collections import defaultdict
import time

parent_dir = os.path.abspath("..")
sys.path.append(parent_dir)

from attribution import patching_effect
from dictionary_learning.interp import examine_dimension
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.dictionary import AutoEncoder

import experiments.probe_training as probe_training
import experiments.utils as utils
import experiments.eval_saes as eval_saes
import experiments.autointerp as autointerp
import experiments.llm_autointerp.llm_query as llm_query

from experiments.pipeline_config import PipelineConfig, FeatureSelection
from experiments.probe_training import (
    load_and_prepare_dataset,
    get_train_test_data,
    test_probe,
    prepare_probe_data,
    get_all_activations,
    Probe,
)

DEBUGGING = False

if DEBUGGING:
    tracer_kwargs = dict(scan=True, validate=True)
else:
    tracer_kwargs = dict(scan=False, validate=False)


# Metric function effectively maximizing the logit difference between the classes: selected, and nonclass


def metric_fn(
    model: LanguageModel, labels: t.Tensor, probe: Probe, probe_act_submodule: utils.submodule_alias
):
    attn_mask = model.input[1]["attention_mask"]
    acts = probe_act_submodule.output[0]
    acts = acts * attn_mask[:, :, None]
    acts = acts.sum(1) / attn_mask.sum(1)[:, None]

    return t.where(labels == utils.POSITIVE_CLASS_LABEL, probe(acts), -probe(acts))


# Attribution Patching


def get_class_nonclass_samples(data: dict, class_idx: int, device: str) -> tuple[list, t.Tensor]:
    """This is for getting equal number of text samples from the chosen class and all other classes.
    We use this for attribution patching."""
    class_samples = data[class_idx]

    if isinstance(class_samples, dict) and isinstance(class_samples.get("input_ids"), t.Tensor):
        # Combine all non-class tensors

        nonclass_input_ids = []
        nonclass_attention_mask = []
        for profession in data:
            if profession != class_idx and isinstance(profession, int):
                nonclass_input_ids.append(data[profession]["input_ids"])
                nonclass_attention_mask.append(data[profession]["attention_mask"])
        nonclass_input_ids = t.cat(nonclass_input_ids, dim=0)
        nonclass_attention_mask = t.cat(nonclass_attention_mask, dim=0)

        # Randomly select indices
        num_class_samples = class_samples["input_ids"].size(0)
        indices = t.randperm(nonclass_input_ids.size(0))[:num_class_samples]

        # Select random samples
        nonclass_input_ids = nonclass_input_ids[indices]
        nonclass_attention_mask = nonclass_attention_mask[indices]

        combined_input_ids = t.cat([class_samples["input_ids"], nonclass_input_ids], dim=0)
        combined_attention_mask = t.cat(
            [class_samples["attention_mask"], nonclass_attention_mask], dim=0
        )

        combined_input_ids[::2] = class_samples["input_ids"]
        combined_input_ids[1::2] = nonclass_input_ids
        combined_attention_mask[::2] = class_samples["attention_mask"]
        combined_attention_mask[1::2] = nonclass_attention_mask

        combined_samples = {
            "input_ids": combined_input_ids,
            "attention_mask": combined_attention_mask,
        }

        num_class_samples = class_samples["input_ids"].size(0)
        num_nonclass_samples = nonclass_input_ids.size(0)
        num_combined_samples = num_class_samples + num_nonclass_samples
    elif isinstance(class_samples, list) and isinstance(class_samples[0], str):
        nonclass_samples = []
        for profession in data:
            if profession != class_idx:
                nonclass_samples.extend(data[profession])

        nonclass_samples = random.sample(nonclass_samples, len(class_samples))
        combined_samples = class_samples + nonclass_samples
        num_class_samples = len(class_samples)
        num_nonclass_samples = len(nonclass_samples)
        num_combined_samples = num_class_samples + num_nonclass_samples
    else:
        raise ValueError("Unsupported input type")

    combined_labels = t.empty(num_combined_samples, dtype=t.int, device=device)
    combined_labels[::2] = utils.POSITIVE_CLASS_LABEL
    combined_labels[1::2] = utils.NEGATIVE_CLASS_LABEL

    return combined_samples, combined_labels


def get_class_samples(data: dict, class_idx: int, device: str) -> tuple[list, t.Tensor]:
    """This is for getting equal number of text samples from the chosen class and all other classes.
    We use this for attribution patching."""
    class_samples = data[class_idx]

    if isinstance(class_samples, list) and isinstance(class_samples[0], str):
        num_class_samples = len(class_samples)
    elif isinstance(class_samples, dict) and isinstance(class_samples["input_ids"], t.Tensor):
        num_class_samples = class_samples["input_ids"].size(0)
    else:
        raise ValueError("Unsupported input type")

    class_labels = t.full(
        (num_class_samples,), utils.POSITIVE_CLASS_LABEL, dtype=t.int, device=device
    )

    return class_samples, class_labels


# TODO: Think about removing support for list of string inputs
def get_paired_class_samples(data: dict, class_idx, device: str) -> tuple[list, t.Tensor]:
    """This is for getting equal number of text samples from the chosen class and all other classes.
    We use this for attribution patching."""

    # TODO: Clean this up
    # Switch from interleaving to shuffling

    if class_idx not in utils.PAIRED_CLASS_KEYS:
        raise ValueError(f"Class {class_idx} not in PAIRED_CLASS_KEYS")

    class_samples = data[class_idx]
    paired_class_idx = utils.PAIRED_CLASS_KEYS[class_idx]
    paired_class_samples = data[paired_class_idx]

    if isinstance(class_samples, list) and isinstance(class_samples[0], str):
        combined_samples = class_samples + paired_class_samples
        num_class_samples = len(class_samples)
        num_nonclass_samples = len(paired_class_samples)
        num_combined_samples = num_class_samples + num_nonclass_samples

        # Interleave the samples
        combined_samples = [None] * num_combined_samples
        combined_samples[::2] = class_samples
        combined_samples[1::2] = paired_class_samples
    elif isinstance(class_samples, dict) and isinstance(class_samples["input_ids"], t.Tensor):
        combined_input_ids = t.cat(
            [class_samples["input_ids"], paired_class_samples["input_ids"]], dim=0
        )
        combined_attention_mask = t.cat(
            [class_samples["attention_mask"], paired_class_samples["attention_mask"]], dim=0
        )

        combined_input_ids[::2] = class_samples["input_ids"]
        combined_input_ids[1::2] = paired_class_samples["input_ids"]
        combined_attention_mask[::2] = class_samples["attention_mask"]
        combined_attention_mask[1::2] = paired_class_samples["attention_mask"]

        combined_samples = {
            "input_ids": combined_input_ids,
            "attention_mask": combined_attention_mask,
        }
        num_class_samples = class_samples["input_ids"].size(0)
        num_nonclass_samples = paired_class_samples["input_ids"].size(0)
        num_combined_samples = num_class_samples + num_nonclass_samples
    else:
        raise ValueError("Unsupported input type")

    assert num_class_samples == num_nonclass_samples

    # combined_labels = [utils.POSITIVE_CLASS_LABEL] * num_class_samples + [
    #     utils.NEGATIVE_CLASS_LABEL
    # ] * num_nonclass_samples

    # Create interleaved labels
    combined_labels = t.empty(num_combined_samples, dtype=t.int, device=device)
    combined_labels[::2] = utils.POSITIVE_CLASS_LABEL
    combined_labels[1::2] = utils.NEGATIVE_CLASS_LABEL

    return combined_samples, combined_labels


def get_effects_per_class(
    model: LanguageModel,
    submodules: list[utils.submodule_alias],
    dictionaries: dict[utils.submodule_alias, AutoEncoder],
    probes: dict[int | str, Probe],
    probe_act_submodule: utils.submodule_alias,
    class_idx: int | str,
    train_bios: dict,
    seed: int,
    batch_size: int = 10,
    patching_method: str = "attrib",
    steps: int = 10,  # only used for ig
) -> t.Tensor:
    """
    Probe_act_submodule is the submodule where the probe is attached, usually resid_post.
    Att the end of the function nodes is a dict of submodules to tensors. This is if we want to intervene on multiple autoencoders.
    We aren't currently using this feature, so we currently only return the tensor.
    """
    device = model.device
    probe = probes[class_idx]

    if isinstance(class_idx, int):
        inputs_train, labels_train = get_class_samples(train_bios, class_idx, device)
        # texts_train, labels_train = get_class_nonclass_samples(train_bios, class_idx, device)
    else:
        inputs_train, labels_train = get_paired_class_samples(train_bios, class_idx, device)

    inputs_train = utils.batch_inputs(inputs_train, batch_size)
    labels_train = utils.batch_inputs(labels_train, batch_size)

    running_total = 0
    running_nodes = None

    n_batches = len(inputs_train)

    for batch_idx, (clean, labels) in enumerate(zip(inputs_train, labels_train)):
        if batch_idx == n_batches:
            break

        effects, _, _, _ = patching_effect(
            clean,
            None,
            model,
            submodules,
            dictionaries,
            metric_fn,
            metric_kwargs=dict(labels=labels, probe=probe, probe_act_submodule=probe_act_submodule),
            method=patching_method,
            steps=steps,
        )
        with t.no_grad():
            if running_nodes is None:
                running_nodes = {
                    k: len(clean) * v.sum(dim=1).mean(dim=0) for k, v in effects.items()
                }
            else:
                for k, v in effects.items():
                    running_nodes[k] += len(clean) * v.sum(dim=1).mean(dim=0)
            running_total += len(clean)
        del effects, _
        gc.collect()

    nodes = {k: v / running_total for k, v in running_nodes.items()}
    # Convert SparseAct to Tensor
    nodes = {k: v.act for k, v in nodes.items()}

    assert len(nodes) == 1, "Only one submodule should be intervened on"
    node_value = next(iter(nodes.values()))

    return node_value


def get_all_node_effects_for_one_sae(
    model: LanguageModel,
    submodules: list[utils.submodule_alias],
    dictionaries: dict[utils.submodule_alias, AutoEncoder],
    ae_path: str,
    force_recompute: bool,
    probes: dict[int | str, Probe],
    probe_act_submodule: utils.submodule_alias,
    chosen_class_indices: list[int | str],
    train_bios: dict,
    seed: int,
    batch_size: int = 10,
    patching_method: str = "attrib",
    steps: int = 10,  # only used for ig
) -> t.Tensor:
    node_effects_path = os.path.join(ae_path, "node_effects.pkl")

    if os.path.exists(node_effects_path) and not force_recompute:
        print(f"Loading node effects from {node_effects_path}")

        with open(node_effects_path, "rb") as f:
            node_effects = pickle.load(f)
        return node_effects
    if not os.path.exists(node_effects_path):
        print(f"Node effects not found, computing for {ae_path}")
    elif force_recompute:
        print(f"Recomputing node effects for {ae_path}")

    node_effects = {}

    for ablated_class_idx in tqdm(chosen_class_indices, "Getting node effects"):
        node_effects[ablated_class_idx] = get_effects_per_class(
            model,
            submodules,
            dictionaries,
            probes,
            probe_act_submodule,
            ablated_class_idx,
            train_bios,
            seed,
            batch_size=batch_size,
            patching_method=patching_method,
            steps=steps,
        )
    node_effects = utils.to_device(node_effects, "cpu")

    save_log_files(ae_path, node_effects, "node_effects", ".pkl")

    return node_effects


# Get the output activations for the submodule where some saes are ablated
# Currently deprecated
def get_acts_ablated(text, model, submodules, dictionaries, to_ablate):
    is_tuple = {}
    with t.no_grad(), model.trace("_"):
        for submodule in submodules:
            is_tuple[submodule] = type(submodule.output.shape) == tuple

    with t.no_grad(), model.trace(text, **tracer_kwargs):
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            feat_idxs = to_ablate[submodule]
            x = submodule.output
            if is_tuple[submodule]:
                x = x[0]
            x_hat, f = dictionary(x, output_features=True)
            res = x - x_hat
            f[..., feat_idxs] = 0.0  # zero ablation
            if is_tuple[submodule]:
                submodule.output[0][:] = dictionary.decode(f) + res
            else:
                submodule.output = dictionary.decode(f) + res
        attn_mask = model.input[1]["attention_mask"]
        act = model.gpt_neox.layers[layer].output[0]
        act = act * attn_mask[:, :, None]
        act = act.sum(1) / attn_mask.sum(1)[:, None]
        act = act.save()

    t.cuda.empty_cache()
    gc.collect()

    return act.value


# Get the output activations for the submodule where some saes are ablated
@t.no_grad()
def get_all_acts_ablated(
    text_inputs: list[str],
    model: LanguageModel,
    submodules: list[utils.submodule_alias],
    dictionaries: dict[utils.submodule_alias, AutoEncoder],
    to_ablate: t.Tensor,
    batch_size: int,
    probe_submodule: utils.submodule_alias,
):
    text_batches = utils.batch_inputs(text_inputs, batch_size)

    is_tuple = {}
    with t.no_grad(), model.trace("_"):
        for submodule in submodules:
            is_tuple[submodule] = type(submodule.output.shape) == tuple

    all_acts_list_BD = []
    for text_batch_BL in text_batches:
        with t.no_grad(), model.trace(text_batch_BL, **tracer_kwargs):
            for submodule in submodules:
                dictionary = dictionaries[submodule]
                # feat_idxs = to_ablate[submodule] # Uncomment this line to restore ablating multiple SAEs
                feat_idxs = to_ablate
                x = submodule.output
                if is_tuple[submodule]:
                    x = x[0]
                x_hat, f = dictionary(x, output_features=True)
                res = x - x_hat
                f[..., feat_idxs] = 0.0  # zero ablation
                if is_tuple[submodule]:
                    submodule.output[0][:] = dictionary.decode(f) + res
                else:
                    submodule.output = dictionary.decode(f) + res
            attn_mask = model.input[1]["attention_mask"]
            act = probe_submodule.output[0]
            act = act * attn_mask[:, :, None]
            act = act.sum(1) / attn_mask.sum(1)[:, None]
            act = act.save()
        all_acts_list_BD.append(act.value)

    all_acts_bD = t.cat(all_acts_list_BD, dim=0)

    return all_acts_bD


# putting feats_to_ablate in a more useful format
def n_hot(feats, dim, device="cpu"):
    out = t.zeros(dim, dtype=t.bool, device=device)
    for feat in feats:
        out[feat] = True
    return out


def select_significant_features(
    node_effects: dict[int, dict[utils.submodule_alias, t.Tensor]],
    dict_size: int,
    T_effect: float = 0.001,
    verbose: bool = True,
    convert_to_n_hot: bool = True,
    device: str = "cpu",
) -> dict[int, dict[utils.submodule_alias, t.Tensor]]:
    """There's a bug somewhere in here if the T_effect is too high, it will return an empty dict."""
    feats_above_T = {}

    for abl_class_idx in node_effects.keys():
        total_features_per_abl_class = 0
        feats_above_T[abl_class_idx] = defaultdict(list)
        for submodule in node_effects[abl_class_idx].keys():
            # TODO: Warning about .nonzero() and bools
            for feat_idx in (node_effects[abl_class_idx][submodule] > T_effect).nonzero():
                feats_above_T[abl_class_idx][submodule].append(feat_idx.item())
                total_features_per_abl_class += 1
        if convert_to_n_hot:
            feats_above_T[abl_class_idx] = {
                submodule: n_hot(feats, dict_size, device)
                for submodule, feats in feats_above_T[abl_class_idx].items()
            }
        if verbose:
            print(
                f"T_effect {T_effect}, class {abl_class_idx}, all submodules, #significant features: {total_features_per_abl_class}"
            )

    return feats_above_T


def select_significant_features2(
    node_effects: dict[int | str, t.Tensor],
    T_effect: float = 0.001,
    verbose: bool = True,
) -> dict[int | str, t.Tensor]:
    """This function is more idiomatic pytorch and doesn't have the bug of returning an empty dict."""
    # TODO: Switch over to this function, or maybe use the other one for the unique class features.
    feats_above_T = {}

    for abl_class_idx in node_effects.keys():
        total_features_per_abl_class = 0
        feats_above_T[abl_class_idx] = {}
        feats_above_T[abl_class_idx] = node_effects[abl_class_idx] > T_effect
        total_features_per_abl_class += feats_above_T[abl_class_idx].sum().item()

    if verbose:
        print(
            f"T_effect {T_effect}, class {abl_class_idx}, all submodules, #significant features: {total_features_per_abl_class}"
        )

    return feats_above_T


def select_top_n_features(
    node_effects: dict[int | str, t.Tensor],
    n: int,
) -> dict[int | str, t.Tensor]:
    top_n_features = {}

    for abl_class_idx, effects in node_effects.items():
        top_n_features[abl_class_idx] = {}

        assert (
            n <= effects.numel()
        ), f"n ({n}) must not be larger than the number of features ({effects.numel()}) for ablation class {abl_class_idx}"

        # Find non-zero effects
        non_zero_mask = effects != 0
        non_zero_effects = effects[non_zero_mask]
        num_non_zero = non_zero_effects.numel()

        if num_non_zero < n:
            print(
                f"WARNING: only {num_non_zero} non-zero effects found for ablation class {abl_class_idx}, which is less than the requested {n}."
            )

        # Select top n or all non-zero effects, whichever is smaller
        k = min(n, num_non_zero)

        if k == 0:
            print(
                f"WARNING: No non-zero effects found for ablation class {abl_class_idx}. Returning an empty mask."
            )
            top_n_features[abl_class_idx] = t.zeros_like(effects, dtype=t.bool)
        else:
            # Get the indices of the top N effects
            _, top_indices = t.topk(effects, k)

            # Create a boolean mask tensor
            mask = t.zeros_like(effects, dtype=t.bool)
            mask[top_indices] = True

            top_n_features[abl_class_idx] = mask

    return top_n_features


def select_unique_class_features(
    node_effects: dict[int, dict[utils.submodule_alias, t.Tensor]],
    dict_size: int,
    T_effect: float = 0.001,
    T_max_sideeffect: float = 0.000001,
    verbose: bool = True,
    device: str = "cpu",
) -> dict[int, dict[utils.submodule_alias, t.Tensor]]:
    non_neglectable_feats = select_significant_features(
        node_effects, dict_size, T_max_sideeffect, convert_to_n_hot=False, verbose=True
    )
    significant_feats = select_significant_features(
        node_effects, dict_size, T_effect, convert_to_n_hot=False, verbose=True
    )

    feats_above_T = {}
    for abl_class_idx in node_effects.keys():
        total_features_per_abl_class = 0
        feats_above_T[abl_class_idx] = defaultdict(list)
        for submodule in node_effects[abl_class_idx].keys():
            # Get a blacklist of features that have side effects above T_max_sideeffect in other submodules
            sideeffect_features = []
            for other_class_idx in node_effects.keys():
                if other_class_idx != abl_class_idx:
                    sideeffect_features.extend(non_neglectable_feats[other_class_idx][submodule])
            sideeffect_features = set(sideeffect_features)
            if verbose:
                print(f"sideeffect features: {len(sideeffect_features)}")

            # Add features above T_effect that are not in the blacklist
            for feat_idx in significant_feats[abl_class_idx][submodule]:
                if feat_idx not in sideeffect_features:
                    feats_above_T[abl_class_idx][submodule].append(feat_idx)
                    total_features_per_abl_class += 1
        feats_above_T[abl_class_idx] = {
            submodule: n_hot(feats, dict_size, device)
            for submodule, feats in feats_above_T[abl_class_idx].items()
        }
        if verbose:
            print(
                f"T_effect {T_effect}, class {abl_class_idx}, all submodules, #unique features: {total_features_per_abl_class}"
            )

    return feats_above_T


def select_features(
    selection_method: FeatureSelection,
    node_effects: dict[int | str, t.Tensor],
    dict_size: int,
    T_effects: list[float],
    T_max_sideeffect: float,
    verbose: bool = False,
) -> dict[int | float, dict[int | str, t.Tensor]]:
    selected_features = {}
    if selection_method == FeatureSelection.unique:
        for T_effect in T_effects:
            selected_features[T_effect] = select_unique_class_features(
                node_effects,
                dict_size,
                T_effect=T_effect,
                T_max_sideeffect=T_max_sideeffect,
                verbose=verbose,
            )
    elif selection_method == FeatureSelection.above_threshold:
        for T_effect in T_effects:
            selected_features[T_effect] = select_significant_features2(
                node_effects, T_effect=T_effect, verbose=verbose
            )
    elif selection_method == FeatureSelection.top_n:
        for n in T_effects:
            selected_features[n] = select_top_n_features(node_effects, n)
    else:
        raise ValueError("Invalid selection method")

    for T_effect in T_effects:
        for ablated_class_idx in selected_features[T_effect]:
            mask = selected_features[T_effect][ablated_class_idx]
            effects = node_effects[ablated_class_idx]
            assert mask.size() == effects.size(), "Mask and effects must have the same size"

    return selected_features


def save_log_files(
    ae_path: str, data: dict, base_filename: str, extension: str, save_backup: bool = False
):
    # Always save/overwrite the main file
    main_file = os.path.join(ae_path, f"{base_filename}{extension}")
    with open(main_file, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved main file: {base_filename}{extension}")

    if save_backup:
        # Find the next available number for the backup file
        counter = 1
        while True:
            backup_filename = f"{base_filename}{counter}{extension}"
            full_path = os.path.join(ae_path, backup_filename)

            if not os.path.exists(full_path):
                with open(full_path, "wb") as f:
                    pickle.dump(data, f)
                print(f"Saved backup as: {backup_filename}")
                break

            counter += 1


def run_interventions(
    submodule_trainers: dict,
    p_config: PipelineConfig,
    sweep_name: str,
    random_seed: int,
    device: str = "cuda",
    verbose: bool = False,
):
    T_max_sideeffect = 0.000001  # Deprecated

    spurious_correlation_removal = utils.check_if_spurious_correlation_removal(
        p_config.chosen_class_indices
    )

    t.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    model_eval_config = utils.ModelEvalConfig.from_sweep_name(sweep_name)
    model_name = model_eval_config.full_model_name

    llm_batch_size, patching_batch_size, eval_results_batch_size = utils.get_batch_sizes(
        model_eval_config,
        p_config.reduced_GPU_memory,
        p_config.train_set_size,
        p_config.test_set_size,
        p_config.probe_train_set_size,
        p_config.probe_test_set_size,
    )

    model = LanguageModel(
        model_name,
        device_map=device,
        dispatch=True,
        attn_implementation="eager",
        torch_dtype=p_config.model_dtype,
    )

    ae_group_paths = utils.get_ae_group_paths(
        p_config.dictionaries_path, sweep_name, submodule_trainers
    )
    ae_paths = utils.get_ae_paths(ae_group_paths)

    # TODO: experiment with different context lengths
    context_length = utils.get_ctx_length(ae_paths)

    probe_layer = model_eval_config.probe_layer

    if model_name == "google/gemma-2-2b":
        if isinstance(p_config.gemma_probe_layer, int):
            probe_layer = p_config.gemma_probe_layer
        elif p_config.gemma_probe_layer == "sae_layer":
            probe_layer = utils.get_sae_layer(ae_paths)
        else:
            raise ValueError("Invalid probe layer")

    probe_act_submodule = utils.get_submodule(model, "resid_post", probe_layer)

    # This will only run eval_saes on autoencoders that don't yet have a eval_results.json file
    eval_saes.eval_saes(
        model,
        ae_paths,
        p_config.eval_saes_n_inputs,
        eval_results_batch_size,
        device,
        overwrite_prev_results=p_config.force_eval_results_recompute,
    )

    if p_config.use_autointerp:
        autointerp.get_autointerp_inputs_for_all_saes(
            model,
            p_config.max_activations_collection_n_inputs,
            llm_batch_size,
            context_length,
            p_config.top_k_inputs_act_collect,
            ae_paths,
            force_rerun=p_config.force_max_activations_recompute,
        )

    dataset, _ = load_and_prepare_dataset()
    train_bios, test_bios = get_train_test_data(
        dataset,
        p_config.train_set_size,
        p_config.test_set_size,
        p_config.include_gender,
    )

    train_bios = utils.tokenize_data(train_bios, model.tokenizer, context_length, device)
    test_bios = utils.tokenize_data(test_bios, model.tokenizer, context_length, device)

    only_model_name = model_name.split("/")[-1]
    probe_path = f"{p_config.probes_dir}/{only_model_name}/probes_ctx_len_{context_length}_layer_{probe_layer}.pkl"

    # TODO: Add logic to ensure probes share keys with train_bios and test_bios
    # We train the probes and save them as a file.
    if not os.path.exists(probe_path) or p_config.force_probe_recompute:
        if p_config.force_probe_recompute:
            print("Force recomputing probes")
        else:
            print("Probes not found, training probes")
        probe_training.train_probes(
            p_config.probe_train_set_size,
            p_config.probe_test_set_size,
            model,
            context_length=context_length,
            probe_batch_size=p_config.probe_batch_size,
            llm_batch_size=llm_batch_size,
            device=device,
            probe_output_filename=probe_path,
            dataset_name="bias_in_bios",
            probe_dir=p_config.probes_dir,
            llm_model_name=model_name,
            epochs=p_config.probe_epochs,
            model_dtype=p_config.model_dtype,
            spurious_correlation_removal=spurious_correlation_removal,
        )

    with open(probe_path, "rb") as f:
        probes = pickle.load(f)

    ### Get activations for original model, all classes
    print("Getting activations for original model")
    test_acts = {}
    for class_idx in tqdm(
        p_config.chosen_class_indices, desc="Getting activations per evaluated class"
    ):
        test_acts[class_idx] = get_all_activations(
            test_bios[class_idx], model, llm_batch_size, probe_act_submodule
        )

        if class_idx in utils.PAIRED_CLASS_KEYS:
            paired_class_idx = utils.PAIRED_CLASS_KEYS[class_idx]
            test_acts[paired_class_idx] = get_all_activations(
                test_bios[paired_class_idx], model, llm_batch_size, probe_act_submodule
            )

    test_accuracies = probe_training.get_probe_test_accuracy(
        probes, p_config.chosen_class_indices, test_acts, p_config.probe_batch_size
    )
    del test_acts

    # %%
    ### Get activations for ablated models
    # ablating the top features for each class
    print("Getting activations for ablated models")

    for ae_path in ae_paths:
        print(f"Running ablation for {ae_path}")
        submodules = []
        dictionaries = {}
        submodule, dictionary, sae_config = utils.load_dictionary(model, ae_path, device)
        dictionary = dictionary.to(dtype=p_config.model_dtype)
        submodules.append(submodule)
        dictionaries[submodule] = dictionary
        dict_size = sae_config["trainer"]["dict_size"]

        class_accuracies = {"clean_acc": test_accuracies}

        # For every class, we get the indirect effects of every SAE feature wrt. the class probe
        node_effects = get_all_node_effects_for_one_sae(
            model=model,
            submodules=submodules,
            dictionaries=dictionaries,
            ae_path=ae_path,
            force_recompute=p_config.force_node_effects_recompute,
            probes=probes,
            probe_act_submodule=probe_act_submodule,
            chosen_class_indices=p_config.chosen_class_indices,
            train_bios=train_bios,
            seed=random_seed,
            batch_size=patching_batch_size,
            patching_method=p_config.attribution_patching_method,
            steps=p_config.ig_steps,
        )

        if p_config.use_autointerp:
            # This will save node_effects_auto_interp.pkl, node_effects_bias_shift_dir1.pkl, and node_effects_bias_shift_dir2.pkl alongside each SAE
            node_effects_auto_interp, node_effects_bias_shift_dir1, node_effects_bias_shift_dir2 = (
                llm_query.perform_llm_autointerp(
                    tokenizer=model.tokenizer,
                    p_config=p_config,
                    ae_path=ae_path,
                    debug_mode=True,
                )
            )
            all_node_effects = [
                (node_effects_auto_interp, "_auto_interp", p_config.autointerp_t_effects),
                (node_effects_bias_shift_dir1, "_bias_shift_dir1", p_config.autointerp_t_effects),
                (node_effects_bias_shift_dir2, "_bias_shift_dir2", p_config.autointerp_t_effects),
                (node_effects, "_attrib", p_config.attrib_t_effects),
            ]
        else:
            all_node_effects = [(node_effects, "_attrib", p_config.attrib_t_effects)]

        t.cuda.empty_cache()
        gc.collect()

        for node_effects_group, effects_group_name, T_effects in all_node_effects:
            output_base_filename = f"class_accuracies{effects_group_name}"

            if (
                os.path.exists(os.path.join(ae_path, f"{output_base_filename}.pkl"))
                and not p_config.force_ablations_recompute
            ):
                print(f"Skipping ablations for {ae_path} {effects_group_name}")
                continue
            if not os.path.exists(os.path.join(ae_path, f"{output_base_filename}.pkl")):
                print(f"Running ablations for {ae_path} {effects_group_name}")
            elif p_config.force_ablations_recompute:
                print(f"Recomputing ablations for {ae_path} {effects_group_name}")

            selected_features = select_features(
                p_config.selection_method,
                node_effects_group,
                dict_size,
                T_effects,
                T_max_sideeffect,
                verbose=verbose,
            )

            node_effects_group_classes = list(node_effects_group.keys())

            with t.inference_mode():
                # Now that we have collected node effects and selected features, we ablate the selected features and measure the change in probe accuracy
                for ablated_class_idx in node_effects_group_classes:
                    class_accuracies[ablated_class_idx] = {}
                    print(f"evaluating class {ablated_class_idx}")

                    for T_effect in T_effects:
                        class_accuracies[ablated_class_idx][T_effect] = {}
                        selected_features_mask = selected_features[T_effect][ablated_class_idx]

                        if t.all(selected_features_mask == 0):
                            print(f"No features selected for T_effect = {T_effect}")
                            # If no features are selected, we skip the ablation
                            # We set the accuracy to the clean accuracy for ease of plotting later
                            class_accuracies[ablated_class_idx][T_effect] = test_accuracies
                            continue

                        if verbose:
                            print(f"Running ablation for T_effect = {T_effect}")
                            print(f"Ablating {selected_features_mask.sum()} features")
                        test_acts_ablated = {}
                        for evaluated_class_idx in tqdm(
                            node_effects_group_classes, desc="Getting activations"
                        ):
                            test_acts_ablated[evaluated_class_idx] = get_all_acts_ablated(
                                test_bios[evaluated_class_idx],
                                model,
                                submodules,
                                dictionaries,
                                selected_features_mask,
                                llm_batch_size,
                                probe_act_submodule,
                            )

                            if evaluated_class_idx in utils.PAIRED_CLASS_KEYS:
                                paired_class_idx = utils.PAIRED_CLASS_KEYS[evaluated_class_idx]
                                test_acts_ablated[paired_class_idx] = get_all_acts_ablated(
                                    test_bios[paired_class_idx],
                                    model,
                                    submodules,
                                    dictionaries,
                                    selected_features_mask,
                                    llm_batch_size,
                                    probe_act_submodule,
                                )

                        ablated_class_accuracies = probe_training.get_probe_test_accuracy(
                            probes,
                            node_effects_group_classes,
                            test_acts_ablated,
                            p_config.probe_batch_size,
                        )

                        class_accuracies[ablated_class_idx][T_effect] = ablated_class_accuracies

                        for evaluated_class_idx in ablated_class_accuracies:
                            if verbose:
                                print(
                                    f"Ablated {ablated_class_idx}, evaluated {evaluated_class_idx} test accuracy: {ablated_class_accuracies[evaluated_class_idx]['acc']}"
                                )

            class_accuracies = utils.to_device(class_accuracies, "cpu")

            save_log_files(
                ae_path,
                class_accuracies,
                output_base_filename,
                ".pkl",
                save_backup=False,
            )

    # Extract results to a separate folder
    src_folder = os.path.join(p_config.dictionaries_path, sweep_name)

    if spurious_correlation_removal:
        output_folder_name = f"{sweep_name}_probe_layer_{probe_layer}_spurious_removal"
    else:
        output_folder_name = f"{sweep_name}_probe_layer_{probe_layer}_tpp"
    results_folder = os.path.join(src_folder, output_folder_name)

    utils.extract_results(src_folder, results_folder, p_config.saving_exclude_files, ae_paths)


# %%

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    random_seed = 42

    # Use for debugging / any time you need to run from root dir
    # dictionaries_path = "dictionary_learning/dictionaries"
    # probes_dir = "experiments/trained_bib_probes"

    # Example of sweeping over all SAEs in a sweep
    ae_sweep_paths = {"pythia70m_test_sae": None}

    # Example of sweeping over all SAEs in a submodule
    ae_sweep_paths = {"pythia70m_test_sae": {"resid_post_layer_3": {"trainer_ids": None}}}

    # Example of sweeping over a single SAE
    ae_sweep_paths = {"pythia70m_test_sae": {"resid_post_layer_3": {"trainer_ids": [0]}}}

    ae_sweep_paths = {"pythia70m_sweep_standard_ctx128_0712": None}

    ae_sweep_paths = {
        "pythia70m_sweep_standard_ctx128_0712": {"resid_post_layer_3": {"trainer_ids": None}}
    }
    ae_sweep_paths = {
        "pythia70m_sweep_standard_ctx128_0712": {
            "resid_post_layer_3": {"trainer_ids": [1, 7, 11, 18]}
            # "resid_post_layer_3": {"trainer_ids": [18]}
        }
    }

    ae_sweep_paths = {
        "pythia70m_sweep_topk_ctx128_0730": {
            # "resid_post_layer_0": {"trainer_ids": None},
            # "resid_post_layer_1": {"trainer_ids": None},
            # "resid_post_layer_2": {"trainer_ids": None},
            # "resid_post_layer_3": {"trainer_ids": None},
            # "resid_post_layer_4": {"trainer_ids": None},
            "resid_post_layer_3": {"trainer_ids": [6]},
        }
    }

    trainer_ids = [2, 6, 10, 14, 18]

    ae_sweep_paths = {
        "pythia70m_sweep_standard_ctx128_0712": {
            #     # "resid_post_layer_0": {"trainer_ids": None},
            #     # "resid_post_layer_1": {"trainer_ids": None},
            #     # "resid_post_layer_2": {"trainer_ids": None},
            "resid_post_layer_3": {"trainer_ids": [6]},
            #     "resid_post_layer_4": {"trainer_ids": None},
        },
        # "pythia70m_sweep_gated_ctx128_0730": {
        #     # "resid_post_layer_0": {"trainer_ids": None},
        #     # "resid_post_layer_1": {"trainer_ids": None},
        #     # "resid_post_layer_2": {"trainer_ids": None},
        #     "resid_post_layer_3": {"trainer_ids": [2, 6, 10, 18]},
        #     # "resid_post_layer_4": {"trainer_ids": None},
        # },
        # "pythia70m_sweep_panneal_ctx128_0730": {
        #     # "resid_post_layer_0": {"trainer_ids": None},
        #     # "resid_post_layer_1": {"trainer_ids": None},
        #     # "resid_post_layer_2": {"trainer_ids": None},
        #     "resid_post_layer_3": {"trainer_ids": trainer_ids},
        #     # "resid_post_layer_4": {"trainer_ids": None},
        # },
        # "pythia70m_sweep_topk_ctx128_0730": {
        #     # "resid_post_layer_0": {"trainer_ids": None},
        #     # "resid_post_layer_1": {"trainer_ids": None},
        #     # "resid_post_layer_2": {"trainer_ids": None},
        #     "resid_post_layer_3": {"trainer_ids": [2, 6, 10, 18]},
        #     # "resid_post_layer_4": {"trainer_ids": None},
        # },
    }

    trainer_ids = None
    trainer_ids = [0, 2, 4, 5]

    ae_sweep_paths = {
        "gemma-2-2b_sweep_topk_ctx128_ef8_0824": {
            # "resid_post_layer_3": {"trainer_ids": trainer_ids},
            # "resid_post_layer_7": {"trainer_ids": trainer_ids},
            "resid_post_layer_11": {"trainer_ids": trainer_ids},
            # "resid_post_layer_15": {"trainer_ids": trainer_ids},
            # "resid_post_layer_19": {"trainer_ids": trainer_ids},
        },
        "gemma-2-2b_sweep_standard_ctx128_ef8_0824": {
            # "resid_post_layer_3": {"trainer_ids": trainer_ids},
            # "resid_post_layer_7": {"trainer_ids": trainer_ids},
            "resid_post_layer_11": {"trainer_ids": trainer_ids},
            # "resid_post_layer_15": {"trainer_ids": trainer_ids},
            # "resid_post_layer_19": {"trainer_ids": trainer_ids},
        },
    }

    pipeline_config = PipelineConfig()

    if pipeline_config.use_autointerp:
        with open("../anthropic_api_key.txt", "r") as f:
            api_key = f.read().strip()

        os.environ["ANTHROPIC_API_KEY"] = api_key

    # This will look for any empty folders in any ae_path and raise an error if it finds any
    for sweep_name, submodule_trainers in ae_sweep_paths.items():
        ae_group_paths = utils.get_ae_group_paths(
            pipeline_config.dictionaries_path, sweep_name, submodule_trainers
        )

    start_time = time.time()

    for sweep_name, submodule_trainers in ae_sweep_paths.items():
        run_interventions(
            submodule_trainers,
            pipeline_config,
            sweep_name,
            random_seed,
            verbose=True,
        )

    end_time = time.time()

    print(f"Time taken: {end_time - start_time} seconds")
