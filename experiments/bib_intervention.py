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
from enum import Enum
import time

parent_dir = os.path.abspath("..")
sys.path.append(parent_dir)

from attribution import patching_effect
from dictionary_learning.interp import examine_dimension
from dictionary_learning.utils import hf_dataset_to_generator
import experiments.probe_training as probe_training
import experiments.utils as utils
import experiments.eval_saes as eval_saes

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


class FeatureSelection(Enum):
    unique = 1
    above_threshold = 2
    top_n = 3


# Metric function effectively maximizing the logit difference between the classes: selected, and nonclass


def metric_fn(model, labels, probe, probe_act_submodule):
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
    nonclass_samples = []

    for profession in data:
        if profession != class_idx:
            nonclass_samples.extend(data[profession])

    if isinstance(class_samples, dict) and isinstance(class_samples.get("input_ids"), t.Tensor):
        # Combine all non-class tensors
        nonclass_input_ids = t.cat([sample["input_ids"] for sample in nonclass_samples], dim=0)
        nonclass_attention_mask = t.cat(
            [sample["attention_mask"] for sample in nonclass_samples], dim=0
        )

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
def get_paired_class_samples(data: dict, class_idx: int, device: str) -> tuple[list, t.Tensor]:
    """This is for getting equal number of text samples from the chosen class and all other classes.
    We use this for attribution patching."""

    # TODO: Clean this up
    # I'm interleaving the samples because I'm sorting the samples by length.
    # This can minimize the number of padding tokens in the batch for efficiency.

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


# shift_dataset = load_dataset("LabHC/bias_in_bios", streaming=True)


# To fit on 24GB VRAM GPU, I set the next 2 default batch_sizes to 64
def get_data(train=True, ambiguous=True, gender_balanced=False, batch_size=64, seed=42):
    profession_dict = {"professor": 21, "nurse": 13}
    male_prof = "professor"
    female_prof = "nurse"
    DEVICE = "cuda"
    if train:
        data = shift_dataset["train"]
    else:
        data = shift_dataset["test"]
    if ambiguous:
        neg = [
            x["hard_text"]
            for x in data
            if x["profession"] == profession_dict[male_prof] and x["gender"] == 0
        ]
        pos = [
            x["hard_text"]
            for x in data
            if x["profession"] == profession_dict[female_prof] and x["gender"] == 1
        ]
        n = min([len(neg), len(pos)])
        neg, pos = neg[:n], pos[:n]
        data = neg + pos
        labels = [0] * n + [1] * n
        idxs = list(range(2 * n))
        random.Random(seed).shuffle(idxs)
        data, labels = [data[i] for i in idxs], [labels[i] for i in idxs]
        true_labels = spurious_labels = labels
    elif not gender_balanced:
        neg_neg = [
            x["hard_text"]
            for x in data
            if x["profession"] == profession_dict[male_prof] and x["gender"] == 0
        ]
        neg_pos = [
            x["hard_text"]
            for x in data
            if x["profession"] == profession_dict[male_prof] and x["gender"] == 1
        ]
        pos_neg = [
            x["hard_text"]
            for x in data
            if x["profession"] == profession_dict[female_prof] and x["gender"] == 0
        ]
        pos_pos = [
            x["hard_text"]
            for x in data
            if x["profession"] == profession_dict[female_prof] and x["gender"] == 1
        ]
        n = min([len(neg_neg), len(neg_pos), len(pos_neg), len(pos_pos)])
        neg_neg, neg_pos, pos_neg, pos_pos = neg_neg[:n], neg_pos[:n], pos_neg[:n], pos_pos[:n]
        data = neg_neg + neg_pos + pos_neg + pos_pos
        true_labels = [0] * n + [0] * n + [1] * n + [1] * n
        spurious_labels = [0] * n + [1] * n + [0] * n + [1] * n
        idxs = list(range(4 * n))
        random.Random(seed).shuffle(idxs)
        data, true_labels, spurious_labels = (
            [data[i] for i in idxs],
            [true_labels[i] for i in idxs],
            [spurious_labels[i] for i in idxs],
        )
    else:
        neg_neg = [
            x["hard_text"]
            for x in data
            if x["profession"] == profession_dict[male_prof] and x["gender"] == 0
        ]
        neg_pos = [
            x["hard_text"]
            for x in data
            if x["profession"] == profession_dict[male_prof] and x["gender"] == 1
        ]
        pos_neg = [
            x["hard_text"]
            for x in data
            if x["profession"] == profession_dict[female_prof] and x["gender"] == 0
        ]
        pos_pos = [
            x["hard_text"]
            for x in data
            if x["profession"] == profession_dict[female_prof] and x["gender"] == 1
        ]
        n = min([len(neg_neg), len(neg_pos), len(pos_neg), len(pos_pos)])
        neg_neg, neg_pos, pos_neg, pos_pos = neg_neg[:n], neg_pos[:n], pos_neg[:n], pos_pos[:n]
        data = neg_neg + neg_pos + pos_neg + pos_pos
        true_labels = [0] * n + [1] * n + [0] * n + [1] * n
        spurious_labels = [0] * n + [0] * n + [1] * n + [1] * n
        idxs = list(range(4 * n))
        random.Random(seed).shuffle(idxs)
        data, true_labels, spurious_labels = (
            [data[i] for i in idxs],
            [true_labels[i] for i in idxs],
            [spurious_labels[i] for i in idxs],
        )

    batches = [
        (
            data[i : i + batch_size],
            t.tensor(true_labels[i : i + batch_size], device=DEVICE),
            t.tensor(spurious_labels[i : i + batch_size], device=DEVICE),
        )
        for i in range(0, len(data), batch_size)
    ]

    return batches


def get_effects_per_class(
    model: LanguageModel,
    submodules: list[utils.submodule_alias],
    dictionaries: dict[utils.submodule_alias, nn.Module],
    probes,
    probe_act_submodule: utils.submodule_alias,
    class_idx: int,
    train_bios,
    seed: int,
    device: str,
    batch_size: int = 10,
    patching_method: str = "ig",
    steps: int = 10,  # only used for ig
) -> dict[utils.submodule_alias, t.Tensor]:
    """
    Probe_act_submodule is the submodule where the probe is attached, usually resid_post
    """
    probe = probes[class_idx]

    if class_idx >= 0:
        texts_train, labels_train = get_class_samples(train_bios, class_idx, device)
        # texts_train, labels_train = get_class_nonclass_samples(train_bios, class_idx, device)
    else:
        texts_train, labels_train = get_paired_class_samples(train_bios, class_idx, device)

    texts_train = utils.batch_inputs(texts_train, batch_size)
    labels_train = utils.batch_inputs(labels_train, batch_size)

    running_total = 0
    running_nodes = None

    n_batches = len(texts_train)

    for batch_idx, (clean, labels) in enumerate(zip(texts_train, labels_train)):
        # for batch_idx, (clean, labels, _) in tqdm(
        #     enumerate(
        #         get_data(
        #             train=True, ambiguous=False, gender_balanced=True, batch_size=batch_size, seed=42
        #         )
        #     ),
        #     total=n_batches,
        # ):
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
    return nodes


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
    submodules,
    dictionaries,
    to_ablate,
    batch_size: int,
    probe_layer: int,
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
            act = model.gpt_neox.layers[probe_layer].output[0]
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
    node_effects: dict[int, dict[utils.submodule_alias, t.Tensor]],
    dict_size: int,
    T_effect: float = 0.001,
    verbose: bool = True,
    convert_to_n_hot: bool = True,
) -> dict[int, dict[utils.submodule_alias, t.Tensor]]:
    """This function is more idiomatic pytorch and doesn't have the bug of returning an empty dict."""
    # TODO: Switch over to this function, or maybe use the other one for the unique class features.
    feats_above_T = {}

    for abl_class_idx in node_effects.keys():
        total_features_per_abl_class = 0
        feats_above_T[abl_class_idx] = {}
        for submodule in node_effects[abl_class_idx].keys():
            feats_above_T[abl_class_idx][submodule] = (
                node_effects[abl_class_idx][submodule] > T_effect
            )
            total_features_per_abl_class += feats_above_T[abl_class_idx][submodule].sum().item()

        if verbose:
            print(
                f"T_effect {T_effect}, class {abl_class_idx}, all submodules, #significant features: {total_features_per_abl_class}"
            )

    return feats_above_T


def select_top_n_features(
    node_effects: dict[int, dict[utils.submodule_alias, t.Tensor]],
    n: int,
) -> dict[int, dict[utils.submodule_alias, t.Tensor]]:
    top_n_features = {}

    for abl_class_idx, submodules in node_effects.items():
        top_n_features[abl_class_idx] = {}

        for submodule, effects in submodules.items():
            assert (
                n <= effects.numel()
            ), f"n ({n}) must not be larger than the number of features ({effects.numel()}) for ablation class {abl_class_idx}, submodule {submodule}"

            # Get the indices of the top N effects
            _, top_indices = t.topk(effects, n)

            # Create a boolean mask tensor
            mask = t.zeros_like(effects, dtype=t.bool)
            mask[top_indices] = True

            top_n_features[abl_class_idx][submodule] = mask

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
    node_effects: dict,
    dict_size: int,
    T_effects: list[float],
    T_max_sideeffect: float,
    verbose: bool = False,
) -> dict[float, dict[int, dict[utils.submodule_alias, t.Tensor]]]:
    unique_feats = {}
    if selection_method == FeatureSelection.unique:
        for T_effect in T_effects:
            unique_feats[T_effect] = select_unique_class_features(
                node_effects,
                dict_size,
                T_effect=T_effect,
                T_max_sideeffect=T_max_sideeffect,
                verbose=verbose,
            )
    elif selection_method == FeatureSelection.above_threshold:
        for T_effect in T_effects:
            unique_feats[T_effect] = select_significant_features2(
                node_effects, dict_size, T_effect=T_effect, verbose=verbose
            )
    elif selection_method == FeatureSelection.top_n:
        for T_effect in T_effects:
            unique_feats[T_effect] = select_top_n_features(node_effects, T_effect)
    else:
        raise ValueError("Invalid selection method")

    return unique_feats


def save_log_files(ae_path: str, data: dict, base_filename: str, extension: str):
    # Always save/overwrite the main file
    main_file = os.path.join(ae_path, f"{base_filename}{extension}")
    with open(main_file, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved main file: {base_filename}{extension}")

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
    sweep_name: str,
    dictionaries_path: str,
    probes_dir: str,
    selection_method: FeatureSelection,
    probe_train_set_size: int,
    probe_test_set_size: int,
    train_set_size: int,
    test_set_size: int,
    probe_batch_size: int,
    llm_batch_size: int,
    eval_results_batch_size: int,
    patching_batch_size: int,
    T_effects: list[float],
    T_max_sideeffect: float,
    max_classes: int,
    random_seed: int,
    include_gender: bool,
    chosen_class_indices: Optional[list[int]] = None,
    device: str = "cuda",
    verbose: bool = False,
):
    t.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    model_eval_config = utils.ModelEvalConfig.from_sweep_name(sweep_name)
    model_name = model_eval_config.full_model_name

    # TODO: Better way to do this, maybe using d_model and context_length?
    if "160m" in model_name:
        llm_batch_size //= 2
        patching_batch_size //= 2

    model = LanguageModel(model_name, device_map=device, dispatch=True)

    probe_layer = model_eval_config.probe_layer
    probe_act_submodule = utils.get_submodule(model, "resid_post", probe_layer)

    ae_group_paths = utils.get_ae_group_paths(dictionaries_path, sweep_name, submodule_trainers)
    ae_paths = utils.get_ae_paths(ae_group_paths)

    # TODO: experiment with different context lengths
    context_length = utils.get_ctx_length(ae_paths)

    dataset, _ = load_and_prepare_dataset()
    train_bios, test_bios = get_train_test_data(
        dataset,
        train_set_size,
        test_set_size,
        include_gender,
        sort_by_length=True,
    )

    train_bios = utils.tokenize_data(train_bios, model.tokenizer, context_length, device)
    test_bios = utils.tokenize_data(test_bios, model.tokenizer, context_length, device)

    # TODO: Add batching so n_inputs is actually n_inputs
    eval_saes_n_inputs = 10000

    # This will only run eval_saes on autoencoders that don't yet have a eval_results.json file
    eval_saes.eval_saes(
        model,
        ae_paths,
        eval_saes_n_inputs,
        eval_results_batch_size,
        device,
        overwrite_prev_results=False,
    )

    only_model_name = model_name.split("/")[-1]
    probe_path = f"{probes_dir}/{only_model_name}/probes_ctx_len_{context_length}.pkl"

    # TODO: Add logic to ensure probes share keys with train_bios and test_bios
    if not os.path.exists(probe_path):
        print("Probes not found, training probes")
        probes = probe_training.train_probes(
            train_set_size=probe_train_set_size,
            test_set_size=probe_test_set_size,
            context_length=context_length,
            probe_batch_size=probe_batch_size,
            llm_batch_size=llm_batch_size,
            device=device,
            probe_dir=probes_dir,
            llm_model_name=model_name,
            epochs=10,
            include_gender=True,  # It's cheap to calculate and avoids potential bugs
        )

    with open(probe_path, "rb") as f:
        probes = pickle.load(f)

    if chosen_class_indices is not None:
        all_classes_list = chosen_class_indices
    else:
        all_classes_list = sorted(list(probes.keys()))[:max_classes]
    print(f"all_classes_list: {all_classes_list}")

    ### Get activations for original model, all classes
    print("Getting activations for original model")
    test_acts = {}
    for class_idx in tqdm(all_classes_list, desc="Getting activations per evaluated class"):
        test_acts[class_idx] = get_all_activations(
            test_bios[class_idx], model, llm_batch_size, probe_layer, context_length
        )

        if class_idx in utils.PAIRED_CLASS_KEYS:
            paired_class_idx = utils.PAIRED_CLASS_KEYS[class_idx]
            test_acts[paired_class_idx] = get_all_activations(
                test_bios[paired_class_idx], model, llm_batch_size, probe_layer, context_length
            )

    test_accuracies = probe_training.get_probe_test_accuracy(
        probes, all_classes_list, test_acts, probe_batch_size, verbose, device=device
    )
    # %%
    ### Get activations for ablated models
    # ablating the top features for each class
    print("Getting activations for ablated models")

    for ae_path in ae_paths:
        print(f"Running ablation for {ae_path}")
        submodules = []
        dictionaries = {}
        submodule, dictionary, config = utils.load_dictionary(model, ae_path, device)
        submodules.append(submodule)
        dictionaries[submodule] = dictionary
        dict_size = config["trainer"]["dict_size"]
        context_length = config["buffer"]["ctx_len"]

        # ae_name_lookup is useful if we are using attribution patching on multiple submodules
        ae_name_lookup = {submodule: ae_path}

        node_effects = {}
        class_accuracies = test_accuracies.copy()

        for ablated_class_idx in tqdm(all_classes_list, "Getting node effects"):
            node_effects[ablated_class_idx] = {}

            node_effects[ablated_class_idx] = get_effects_per_class(
                model,
                submodules,
                dictionaries,
                probes,
                probe_act_submodule,
                ablated_class_idx,
                train_bios,
                random_seed,
                device,
                batch_size=patching_batch_size,
                patching_method="attrib",
                # patching_method="ig",
                steps=5,
            )

        node_effects_cpu = utils.to_device(node_effects, "cpu")
        # Replace submodule keys with submodule_ae_path
        for abl_class_idx in node_effects_cpu.keys():
            node_effects_cpu[abl_class_idx] = {
                ae_name_lookup[submodule]: effects
                for submodule, effects in node_effects_cpu[abl_class_idx].items()
            }

        for key, value in node_effects[-2].items():
            node_effects_2 = node_effects[-2][key] > 0.1
            for idx in node_effects_2.nonzero():
                print(f"idx: {idx} value {node_effects[-2][key][idx]}")

        save_log_files(ae_path, node_effects_cpu, "node_effects", ".pkl")
        del node_effects_cpu
        t.cuda.empty_cache()
        gc.collect()

        unique_feats = select_features(
            selection_method, node_effects, dict_size, T_effects, T_max_sideeffect, verbose=verbose
        )

        for ablated_class_idx in all_classes_list:
            class_accuracies[ablated_class_idx] = {}
            print(f"evaluating class {ablated_class_idx}")

            for T_effect in T_effects:
                feats = unique_feats[T_effect][ablated_class_idx]

                if len(feats) == 0:
                    print(f"No features selected for T_effect = {T_effect}")
                    continue

                class_accuracies[ablated_class_idx][T_effect] = {}
                if verbose:
                    print(f"Running ablation for T_effect = {T_effect}")
                test_acts_ablated = {}
                for evaluated_class_idx in tqdm(all_classes_list, desc="Getting activations"):
                    test_acts_ablated[evaluated_class_idx] = get_all_acts_ablated(
                        test_bios[evaluated_class_idx],
                        model,
                        submodules,
                        dictionaries,
                        feats,
                        llm_batch_size,
                        probe_layer,
                    )

                    if evaluated_class_idx in utils.PAIRED_CLASS_KEYS:
                        paired_class_idx = utils.PAIRED_CLASS_KEYS[evaluated_class_idx]
                        test_acts_ablated[paired_class_idx] = get_all_acts_ablated(
                            test_bios[paired_class_idx],
                            model,
                            submodules,
                            dictionaries,
                            feats,
                            llm_batch_size,
                            probe_layer,
                        )

                for evaluated_class_idx in all_classes_list:
                    batch_test_acts, batch_test_labels = prepare_probe_data(
                        test_acts_ablated,
                        evaluated_class_idx,
                        probe_batch_size,
                        device=device,
                        single_class=False,
                    )
                    test_acc_probe = test_probe(
                        batch_test_acts,
                        batch_test_labels,
                        probes[evaluated_class_idx],
                        precomputed_acts=True,
                    )
                    if verbose:
                        print(
                            f"Ablated {ablated_class_idx}, evaluated {evaluated_class_idx} test accuracy: {test_acc_probe}"
                        )
                    class_accuracies[ablated_class_idx][T_effect][evaluated_class_idx] = (
                        test_acc_probe
                    )

                del test_acts_ablated
                del batch_test_acts
                del batch_test_labels
                t.cuda.empty_cache()
                gc.collect()

        class_accuracies = utils.to_device(class_accuracies, "cpu")

        save_log_files(ae_path, class_accuracies, "class_accuracies", ".pkl")


# %%

if __name__ == "__main__":
    selection_method = FeatureSelection.above_threshold
    selection_method = FeatureSelection.top_n

    random_seed = random.randint(0, 1000)
    num_classes = 5

    chosen_class_indices = [-4, -2, 0, 1, 2]
    # chosen_class_indices = [-4, -2, 0, 1]
    # chosen_class_indices = [-2]

    include_gender = True

    probe_train_set_size = 5000
    probe_test_set_size = 1000

    # Load datset and probes
    train_set_size = 500
    test_set_size = 500
    probe_batch_size = 500
    llm_batch_size = 250
    eval_results_batch_size = 100

    # Attribution patching variables
    patching_batch_size = 100

    reduced_GPU_memory = False

    if reduced_GPU_memory:
        probe_batch_size = 50
        llm_batch_size = 10
        patching_batch_size = 10

    top_n_features = [2, 5, 10, 20, 50, 100, 500]
    # top_n_features = [10, 50, 500, 1000]
    # top_n_features = [10]
    T_effects_all_classes = [0.1, 0.05, 0.025, 0.01, 0.001]
    T_effects_all_classes = [0.1, 0.01]
    T_effects_unique_class = [1e-4, 1e-8]

    if selection_method == FeatureSelection.top_n:
        T_effects = top_n_features
    elif selection_method == FeatureSelection.above_threshold:
        T_effects = T_effects_all_classes
    elif selection_method == FeatureSelection.unique:
        T_effects = T_effects_unique_class
    else:
        raise ValueError("Invalid selection method")

    T_max_sideeffect = 5e-3

    dictionaries_path = "../dictionary_learning/dictionaries"
    probes_dir = "trained_bib_probes"

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
            "resid_post_layer_3": {"trainer_ids": [10]},
        }
    }
    ae_sweep_paths = {
        "pythia70m_sweep_standard_ctx128_0712": {
            #     # "resid_post_layer_0": {"trainer_ids": None},
            #     # "resid_post_layer_1": {"trainer_ids": None},
            #     # "resid_post_layer_2": {"trainer_ids": None},
            "resid_post_layer_3": {"trainer_ids": [2, 6, 10, 18]},
            #     "resid_post_layer_4": {"trainer_ids": None},
        },
        "pythia70m_sweep_gated_ctx128_0730": {
            # "resid_post_layer_0": {"trainer_ids": None},
            # "resid_post_layer_1": {"trainer_ids": None},
            # "resid_post_layer_2": {"trainer_ids": None},
            "resid_post_layer_3": {"trainer_ids": [2, 6, 10, 18]},
            # "resid_post_layer_4": {"trainer_ids": None},
        },
        # "pythia70m_sweep_panneal_ctx128_0730": {
        #     # "resid_post_layer_0": {"trainer_ids": None},
        #     # "resid_post_layer_1": {"trainer_ids": None},
        #     # "resid_post_layer_2": {"trainer_ids": None},
        #     "resid_post_layer_3": {"trainer_ids": None},
        #     # "resid_post_layer_4": {"trainer_ids": None},
        # },
        "pythia70m_sweep_topk_ctx128_0730": {
            # "resid_post_layer_0": {"trainer_ids": None},
            # "resid_post_layer_1": {"trainer_ids": None},
            # "resid_post_layer_2": {"trainer_ids": None},
            "resid_post_layer_3": {"trainer_ids": [2, 6, 10, 18]},
            # "resid_post_layer_4": {"trainer_ids": None},
        },
    }

    # This will look for any empty folders in any ae_path and raise an error if it finds any
    for sweep_name, submodule_trainers in ae_sweep_paths.items():
        ae_group_paths = utils.get_ae_group_paths(dictionaries_path, sweep_name, submodule_trainers)

    start_time = time.time()

    for sweep_name, submodule_trainers in ae_sweep_paths.items():
        run_interventions(
            submodule_trainers,
            sweep_name,
            dictionaries_path,
            probes_dir,
            selection_method,
            probe_train_set_size,
            probe_test_set_size,
            train_set_size,
            test_set_size,
            probe_batch_size,
            llm_batch_size,
            eval_results_batch_size,
            patching_batch_size,
            T_effects,
            T_max_sideeffect,
            num_classes,
            random_seed,
            include_gender=include_gender,
            chosen_class_indices=chosen_class_indices,
            verbose=True,
        )

    end_time = time.time()

    print(f"Time taken: {end_time - start_time} seconds")

# %%
