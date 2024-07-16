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

# labels[0] = true labels (profession)
# labels[1] = spurious labels (gender)


def metric_fn(model, labels=None, probe=None, probe_layer=None):
    attn_mask = model.input[1]["attention_mask"]
    acts = model.gpt_neox.layers[probe_layer].output[0]
    acts = acts * attn_mask[:, :, None]
    acts = acts.sum(1) / attn_mask.sum(1)[:, None]

    return t.where(labels == 0, probe(acts), -probe(acts))


# Attribution Patching


def get_class_nonclass_samples(
    data: dict, class_idx: int, batch_size: int, device: str, seed: int
) -> tuple[list, t.Tensor]:
    """This is for getting equal number of text samples from the chosen class and all other classes.
    We use this for attribution patching."""
    class_samples = data[class_idx]
    nonclass_samples = []

    for profession in data:
        if profession != class_idx:
            nonclass_samples.extend(data[profession])

    nonclass_samples = random.sample(nonclass_samples, len(class_samples))

    combined_samples = class_samples + nonclass_samples
    combined_labels = t.zeros(len(combined_samples), device=device)
    combined_labels[: len(class_samples)] = 1

    batched_samples = utils.batch_list(combined_samples, batch_size)
    batched_labels = utils.batch_list(combined_labels, batch_size)

    return batched_samples, batched_labels


def get_effects_per_class(
    model,
    submodules,
    dictionaries,
    probes,
    class_idx,
    train_bios,
    seed: int,
    probe_layer: int,
    device: str,
    n_batches=None,
    batch_size=10,
    patching_method="ig",
    steps=10,
) -> dict[utils.submodule_alias, t.Tensor]:
    probe = probes[class_idx]
    texts_train, labels_train = get_class_nonclass_samples(
        train_bios, class_idx, batch_size, device, seed
    )
    if n_batches is not None:
        if len(texts_train) > n_batches:
            texts_train = texts_train[:n_batches]
            labels_train = labels_train[:n_batches]

    running_total = 0
    running_nodes = None

    for batch_idx, (clean, labels) in tqdm(
        enumerate(zip(texts_train, labels_train)), total=n_batches
    ):
        if batch_idx == n_batches:
            break

        effects, _, _, _ = patching_effect(
            clean,
            None,
            model,
            submodules,
            dictionaries,
            metric_fn,
            metric_kwargs=dict(labels=labels, probe=probe, probe_layer=probe_layer),
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

    text_batches = utils.batch_list(text_inputs, batch_size)
    assert type(text_batches[0][0]) == str

    is_tuple = {}
    with t.no_grad(), model.trace("_"):
        for submodule in submodules:
            is_tuple[submodule] = type(submodule.output.shape) == tuple

    all_acts_list_BD = []
    for text_batch_BL in tqdm(text_batches, desc="Getting activations"):
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
def n_hot(feats, dim):
    out = t.zeros(dim, dtype=t.bool, device=DEVICE)
    for feat in feats:
        out[feat] = True
    return out


def select_significant_features(
    node_effects: dict[int, dict[utils.submodule_alias, t.Tensor]],
    dict_size: int,
    T_effect: float = 0.001,
    verbose: bool = True,
    convert_to_n_hot: bool = True,
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
                submodule: n_hot(feats, activation_dim)
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
            submodule: n_hot(feats, dict_size)
            for submodule, feats in feats_above_T[abl_class_idx].items()
        }
        if verbose:
            print(
                f"T_effect {T_effect}, class {abl_class_idx}, all submodules, #unique features: {total_features_per_abl_class}"
            )

    return feats_above_T

    top_feats_to_ablate = {}
    total_features = 0
    for submodule in submodules:
        sideeffect_features = []
        for other_submodule in submodules:
            if other_submodule != submodule:
                print(non_neglectable_effects[other_submodule])
                print(len(non_neglectable_effects[other_submodule]))
                sideeffect_features.extend(non_neglectable_effects[other_submodule])
        print(len(sideeffect_features))
        sideeffect_features = set(sideeffect_features)
        if verbose:
            print(f"sideeffect features: {len(sideeffect_features)}")

        # Get the top features to ablate for this submodule
        top_feats_to_ablate[submodule] = []
        for feat in significant_effects[submodule]:
            if feat not in sideeffect_features:
                top_feats_to_ablate[submodule].append(feat)
                total_features += 1
    if verbose:
        print(f"total ablation features: {total_features}")

    top_feats_to_ablate = {
        submodule: n_hot(feats, dict_size) for submodule, feats in top_feats_to_ablate.items()
    }
    return top_feats_to_ablate


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
            unique_feats[T_effect] = select_significant_features(
                node_effects, dict_size, T_effect=T_effect, verbose=verbose
            )
    elif selection_method == FeatureSelection.top_n:
        for T_effect in T_effects:
            unique_feats[T_effect] = select_top_n_features(node_effects, T_effect)
    else:
        raise ValueError("Invalid selection method")

    return unique_feats


## Plotting functions


def plot_feature_effects_above_threshold(nodes, threshold=0.05):
    all_values = []
    for key in nodes.keys():
        all_values.append(nodes[key].cpu().numpy().reshape(-1))
    all_values = [x for sublist in all_values for x in sublist]
    all_values = [x for x in all_values if x > threshold]

    all_values = sorted(all_values, reverse=True)
    plt.scatter(range(len(all_values)), all_values)
    plt.title("all_values")
    plt.show()


# %%


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
    n_eval_batches: int,
    patching_batch_size: int,
    T_effects: list[float],
    T_max_sideeffect: float,
    max_classes: int,
    random_seed: int,
    device: str = "cuda",
    verbose: bool = False,
):
    t.manual_seed(random_seed)

    model_eval_config = utils.ModelEvalConfig.from_sweep_name(sweep_name)
    model_name = model_eval_config.full_model_name

    model = LanguageModel(model_name, device_map=device, dispatch=True)

    probe_layer = model_eval_config.probe_layer

    ae_group_paths = utils.get_ae_group_paths(dictionaries_path, sweep_name, submodule_trainers)
    ae_paths = utils.get_ae_paths(ae_group_paths)

    # TODO: experiment with different context lengths
    context_length = utils.get_ctx_length(ae_paths)

    dataset, _ = load_and_prepare_dataset()
    train_bios, test_bios = get_train_test_data(dataset, train_set_size, test_set_size)

    train_bios = utils.trim_bios_to_context_length(train_bios, context_length)
    test_bios = utils.trim_bios_to_context_length(test_bios, context_length)

    # TODO: Add batching so n_inputs is actually n_inputs
    eval_saes_n_inputs = llm_batch_size * 10

    # This will only run eval_saes on autoencoders that don't yet have a eval_results.json file
    eval_saes.eval_saes(
        model,
        ae_paths,
        eval_saes_n_inputs,
        context_length,
        llm_batch_size,
        device,
        overwrite_prev_results=False,
    )

    probe_path = f"{probes_dir}/probes_ctx_len_{context_length}.pkl"

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
        )

    with open(probe_path, "rb") as f:
        probes = pickle.load(f)

    all_classes_list = list(probes.keys())[:max_classes]

    ### Get activations for original model, all classes
    print("Getting activations for original model")
    test_acts = {}
    for class_idx in tqdm(all_classes_list, desc="Getting activations per evaluated class"):
        class_test_acts = get_all_activations(
            test_bios[class_idx], model, llm_batch_size, probe_layer
        )
        test_acts[class_idx] = class_test_acts

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

        for ablated_class_idx in all_classes_list:
            node_effects[ablated_class_idx] = {}

            node_effects[ablated_class_idx] = get_effects_per_class(
                model,
                submodules,
                dictionaries,
                probes,
                ablated_class_idx,
                train_bios,
                random_seed,
                probe_layer,
                device,
                n_eval_batches,
                batch_size=patching_batch_size,
                patching_method="attrib",
                steps=10,
            )

        node_effects_cpu = utils.to_device(node_effects, "cpu")
        # Replace submodule keys with submodule_ae_path
        for abl_class_idx in node_effects_cpu.keys():
            node_effects_cpu[abl_class_idx] = {
                ae_name_lookup[submodule]: effects
                for submodule, effects in node_effects_cpu[abl_class_idx].items()
            }
        with open(ae_path + "node_effects.pkl", "wb") as f:
            pickle.dump(node_effects_cpu, f)
        del node_effects_cpu
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
                for evaluated_class_idx in all_classes_list:
                    test_acts_ablated[evaluated_class_idx] = get_all_acts_ablated(
                        test_bios[evaluated_class_idx],
                        model,
                        submodules,
                        dictionaries,
                        feats,
                        llm_batch_size,
                        probe_layer,
                    )

                for evaluated_class_idx in all_classes_list:
                    batch_test_acts, batch_test_labels = prepare_probe_data(
                        test_acts_ablated, evaluated_class_idx, probe_batch_size, device=device
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
                    class_accuracies[ablated_class_idx][T_effect][
                        evaluated_class_idx
                    ] = test_acc_probe

                del test_acts_ablated
                del batch_test_acts
                del batch_test_labels
                t.cuda.empty_cache()
                gc.collect()

        class_accuracies = utils.to_device(class_accuracies, "cpu")
        with open(ae_path + "class_accuracies.pkl", "wb") as f:
            pickle.dump(class_accuracies, f)

    # %%


if __name__ == "__main__":

    selection_method = FeatureSelection.above_threshold
    selection_method = FeatureSelection.top_n

    random_seed = random.randint(0, 1000)
    num_classes = 5

    probe_train_set_size = 5000
    probe_test_set_size = 1000

    # Load datset and probes
    train_set_size = 1000
    test_set_size = 1000
    probe_batch_size = 50
    llm_batch_size = 125
    # llm_batch_size = 10

    # Attribution patching variables
    n_eval_batches = 4
    patching_batch_size = 50
    # patching_batch_size = 5

    top_n_features = [5, 10, 20, 50, 100, 500]
    top_n_features = [5, 500]
    T_effects_all_classes = [0.1, 0.01, 0.005, 0.001]
    # T_effects_all_classes = [0.001]
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

    # Example of sweeping over all SAEs in a sweep
    ae_sweep_paths = {"pythia70m_test_sae": None}

    # Example of sweeping over all SAEs in a submodule
    ae_sweep_paths = {"pythia70m_test_sae": {"resid_post_layer_3": None}}

    # Example of sweeping over a single SAE
    ae_sweep_paths = {"pythia70m_test_sae": {"resid_post_layer_3": {"trainer_ids": [0]}}}

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
            n_eval_batches,
            patching_batch_size,
            T_effects,
            T_max_sideeffect,
            num_classes,
            random_seed,
        )
