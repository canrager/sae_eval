# %%
# Imports
import sys
import os
import random
import gc
from collections import defaultdict
import einops
import math
import numpy as np
import pickle

import torch as t
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tqdm import tqdm
from typing import Callable, Optional

from datasets import load_dataset
import datasets
from nnsight import LanguageModel

import experiments.utils as utils
import experiments.dataset_info as dataset_info

# Configuration
DEBUGGING = False
SEED = 42
MAXIMUM_SPUROUS_TRAIN_SET_SIZE = 5000

# Set up paths and model
parent_dir = os.path.abspath("..")
sys.path.append(parent_dir)

tracer_kwargs = dict(scan=DEBUGGING, validate=DEBUGGING)


def ensure_shared_keys(train_data: dict, test_data: dict) -> tuple[dict, dict]:
    # Find keys that are in test but not in train
    test_only_keys = set(test_data.keys()) - set(train_data.keys())

    # Find keys that are in train but not in test
    train_only_keys = set(train_data.keys()) - set(test_data.keys())

    # Remove keys from test that are not in train
    for key in test_only_keys:
        print(f"Removing {key} from test set")
        del test_data[key]

    # Remove keys from train that are not in test
    for key in train_only_keys:
        print(f"Removing {key} from train set")
        del train_data[key]

    return train_data, test_data


# Load and prepare dataset
def load_and_prepare_dataset(dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataset_name == "bias_in_bios":
        dataset = load_dataset("LabHC/bias_in_bios")
        train_df = pd.DataFrame(dataset["train"])
        test_df = pd.DataFrame(dataset["test"])
    elif dataset_name == "amazon_reviews_all_ratings":
        dataset = load_dataset(
            "canrager/amazon_reviews_mcauley",
            config_name="dataset_all_categories_and_ratings_train1000_test250",
        )
    elif dataset_name == "amazon_reviews_1and5":
        dataset = load_dataset(
            "canrager/amazon_reviews_mcauley_1and5",
        )
        train_df = pd.DataFrame(dataset["train"])
        test_df = pd.DataFrame(dataset["test"])
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return train_df, test_df


def get_spurious_corr_data(
    df: pd.DataFrame,
    column1_vals: tuple[str, str],
    column2_vals: tuple[str, str],
    dataset_name: str,
    min_samples_per_quadrant: int,
    max_spurious_train_set_size: bool,
    random_seed: int,
) -> dict[str, list[str]]:
    """Returns a dataset of, in the case of bias_in_bios, a key that's something like `female_nurse_data_only`,
    and a value that's a list of bios (strs) of len min_samples_per_quadrant * 2.
    If max_spurious_train_set_size is True, then the spurious correlation dataset size will be larger,
    limited to a max of 10,000 samples."""
    balanced_data = {}

    text_column_name = dataset_info.dataset_metadata[dataset_name]["text_column_name"]
    column1_name = dataset_info.dataset_metadata[dataset_name]["column1_name"]
    column2_name = dataset_info.dataset_metadata[dataset_name]["column2_name"]

    column1_pos = column1_vals[0]
    column1_neg = column1_vals[1]
    column2_pos = column2_vals[0]
    column2_neg = column2_vals[1]

    # NOTE: This is a bit confusing. We select rows from the dataset based on column1_vals and column2_vals,
    # but below, we hardcode the keys as male / female, professor / nurse, etc
    column1_pos_idx = dataset_info.dataset_metadata[dataset_name]["column1_mapping"][column1_pos]
    column1_neg_idx = dataset_info.dataset_metadata[dataset_name]["column1_mapping"][column1_neg]
    column2_pos_idx = dataset_info.dataset_metadata[dataset_name]["column2_mapping"][column2_pos]
    column2_neg_idx = dataset_info.dataset_metadata[dataset_name]["column2_mapping"][column2_neg]

    pos_neg = df[(df[column1_name] == column1_neg_idx) & (df[column2_name] == column2_pos_idx)][
        text_column_name
    ].tolist()
    neg_neg = df[(df[column1_name] == column1_neg_idx) & (df[column2_name] == column2_neg_idx)][
        text_column_name
    ].tolist()

    pos_pos = df[(df[column1_name] == column1_pos_idx) & (df[column2_name] == column2_pos_idx)][
        text_column_name
    ].tolist()
    neg_pos = df[(df[column1_name] == column1_pos_idx) & (df[column2_name] == column2_neg_idx)][
        text_column_name
    ].tolist()

    min_count = min(
        len(pos_neg), len(neg_neg), len(pos_pos), len(neg_pos), min_samples_per_quadrant
    )

    print(f"min_count: {min_count}, min_samples_per_quadrant: {min_samples_per_quadrant}")
    assert min_count == min_samples_per_quadrant

    # For biased classes, we don't have two quadrants per label
    assert len(pos_pos) > min_samples_per_quadrant * 2
    assert len(neg_neg) > min_samples_per_quadrant * 2

    biased_count = min(len(neg_neg), len(pos_pos), MAXIMUM_SPUROUS_TRAIN_SET_SIZE)
    print(f"biased_count: {biased_count}")

    rng = np.random.default_rng(random_seed)

    # Create and shuffle combinations
    combined_pos = pos_pos[:min_count] + pos_neg[:min_count]
    combined_neg = neg_pos[:min_count] + neg_neg[:min_count]
    pos_combined = pos_pos[:min_count] + neg_pos[:min_count]
    neg_combined = pos_neg[:min_count] + neg_neg[:min_count]
    pos_pos = pos_pos[:biased_count]
    neg_neg = neg_neg[:biased_count]

    # Shuffle each combination
    rng.shuffle(combined_pos)
    rng.shuffle(combined_neg)
    rng.shuffle(pos_combined)
    rng.shuffle(neg_combined)
    rng.shuffle(pos_pos)
    rng.shuffle(neg_neg)

    # Assign to balanced_data
    balanced_data["male / female"] = combined_pos  # male data only, to be combined with female data
    balanced_data["female_data_only"] = combined_neg  # female data only
    balanced_data["professor / nurse"] = (
        pos_combined  # professor data only, to be combined with nurse data
    )
    balanced_data["nurse_data_only"] = neg_combined  # nurse data only
    balanced_data["male_professor / female_nurse"] = (
        pos_pos  # male_professor data only, to be combined with female_nurse data
    )
    balanced_data["female_nurse_data_only"] = neg_neg  # female_nurse data only

    for key in balanced_data.keys():
        if max_spurious_train_set_size:
            if "female_nurse" in key:
                balanced_data[key] = balanced_data[key][:MAXIMUM_SPUROUS_TRAIN_SET_SIZE]
                assert len(balanced_data[key]) == MAXIMUM_SPUROUS_TRAIN_SET_SIZE
                continue
        balanced_data[key] = balanced_data[key][: min_samples_per_quadrant * 2]
        assert len(balanced_data[key]) == min_samples_per_quadrant * 2

    return balanced_data


# Dataset balancing and preparation
def get_balanced_dataset_tpp(
    df: pd.DataFrame,
    dataset_name: str,
    min_samples_per_quadrant: int,
    random_seed: int = SEED,
):
    """Returns a dataset of, in the case of bias_in_bios, a key of profession idx,
    and a value of a list of bios (strs) of len min_samples_per_quadrant * 2."""

    text_column_name = dataset_info.dataset_metadata[dataset_name]["text_column_name"]
    column1_name = dataset_info.dataset_metadata[dataset_name]["column1_name"]
    column2_name = dataset_info.dataset_metadata[dataset_name]["column2_name"]

    balanced_df_list = []

    for profession in tqdm(df[column1_name].unique()):
        prof_df = df[df[column1_name] == profession]
        min_count = prof_df[column2_name].value_counts().min()

        if min_count < min_samples_per_quadrant:
            continue

        balanced_prof_df = pd.concat(
            [
                group.sample(n=min_samples_per_quadrant, random_state=random_seed)
                for _, group in prof_df.groupby(column2_name)
            ]
        ).reset_index(drop=True)
        balanced_df_list.append(balanced_prof_df)

    balanced_df = pd.concat(balanced_df_list).reset_index(drop=True)
    grouped = balanced_df.groupby(column1_name)[text_column_name].apply(list)

    balanced_data = {label: texts for label, texts in grouped.items()}

    for key in balanced_data.keys():
        balanced_data[key] = balanced_data[key][: min_samples_per_quadrant * 2]
        assert len(balanced_data[key]) == min_samples_per_quadrant * 2

    return balanced_data


def get_train_test_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
    spurious_corr: bool,
    train_set_size: int,
    test_set_size: int,
    random_seed: int,
    column1_vals: Optional[tuple[str, str]] = None,
    column2_vals: Optional[tuple[str, str]] = None,
    use_max_possible_spurious_train_set: bool = False,
) -> tuple[dict, dict]:
    # 4 is because male / gender for each profession
    minimum_train_samples_per_quadrant = train_set_size // 4
    minimum_test_samples_per_quadrant = test_set_size // 4

    if spurious_corr:
        train_bios = get_spurious_corr_data(
            train_df,
            column1_vals,
            column2_vals,
            dataset_name,
            minimum_train_samples_per_quadrant,
            use_max_possible_spurious_train_set,
            random_seed,
        )

        test_bios = get_spurious_corr_data(
            test_df,
            column1_vals,
            column2_vals,
            dataset_name,
            minimum_test_samples_per_quadrant,
            False,
            random_seed,
        )

    else:
        train_bios = get_balanced_dataset_tpp(
            train_df,
            dataset_name,
            minimum_train_samples_per_quadrant,
            random_seed=random_seed,
        )
        test_bios = get_balanced_dataset_tpp(
            test_df,
            dataset_name,
            minimum_test_samples_per_quadrant,
            random_seed=random_seed,
        )

    train_bios, test_bios = ensure_shared_keys(train_bios, test_bios)

    return train_bios, test_bios


# Probe model and training
class Probe(nn.Module):
    def __init__(self, activation_dim: int, dtype: t.dtype):
        super().__init__()
        self.net = nn.Linear(activation_dim, 1, bias=True, dtype=dtype)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# Deprecated
def get_acts(text):
    with t.no_grad():
        with model.trace(text, **tracer_kwargs):
            attn_mask = model.input[1]["attention_mask"]
            acts = model.gpt_neox.layers[LAYER].output[0]
            acts = acts * attn_mask[:, :, None]
            acts = acts.sum(1) / attn_mask.sum(1)[:, None]
            acts = acts.save()
        return acts.value


@t.no_grad()
def get_all_meaned_activations(
    text_inputs: dict[str, t.Tensor] | list[str],
    model: LanguageModel,
    batch_size: int,
    submodule: utils.submodule_alias,
) -> t.Tensor:
    # TODO: Rename text_inputs
    text_batches = utils.batch_inputs(text_inputs, batch_size)

    all_acts_list_BD = []
    for text_batch_BL in text_batches:
        with model.trace(
            text_batch_BL,
            **tracer_kwargs,
        ):
            attn_mask = model.input[1]["attention_mask"]
            acts_BLD = submodule.output[0]
            acts_BLD = acts_BLD * attn_mask[:, :, None]
            acts_BD = acts_BLD.sum(1) / attn_mask.sum(1)[:, None]
            acts_BD = acts_BD.save()
        all_acts_list_BD.append(acts_BD.value)

    all_acts_bD = t.cat(all_acts_list_BD, dim=0)
    return all_acts_bD


@t.no_grad()
def get_all_activations(
    text_inputs: dict[str, t.Tensor] | list[str],
    model: LanguageModel,
    batch_size: int,
    submodule: utils.submodule_alias,
) -> t.Tensor:
    # TODO: Rename text_inputs
    text_batches = utils.batch_inputs(text_inputs, batch_size)

    all_acts_list_BLD = []

    for text_batch_BL in text_batches:
        with model.trace(
            text_batch_BL,
            **tracer_kwargs,
        ):
            attn_mask = model.input[1]["attention_mask"]
            acts_BLD = submodule.output[0]
            acts_BLD = acts_BLD * attn_mask[:, :, None]
            acts_BLD = acts_BLD.save()

        all_acts_list_BLD.append(acts_BLD.value)

    all_acts_bLD = t.cat(all_acts_list_BLD, dim=0)

    return all_acts_bLD


def get_activation_distribution_diff(
    all_activations: dict[int | str, t.Tensor], class_idx: int | str
) -> t.Tensor:
    positive_acts_BD = all_activations[class_idx]

    positive_distribution_D = positive_acts_BD.mean(dim=0)

    if isinstance(class_idx, int):
        # Calculate negative distribution without concatenation
        negative_sum = t.zeros_like(positive_distribution_D)
        negative_count = 0
        for idx, acts in all_activations.items():
            if idx != class_idx and isinstance(idx, int):
                negative_sum += acts.sum(dim=0)
                negative_count += acts.shape[0]
        negative_distribution_D = negative_sum / negative_count
        distribution_diff_D = positive_distribution_D - negative_distribution_D
    else:
        if class_idx not in utils.PAIRED_CLASS_KEYS:
            raise ValueError(f"Class index {class_idx} is not a valid class index.")

        negative_acts = all_activations[utils.PAIRED_CLASS_KEYS[class_idx]]
        negative_distribution_D = negative_acts.mean(dim=0)

        distribution_diff_D = (positive_distribution_D - negative_distribution_D).abs()

    return distribution_diff_D


def prepare_probe_data(
    all_activations: dict[int | str, t.Tensor],
    class_idx: int | str,
    spurious_corr: bool,
    batch_size: int,
    select_top_k: Optional[int] = None,  # experimental feature
) -> tuple[list[t.Tensor], list[t.Tensor]]:
    """If class_idx is a string, there is a paired class idx in utils.py."""
    positive_acts_BD = all_activations[class_idx]
    device = positive_acts_BD.device

    num_positive = len(positive_acts_BD)

    if spurious_corr:
        assert isinstance(class_idx, str)
        if class_idx not in utils.PAIRED_CLASS_KEYS:
            raise ValueError(f"Class index {class_idx} is not a valid class index.")

        negative_acts = all_activations[utils.PAIRED_CLASS_KEYS[class_idx]]
    else:
        assert isinstance(class_idx, int)
        # Collect all negative class activations and labels
        negative_acts = []
        for idx, acts in all_activations.items():
            assert isinstance(idx, int)
            if idx != class_idx:
                negative_acts.append(acts)

        negative_acts = t.cat(negative_acts)

    # Randomly select num_positive samples from negative class
    indices = t.randperm(len(negative_acts))[:num_positive]
    selected_negative_acts_BD = negative_acts[indices]

    assert selected_negative_acts_BD.shape == positive_acts_BD.shape

    # Experimental feature: find the top k features that differ the most between in distribution and out of distribution
    # zero out the rest. Useful for k-sparse probing experiments.
    if select_top_k is not None:
        positive_distribution_D = positive_acts_BD.mean(dim=(0))
        negative_distribution_D = negative_acts.mean(dim=(0))
        distribution_diff_D = (positive_distribution_D - negative_distribution_D).abs()
        top_k_indices_D = t.argsort(distribution_diff_D, descending=True)[:select_top_k]

        mask_D = t.ones(distribution_diff_D.shape[0], dtype=t.bool, device=positive_acts_BD.device)
        mask_D[top_k_indices_D] = False

        masked_positive_acts_BD = positive_acts_BD.clone()
        masked_negative_acts_BD = selected_negative_acts_BD.clone()

        masked_positive_acts_BD[:, mask_D] = 0.0
        masked_negative_acts_BD[:, mask_D] = 0.0
    else:
        masked_positive_acts_BD = positive_acts_BD
        masked_negative_acts_BD = selected_negative_acts_BD

    # Combine positive and negative samples
    combined_acts = t.cat([masked_positive_acts_BD, masked_negative_acts_BD])

    combined_labels = t.empty(len(combined_acts), dtype=t.int, device=device)
    combined_labels[:num_positive] = utils.POSITIVE_CLASS_LABEL
    combined_labels[num_positive:] = utils.NEGATIVE_CLASS_LABEL

    # Shuffle the combined data
    shuffle_indices = t.randperm(len(combined_acts))
    shuffled_acts = combined_acts[shuffle_indices]
    shuffled_labels = combined_labels[shuffle_indices]

    # Reshape into lists of tensors with specified batch_size
    num_samples = len(shuffled_acts)
    num_batches = num_samples // batch_size

    batched_acts = [
        shuffled_acts[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)
    ]
    batched_labels = [
        shuffled_labels[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)
    ]

    return batched_acts, batched_labels


def train_probe(
    train_input_batches: list,
    train_label_batches: list[t.Tensor],
    test_input_batches: list,
    test_label_batches: list[t.Tensor],
    get_acts: Callable,
    precomputed_acts: bool,
    dim: int,
    epochs: int,
    device: str,
    model_dtype: t.dtype,
    lr: float = 1e-3,
    seed: int = SEED,
) -> tuple[Probe, float]:
    """input_batches can be a list of tensors or strings. If strings, get_acts must be provided."""

    if type(train_input_batches[0]) == str or type(test_input_batches[0]) == str:
        assert precomputed_acts == False
    elif type(train_input_batches[0]) == t.Tensor or type(test_input_batches[0]) == t.Tensor:
        assert precomputed_acts == True

    probe = Probe(dim, model_dtype).to(device)
    optimizer = t.optim.AdamW(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        batch_idx = 0
        for inputs, labels in zip(train_input_batches, train_label_batches):
            if precomputed_acts:
                acts_BD = inputs
            else:
                acts_BD = get_acts(inputs)
            logits_B = probe(acts_BD)
            loss = criterion(logits_B, labels.clone().detach().to(device=device, dtype=model_dtype))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_idx += 1
        print(f"\nEpoch {epoch + 1}/{epochs} Loss: {loss.item()}")

        train_accuracy = test_probe(
            train_input_batches[:30], train_label_batches[:30], probe, get_acts, precomputed_acts
        )

        print(f"Train Accuracy: {train_accuracy}")

        test_accuracy = test_probe(
            test_input_batches, test_label_batches, probe, get_acts, precomputed_acts
        )
        print(f"Test Accuracy: {test_accuracy}")
    return probe, test_accuracy


def test_probe(
    input_batches: list,
    label_batches: list[t.Tensor],
    probe: Probe,
    precomputed_acts: bool,
    get_acts: Optional[Callable] = None,
) -> tuple[float, float, float, float]:
    if precomputed_acts is True:
        assert get_acts is None, "get_acts will not be used if precomputed_acts is True."

    criterion = nn.BCEWithLogitsLoss()

    with t.no_grad():
        corrects_0 = []
        corrects_1 = []
        all_corrects = []
        losses = []

        for input_batch, labels_B in zip(input_batches, label_batches):
            if precomputed_acts:
                acts_BD = input_batch
            else:
                raise NotImplementedError("Currently deprecated.")
                acts_BD = get_acts(input_batch)

            logits_B = probe(acts_BD)
            preds_B = (logits_B > 0.0).long()
            correct_B = (preds_B == labels_B).float()

            all_corrects.append(correct_B)
            corrects_0.append(correct_B[labels_B == 0])
            corrects_1.append(correct_B[labels_B == 1])

            loss = criterion(logits_B, labels_B.to(dtype=probe.net.weight.dtype))
            losses.append(loss)

        accuracy_all = t.cat(all_corrects).mean().item()
        accuracy_0 = t.cat(corrects_0).mean().item() if corrects_0 else 0.0
        accuracy_1 = t.cat(corrects_1).mean().item() if corrects_1 else 0.0
        loss = t.stack(losses).mean().item()

    return accuracy_all, accuracy_0, accuracy_1, loss


def get_probe_test_accuracy(
    probes: dict[str | int, Probe],
    all_class_list: list[str | int],
    all_activations: dict[str | int, t.Tensor],
    probe_batch_size: int,
    spurious_corr: bool,
) -> dict[str, dict[str, float]]:
    test_accuracies = {}
    for class_name in all_class_list:
        batch_test_acts, batch_test_labels = prepare_probe_data(
            all_activations, class_name, spurious_corr, probe_batch_size
        )
        test_acc_probe, acc_0, acc_1, loss = test_probe(
            batch_test_acts, batch_test_labels, probes[class_name], precomputed_acts=True
        )
        test_accuracies[class_name] = {
            "acc": test_acc_probe,
            "acc_0": acc_0,
            "acc_1": acc_1,
            "loss": loss,
        }

    # Tests e.g. male_professor / female_nurse probe on professor / nurse labels
    if spurious_corr:
        for class_name in all_class_list:
            if class_name not in utils.PAIRED_CLASS_KEYS:
                continue
            spurious_class_names = [key for key in utils.PAIRED_CLASS_KEYS if key != class_name]
            batch_test_acts, batch_test_labels = prepare_probe_data(
                all_activations, class_name, spurious_corr, probe_batch_size
            )

            for spurious_class_name in spurious_class_names:
                test_acc_probe, acc_0, acc_1, loss = test_probe(
                    batch_test_acts,
                    batch_test_labels,
                    probes[spurious_class_name],
                    precomputed_acts=True,
                )
                combined_class_name = f"{spurious_class_name} probe on {class_name} data"
                print(f"Test accuracy for {combined_class_name}: {test_acc_probe}")
                test_accuracies[combined_class_name] = {
                    "acc": test_acc_probe,
                    "acc_0": acc_0,
                    "acc_1": acc_1,
                    "loss": loss,
                }

    return test_accuracies


# Main execution
def train_probes(
    train_set_size: int,
    test_set_size: int,
    model: LanguageModel,
    context_length: int,
    probe_train_batch_size: int,
    probe_test_batch_size: int,
    llm_batch_size: int,
    device: str,
    probe_output_filename: str,
    spurious_correlation_removal: bool,
    probe_layer: int,
    epochs: int,
    llm_model_name: str,
    chosen_class_indices: Optional[list[str | int]] = None,  # only required for tpp
    dataset_name: str = "bias_in_bios",
    probe_dir: str = "trained_bib_probes",
    model_dtype: t.dtype = t.bfloat16,
    save_results: bool = True,
    seed: int = SEED,
    column1_vals: Optional[tuple[str, str]] = None,
    column2_vals: Optional[tuple[str, str]] = None,
) -> dict[int, float]:
    t.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model_eval_config = utils.ModelEvalConfig.from_full_model_name(llm_model_name)
    d_model = model_eval_config.activation_dim

    probe_act_submodule = utils.get_submodule(model, "resid_post", probe_layer)

    train_df, test_df = load_and_prepare_dataset(dataset_name)

    train_bios, test_bios = get_train_test_data(
        train_df=train_df,
        test_df=test_df,
        dataset_name=dataset_name,
        spurious_corr=spurious_correlation_removal,
        train_set_size=train_set_size,
        test_set_size=test_set_size,
        random_seed=seed,
        column1_vals=column1_vals,
        column2_vals=column2_vals,
        use_max_possible_spurious_train_set=True,
    )

    if not spurious_correlation_removal:
        train_bios = utils.filter_dataset(train_bios, chosen_class_indices)
        test_bios = utils.filter_dataset(test_bios, chosen_class_indices)

    train_bios = utils.tokenize_data(train_bios, model.tokenizer, context_length, device)
    test_bios = utils.tokenize_data(test_bios, model.tokenizer, context_length, device)

    probes, test_accuracies = {}, {}

    all_train_acts = {}
    all_test_acts = {}

    t.set_grad_enabled(False)

    with t.no_grad():
        for i, class_name in enumerate(train_bios.keys()):
            print(f"Collecting activations for profession: {class_name}")

            all_train_acts[class_name] = get_all_meaned_activations(
                train_bios[class_name], model, llm_batch_size, probe_act_submodule
            )
            all_test_acts[class_name] = get_all_meaned_activations(
                test_bios[class_name], model, llm_batch_size, probe_act_submodule
            )

    t.set_grad_enabled(True)

    for class_name in all_train_acts.keys():
        if class_name in utils.PAIRED_CLASS_KEYS.values():
            continue

        train_acts, train_labels = prepare_probe_data(
            all_train_acts, class_name, spurious_correlation_removal, probe_train_batch_size
        )

        test_acts, test_labels = prepare_probe_data(
            all_test_acts, class_name, spurious_correlation_removal, probe_test_batch_size
        )

        probe, test_accuracy = train_probe(
            train_acts,
            train_labels,
            test_acts,
            test_labels,
            get_acts,
            precomputed_acts=True,
            epochs=epochs,
            dim=d_model,
            device=device,
            model_dtype=model_dtype,
        )

        probes[class_name] = probe
        test_accuracies[class_name] = test_accuracy

    if save_results:
        only_model_name = llm_model_name.split("/")[-1]
        os.makedirs(f"{probe_dir}", exist_ok=True)
        os.makedirs(f"{probe_dir}/{only_model_name}", exist_ok=True)

        with open(probe_output_filename, "wb") as f:
            pickle.dump(probes, f)

    return test_accuracies


if __name__ == "__main__":
    llm_model_name = "EleutherAI/pythia-70m-deduped"
    device = "cuda"
    train_set_size = 1000
    test_set_size = 1000
    context_length = 128
    include_gender = True

    # TODO: I think there may be a scoping issue with model and get_acts(), but we currently aren't using get_acts()
    model = LanguageModel(llm_model_name, device_map=device, dispatch=True)
    probe_dir = "trained_bib_probes"
    only_model_name = llm_model_name.split("/")[-1]

    model_eval_config = utils.ModelEvalConfig.from_full_model_name(llm_model_name)
    probe_layer = model_eval_config.probe_layer

    probe_output_filename = (
        f"{probe_dir}/{only_model_name}/probes_ctx_len_{context_length}_layer_{probe_layer}.pkl"
    )

    test_accuracies = train_probes(
        train_set_size=1000,
        test_set_size=1000,
        model=model,
        context_length=128,
        probe_train_batch_size=50,
        llm_batch_size=20,
        device=device,
        probe_output_filename=probe_output_filename,
        spurious_correlation_removal=False,
        probe_layer=probe_layer,
        dataset_name="bias_in_bios",
        probe_dir=probe_dir,
        llm_model_name=llm_model_name,
        epochs=10,
        seed=SEED,
    )
    print(test_accuracies)
# %%
