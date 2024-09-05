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

# Configuration
DEBUGGING = False
SEED = 42

# Set up paths and model
parent_dir = os.path.abspath("..")
sys.path.append(parent_dir)

tracer_kwargs = dict(scan=DEBUGGING, validate=DEBUGGING)


# Load and prepare dataset
def load_and_prepare_dataset():
    dataset = load_dataset("LabHC/bias_in_bios")
    df = pd.DataFrame(dataset["train"])
    df["combined_label"] = df["profession"].astype(str) + "_" + df["gender"].astype(str)
    return dataset, df


# Profession dictionary
profession_dict = {
    "accountant": 0,
    "architect": 1,
    "attorney": 2,
    "chiropractor": 3,
    "comedian": 4,
    "composer": 5,
    "dentist": 6,
    "dietitian": 7,
    "dj": 8,
    "filmmaker": 9,
    "interior_designer": 10,
    "journalist": 11,
    "model": 12,
    "nurse": 13,
    "painter": 14,
    "paralegal": 15,
    "pastor": 16,
    "personal_trainer": 17,
    "photographer": 18,
    "physician": 19,
    "poet": 20,
    "professor": 21,
    "psychologist": 22,
    "rapper": 23,
    "software_engineer": 24,
    "surgeon": 25,
    "teacher": 26,
    "yoga_teacher": 27,
}
profession_dict_rev = {v: k for k, v in profession_dict.items()}


# Visualization
def plot_label_distribution(df):
    label_counts = df["combined_label"].value_counts().sort_index()
    labels = [
        f"{profession_dict_rev[int(label.split('_')[0])]} ({'Male' if label.split('_')[1] == '0' else 'Female'})"
        for label in label_counts.index
    ]

    plt.figure(figsize=(12, 8))
    plt.bar(labels, label_counts)
    plt.xlabel("(Profession x Gender) Label")
    plt.ylabel("Number of Samples")
    plt.title("Number of Samples per (Profession x Gender) Label")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def add_gender_classes(
    balanced_data: dict, df: pd.DataFrame, cutoff: int, random_seed: int
) -> dict:
    # TODO: Experiment with more professions

    MALE_IDX = 0
    FEMALE_IDX = 1
    professor_idx = profession_dict["professor"]
    nurse_idx = profession_dict["nurse"]

    male_nurse = df[(df["profession"] == nurse_idx) & (df["gender"] == MALE_IDX)][
        "hard_text"
    ].tolist()
    female_nurse = df[(df["profession"] == nurse_idx) & (df["gender"] == FEMALE_IDX)][
        "hard_text"
    ].tolist()

    male_professor = df[(df["profession"] == professor_idx) & (df["gender"] == MALE_IDX)][
        "hard_text"
    ].tolist()
    female_professor = df[(df["profession"] == professor_idx) & (df["gender"] == FEMALE_IDX)][
        "hard_text"
    ].tolist()

    min_count = min(
        len(male_nurse), len(female_nurse), len(male_professor), len(female_professor), cutoff
    )

    assert min_count == cutoff

    rng = np.random.default_rng(random_seed)

    # Create and shuffle combinations
    male_combined = male_professor[:min_count] + male_nurse[:min_count]
    female_combined = female_professor[:min_count] + female_nurse[:min_count]
    professors_combined = male_professor[:min_count] + female_professor[:min_count]
    nurses_combined = male_nurse[:min_count] + female_nurse[:min_count]
    male_professor = male_professor[: min_count * 2]
    female_nurse = female_nurse[: min_count * 2]

    mixed_classes = male_combined + female_combined + professors_combined + nurses_combined
    rng.shuffle(mixed_classes)

    pos_ratio = 1.8
    neg_ratio = 2.0 - pos_ratio

    biased_males_combined = (
        male_professor[: math.ceil(min_count * pos_ratio)]
        + mixed_classes[: math.ceil(min_count * neg_ratio)]
    )
    biased_females_combined = (
        female_professor[: math.ceil(min_count * neg_ratio)]
        + mixed_classes[: math.ceil(min_count * pos_ratio)]
    )

    # Shuffle each combination
    rng.shuffle(male_combined)
    rng.shuffle(female_combined)
    rng.shuffle(professors_combined)
    rng.shuffle(nurses_combined)
    rng.shuffle(male_professor)
    rng.shuffle(female_nurse)

    # Assign to balanced_data
    balanced_data["male / female"] = male_combined
    balanced_data["female_data_only"] = female_combined
    balanced_data["professor / nurse"] = professors_combined
    balanced_data["nurse_data_only"] = nurses_combined
    balanced_data["male_professor / female_nurse"] = male_professor
    balanced_data["female_nurse_data_only"] = female_nurse
    balanced_data["biased_male / biased_female"] = biased_males_combined
    balanced_data["biased_female_data_only"] = biased_females_combined

    return balanced_data


# Dataset balancing and preparation
def get_balanced_dataset(
    dataset,
    min_samples_per_group: int,
    train: bool,
    include_paired_classes: bool,
    random_seed: int = SEED,
):
    df = pd.DataFrame(dataset["train" if train else "test"])
    balanced_df_list = []

    for profession in tqdm(df["profession"].unique()):
        prof_df = df[df["profession"] == profession]
        min_count = prof_df["gender"].value_counts().min()

        if min_count < min_samples_per_group:
            continue

        balanced_prof_df = pd.concat(
            [
                group.sample(n=min_samples_per_group, random_state=random_seed)
                for _, group in prof_df.groupby("gender")
            ]
        ).reset_index(drop=True)
        balanced_df_list.append(balanced_prof_df)

    balanced_df = pd.concat(balanced_df_list).reset_index(drop=True)
    grouped = balanced_df.groupby("profession")["hard_text"].apply(list)

    balanced_data = {label: texts for label, texts in grouped.items()}

    if include_paired_classes:
        balanced_data = add_gender_classes(balanced_data, df, min_samples_per_group, random_seed)

    for key in balanced_data.keys():
        balanced_data[key] = balanced_data[key][: min_samples_per_group * 2]
        assert len(balanced_data[key]) == min_samples_per_group * 2

    return balanced_data


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


def get_train_test_data(
    dataset,
    train_set_size: int,
    test_set_size: int,
    include_paired_classes: bool,
) -> tuple[dict, dict]:
    # 4 is because male / gender for each profession
    minimum_train_samples = train_set_size // 4
    minimum_test_samples = test_set_size // 4

    train_bios = get_balanced_dataset(
        dataset,
        minimum_train_samples,
        train=True,
        include_paired_classes=include_paired_classes,
    )
    test_bios = get_balanced_dataset(
        dataset,
        minimum_test_samples,
        train=False,
        include_paired_classes=include_paired_classes,
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
def get_all_activations(
    text_inputs: list[str], model: LanguageModel, batch_size: int, submodule: utils.submodule_alias
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
    batch_size: int,
    select_top_k: Optional[int] = None,
) -> tuple[list[t.Tensor], list[t.Tensor]]:
    """If class_idx is a string, there is a paired class idx in utils.py."""
    positive_acts_BD = all_activations[class_idx]
    device = positive_acts_BD.device

    num_positive = len(positive_acts_BD)

    if isinstance(class_idx, int):
        # Collect all negative class activations and labels
        negative_acts = []
        for idx, acts in all_activations.items():
            if idx != class_idx and isinstance(idx, int):
                negative_acts.append(acts)

        negative_acts = t.cat(negative_acts)
    else:
        if class_idx not in utils.PAIRED_CLASS_KEYS:
            raise ValueError(f"Class index {class_idx} is not a valid class index.")

        negative_acts = all_activations[utils.PAIRED_CLASS_KEYS[class_idx]]

    # Randomly select num_positive samples from negative class
    indices = t.randperm(len(negative_acts))[:num_positive]
    selected_negative_acts_BD = negative_acts[indices]

    assert selected_negative_acts_BD.shape == positive_acts_BD.shape

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
    lr: float = 1e-2,
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
    probes,
    all_class_list: list[str],
    all_activations: dict[str, t.Tensor],
    probe_batch_size: int,
):
    test_accuracies = {}
    for class_name in all_class_list:
        batch_test_acts, batch_test_labels = prepare_probe_data(
            all_activations, class_name, probe_batch_size
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

    for class_name in all_class_list:
        if class_name not in utils.PAIRED_CLASS_KEYS:
            continue
        spurious_class_names = [key for key in utils.PAIRED_CLASS_KEYS if key != class_name]
        batch_test_acts, batch_test_labels = prepare_probe_data(
            all_activations, class_name, probe_batch_size
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
    probe_batch_size: int,
    llm_batch_size: int,
    device: str,
    probe_output_filename: str,
    probe_dir: str = "trained_bib_probes",
    llm_model_name: str = "EleutherAI/pythia-70m-deduped",
    epochs: int = 10,
    model_dtype: t.dtype = t.bfloat16,
    save_results: bool = True,
    seed: int = SEED,
    include_gender: bool = False,
) -> dict[int, float]:
    """Because we save the probes, we always train them on all classes to avoid potential issues with missing classes. It's only a one-time cost."""
    t.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model_eval_config = utils.ModelEvalConfig.from_full_model_name(llm_model_name)
    d_model = model_eval_config.activation_dim
    probe_layer = model_eval_config.probe_layer
    probe_act_submodule = utils.get_submodule(model, "resid_post", probe_layer)

    dataset, df = load_and_prepare_dataset()

    train_bios, test_bios = get_train_test_data(
        dataset,
        train_set_size,
        test_set_size,
        include_gender,
    )

    train_bios = utils.tokenize_data(train_bios, model.tokenizer, context_length, device)
    test_bios = utils.tokenize_data(test_bios, model.tokenizer, context_length, device)

    probes, test_accuracies = {}, {}

    all_train_acts = {}
    all_test_acts = {}

    with t.no_grad():
        for i, profession in enumerate(train_bios.keys()):
            # if isinstance(profession, int):
            #     continue

            print(f"Collecting activations for profession: {profession}")

            all_train_acts[profession] = get_all_activations(
                train_bios[profession], model, llm_batch_size, probe_act_submodule
            )
            all_test_acts[profession] = get_all_activations(
                test_bios[profession], model, llm_batch_size, probe_act_submodule
            )

    t.set_grad_enabled(True)

    for profession in all_train_acts.keys():
        if profession in utils.PAIRED_CLASS_KEYS.values():
            continue

        train_acts, train_labels = prepare_probe_data(all_train_acts, profession, probe_batch_size)

        test_acts, test_labels = prepare_probe_data(all_test_acts, profession, probe_batch_size)

        if profession == "biased_male / biased_female" or profession == "male / female":
            probe_epochs = 1
        else:
            probe_epochs = epochs

        probe, test_accuracy = train_probe(
            train_acts,
            train_labels,
            test_acts,
            test_labels,
            get_acts,
            precomputed_acts=True,
            epochs=probe_epochs,
            dim=d_model,
            device=device,
            model_dtype=model_dtype,
        )

        probes[profession] = probe
        test_accuracies[profession] = test_accuracy

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
        probe_batch_size=50,
        llm_batch_size=20,
        llm_model_name=llm_model_name,
        epochs=10,
        device=device,
        probe_output_filename=probe_output_filename,
        probe_dir=probe_dir,
        seed=SEED,
        include_gender=include_gender,
    )
    print(test_accuracies)
# %%
