# %%
# Imports
import sys
import os
import random
import gc
from collections import defaultdict
import einops
import math

import torch as t
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tqdm import tqdm
from typing import Callable

from datasets import load_dataset
from nnsight import LanguageModel

# Configuration
DEBUGGING = False
DEVICE = "cuda:0"
SEED = 42
BATCH_SIZE = 128
ACTIVATION_DIM = 512
LAYER = 4
MIN_SAMPLES_PER_GROUP = 1024

# Set up paths and model
parent_dir = os.path.abspath("..")
sys.path.append(parent_dir)

tracer_kwargs = dict(scan=DEBUGGING, validate=DEBUGGING)
model = LanguageModel("EleutherAI/pythia-70m-deduped", device_map=DEVICE, dispatch=True)


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


# Dataset balancing and preparation
def get_balanced_dataset(dataset, min_samples_per_group: int, train: bool, random_seed: int = SEED):
    df = pd.DataFrame(dataset["train" if train else "test"])
    balanced_df_list = []

    for profession in df["profession"].unique():
        prof_df = df[df["profession"] == profession]
        min_count = prof_df["gender"].value_counts().min()

        if min_samples_per_group and min_count < min_samples_per_group:
            continue

        cutoff = min_samples_per_group or min_count
        balanced_prof_df = pd.concat(
            [
                group.sample(n=cutoff, random_state=random_seed)
                for _, group in prof_df.groupby("gender")
            ]
        ).reset_index(drop=True)
        balanced_df_list.append(balanced_prof_df)

    balanced_df = pd.concat(balanced_df_list).reset_index(drop=True)
    grouped = balanced_df.groupby("profession")["hard_text"].apply(list)
    return {label: shuffle(texts) for label, texts in grouped.items()}


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


def batch_list(input_list, batch_size):
    return [input_list[i : i + batch_size] for i in range(0, len(input_list), batch_size)]


def sample_from_classes(data_dict, chosen_class):
    total_samples = len(data_dict[chosen_class])
    all_classes = list(data_dict.keys())
    all_classes.remove(chosen_class)
    random_class_indices = random.choices(all_classes, k=total_samples)

    samples_count = defaultdict(int)
    for class_idx in random_class_indices:
        samples_count[class_idx] += 1

    sampled_data = []
    for class_idx, count in samples_count.items():
        sampled_data.extend(random.sample(data_dict[class_idx], count))

    return sampled_data


def create_labeled_dataset(data_dict, chosen_class, batch_size):
    in_class_data = data_dict[chosen_class]
    other_class_data = sample_from_classes(data_dict, chosen_class)

    combined_dataset = [(sample, 0) for sample in in_class_data] + [
        (sample, 1) for sample in other_class_data
    ]
    random.shuffle(combined_dataset)

    bio_texts, bio_labels = zip(*combined_dataset)
    text_batches = [
        bio_texts[i : i + batch_size] for i in range(0, len(combined_dataset), batch_size)
    ]
    label_batches = [
        t.tensor(bio_labels[i : i + batch_size], device=DEVICE)
        for i in range(0, len(combined_dataset), batch_size)
    ]

    return text_batches, label_batches


# Probe model and training
class Probe(nn.Module):
    def __init__(self, activation_dim):
        super().__init__()
        self.net = nn.Linear(activation_dim, 1, bias=True)

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
def get_all_activations(text_batches: list[list[str]]) -> t.Tensor:
    all_acts_list_BD = []
    for text_batch_BL in tqdm(text_batches, desc="Getting activations"):
        with model.trace(text_batch_BL, **tracer_kwargs):
            attn_mask = model.input[1]["attention_mask"]
            acts_BLD = model.gpt_neox.layers[LAYER].output[0]
            acts_BLD = acts_BLD * attn_mask[:, :, None]
            acts_BD = acts_BLD.sum(1) / attn_mask.sum(1)[:, None]
            acts_BD = acts_BD.save()
        all_acts_list_BD.append(acts_BD.value)

    all_acts_bD = t.cat(all_acts_list_BD, dim=0)
    return all_acts_bD


def prepare_probe_data(
    all_activations: dict[int, t.Tensor],
    all_non_class_activations: dict[int, t.Tensor],
    class_idx: int,
    batch_size: int,
) -> tuple[t.Tensor, t.Tensor]:
    positive_acts = all_activations[class_idx]

    num_positive = len(positive_acts)

    # Collect all negative class activations and labels
    negative_acts = []
    for idx, acts in all_non_class_activations.items():
        if idx != class_idx:
            negative_acts.append(acts)

    negative_acts = t.cat(negative_acts)

    assert negative_acts.shape == positive_acts.shape

    # Combine positive and negative samples
    combined_acts = t.cat([positive_acts, negative_acts])
    combined_labels = t.zeros(len(combined_acts), device=DEVICE)
    combined_labels[num_positive:] = 1

    # Shuffle the combined data
    shuffle_indices = t.randperm(len(combined_acts))
    shuffled_acts = combined_acts[shuffle_indices]
    shuffled_labels = combined_labels[shuffle_indices]

    # Reshape into lists of tensors with specified batch_size
    num_samples = len(shuffled_acts)
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division

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
    lr: float = 1e-2,
    epochs: int = 1,
    dim: int = ACTIVATION_DIM,
    seed: int = SEED,
):
    """input_batches can be a list of tensors or strings. If strings, get_acts must be provided."""

    if type(train_input_batches[0]) == str or type(test_input_batches[0]) == str:
        assert precomputed_acts == False
    elif type(train_input_batches[0]) == t.Tensor or type(test_input_batches[0]) == t.Tensor:
        assert precomputed_acts == True

    t.manual_seed(seed)
    probe = Probe(dim).to(DEVICE)
    optimizer = t.optim.AdamW(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    losses = t.zeros(epochs, len(train_input_batches))
    for epoch in range(epochs):
        batch_idx = 0
        for inputs, labels in zip(train_input_batches, train_label_batches):
            if precomputed_acts:
                acts_BD = inputs
            else:
                acts_BD = get_acts(inputs)
            logits_B = probe(acts_BD)
            loss = criterion(logits_B, t.tensor(labels, device=DEVICE, dtype=t.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses[epoch, batch_idx] = loss
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
    return probe, losses


def test_probe(
    input_batches: list,
    label_batches: list[t.Tensor],
    probe: Probe,
    get_acts: Callable,
    precomputed_acts: bool,
):
    with t.no_grad():
        corrects = []
        for input_batch, labels_B in zip(input_batches, label_batches):
            if precomputed_acts:
                acts_BD = input_batch
            else:
                acts_BD = get_acts(input_batch)
            logits_B = probe(acts_BD)
            preds_B = (logits_B > 0.0).long()
            corrects.append((preds_B == labels_B).float())
        return t.cat(corrects).mean().item()


# Main execution
def main():
    dataset, df = load_and_prepare_dataset()
    plot_label_distribution(df)

    train_set_size = 5000
    test_set_size = 1000

    minimum_train_samples = train_set_size // 4
    minimum_test_samples = test_set_size // 4

    train_bios_gender_balanced = get_balanced_dataset(dataset, minimum_train_samples, train=True)
    test_bios_gender_balanced = get_balanced_dataset(dataset, minimum_test_samples, train=False)

    train_bios_gender_balanced, test_bios_gender_balanced = ensure_shared_keys(
        train_bios_gender_balanced, test_bios_gender_balanced
    )

    num_classes = len(train_bios_gender_balanced)
    num_train_non_class_samples = math.ceil(train_set_size / num_classes)
    num_test_non_class_samples = math.ceil(test_set_size / num_classes)

    probes, losses = {}, {}

    all_train_acts = {}
    all_test_acts = {}

    train_non_class_acts = {}
    test_non_class_acts = {}

    for i, profession in enumerate(train_bios_gender_balanced.keys()):
        t.cuda.empty_cache()
        gc.collect()
        print(f"Training probe for profession: {profession}")
        train_input_batches = batch_list(train_bios_gender_balanced[profession], BATCH_SIZE)

        test_input_batches = batch_list(test_bios_gender_balanced[profession], BATCH_SIZE)

        all_train_acts[profession] = get_all_activations(train_input_batches)
        all_test_acts[profession] = get_all_activations(test_input_batches)

        train_non_class_acts[profession] = all_train_acts[profession][:num_train_non_class_samples]
        test_non_class_acts[profession] = all_test_acts[profession][:num_test_non_class_samples]

        # For debugging
        # if i > 1:
        # break

    t.set_grad_enabled(True)

    probe_batch_size = 32

    for profession in all_train_acts.keys():
        train_acts, train_labels = prepare_probe_data(
            all_train_acts, train_non_class_acts, profession, probe_batch_size
        )

        test_acts, test_labels = prepare_probe_data(
            all_test_acts, test_non_class_acts, profession, probe_batch_size
        )

        probe, loss = train_probe(
            train_acts,
            train_labels,
            test_acts,
            test_labels,
            get_acts,
            precomputed_acts=True,
            epochs=10,
        )

        probes[profession] = probe
        losses[profession] = loss

    os.makedirs("trained_bib_probes", exist_ok=True)
    t.save(probes, "trained_bib_probes/probes_0705.pt")
    t.save(losses, "trained_bib_probes/losses_0705.pt")


if __name__ == "__main__":
    main()
# %%
