#%%
# Imports
import sys
import os
import random
import gc
from collections import defaultdict

import torch as t
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tqdm import tqdm

from datasets import load_dataset
from nnsight import LanguageModel

# Configuration
DEBUGGING = False
DEVICE = 'cuda:0'
SEED = 42
BATCH_SIZE = 128
ACTIVATION_DIM = 512
LAYER = 4
MIN_SAMPLES_PER_GROUP = 1024

# Set up paths and model
parent_dir = os.path.abspath('..')
sys.path.append(parent_dir)

tracer_kwargs = dict(scan=DEBUGGING, validate=DEBUGGING)
model = LanguageModel('EleutherAI/pythia-70m-deduped', device_map=DEVICE, dispatch=True)

# Load and prepare dataset
def load_and_prepare_dataset():
    dataset = load_dataset("LabHC/bias_in_bios")
    df = pd.DataFrame(dataset['train'])
    df['combined_label'] = df['profession'].astype(str) + '_' + df['gender'].astype(str)
    return dataset, df

# Profession dictionary
profession_dict = {
    'accountant': 0, 'architect': 1, 'attorney': 2, 'chiropractor': 3, 'comedian': 4,
    'composer': 5, 'dentist': 6, 'dietitian': 7, 'dj': 8, 'filmmaker': 9,
    'interior_designer': 10, 'journalist': 11, 'model': 12, 'nurse': 13,
    'painter': 14, 'paralegal': 15, 'pastor': 16, 'personal_trainer': 17,
    'photographer': 18, 'physician': 19, 'poet': 20, 'professor': 21,
    'psychologist': 22, 'rapper': 23, 'software_engineer': 24, 'surgeon': 25,
    'teacher': 26, 'yoga_teacher': 27
}
profession_dict_rev = {v: k for k, v in profession_dict.items()}

# Visualization
def plot_label_distribution(df):
    label_counts = df['combined_label'].value_counts().sort_index()
    labels = [f"{profession_dict_rev[int(label.split('_')[0])]} ({'Male' if label.split('_')[1] == '0' else 'Female'})"
              for label in label_counts.index]

    plt.figure(figsize=(12, 8))
    plt.bar(labels, label_counts)
    plt.xlabel('(Profession x Gender) Label')
    plt.ylabel('Number of Samples')
    plt.title('Number of Samples per (Profession x Gender) Label')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# Dataset balancing and preparation
def get_balanced_dataset(dataset, min_samples_per_group, train=True):
    df = pd.DataFrame(dataset['train' if train else 'test'])
    balanced_df_list = []

    for profession in df['profession'].unique():
        prof_df = df[df['profession'] == profession]
        min_count = prof_df['gender'].value_counts().min()
        
        if min_samples_per_group and min_count < min_samples_per_group:
            continue
        
        cutoff = min_samples_per_group or min_count
        balanced_prof_df = prof_df.groupby('gender').apply(lambda x: x.sample(n=cutoff)).reset_index(drop=True)
        balanced_df_list.append(balanced_prof_df)

    balanced_df = pd.concat(balanced_df_list).reset_index(drop=True)
    grouped = balanced_df.groupby('profession')['hard_text'].apply(list)
    return {label: shuffle(texts) for label, texts in grouped.items()}

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

    combined_dataset = [(sample, 0) for sample in in_class_data] + [(sample, 1) for sample in other_class_data]
    random.shuffle(combined_dataset)

    bio_texts, bio_labels = zip(*combined_dataset)
    text_batches = [bio_texts[i:i + batch_size] for i in range(0, len(combined_dataset), batch_size)]
    label_batches = [t.tensor(bio_labels[i:i + batch_size], device=DEVICE) for i in range(0, len(combined_dataset), batch_size)]

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
            attn_mask = model.input[1]['attention_mask']
            acts = model.gpt_neox.layers[LAYER].output[0]
            acts = acts * attn_mask[:, :, None]
            acts = acts.sum(1) / attn_mask.sum(1)[:, None]
            acts = acts.save()
        return acts.value

def train_probe(text_batches, label_batches, get_acts, lr=1e-2, epochs=1, dim=ACTIVATION_DIM, seed=SEED):
    t.manual_seed(seed)
    probe = Probe(dim).to(DEVICE)
    optimizer = t.optim.AdamW(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    losses = t.zeros(epochs * len(text_batches))
    batch_idx = 0
    for epoch in range(epochs):
        for text, labels in zip(text_batches, label_batches):
            acts = get_acts(text)
            logits = probe(acts)
            loss = criterion(logits, t.tensor(labels, device=DEVICE, dtype=t.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses[batch_idx] = loss
            batch_idx += 1
    return probe, losses

def test_probe(text_batches, label_batches, probe, get_acts):
    with t.no_grad():
        corrects = []
        for text, labels in zip(text_batches, label_batches):
            acts = get_acts(text)
            logits = probe(acts)
            preds = (logits > 0.0).long()
            corrects.append((preds == labels).float())
        return t.cat(corrects).mean().item()

# Main execution
def main():
    dataset, df = load_and_prepare_dataset()
    plot_label_distribution(df)

    bios_gender_balanced = get_balanced_dataset(dataset, MIN_SAMPLES_PER_GROUP, train=True)
    
    probes, losses = {}, {}
    for profession in bios_gender_balanced.keys():
        t.cuda.empty_cache()
        gc.collect()
        print(f'Training probe for profession: {profession}')
        text_batches, label_batches = create_labeled_dataset(bios_gender_balanced, profession, BATCH_SIZE)
        probe, loss = train_probe(text_batches, label_batches, get_acts, epochs=1)
        probes[profession] = probe
        losses[profession] = loss

    os.makedirs('trained_bib_probes', exist_ok=True)
    t.save(probes, 'trained_bib_probes/probes_0705.pt')
    t.save(losses, 'trained_bib_probes/losses_0705.pt')

    bios_test = get_balanced_dataset(dataset, min_samples_per_group=50, train=False)
    test_accuracies = {}
    for profession, probe in probes.items():
        text_batches, label_batches = create_labeled_dataset(bios_test, profession, BATCH_SIZE)
        accuracy = test_probe(text_batches, label_batches, probe, get_acts)
        print(f'Profession: {profession}, Accuracy: {accuracy}')
        test_accuracies[profession] = accuracy

if __name__ == "__main__":
    main()
# %%
