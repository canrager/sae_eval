#%%
# Imports

from datasets import load_dataset
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

# tracer_kwargs = dict(scan=DEBUGGING, validate=DEBUGGING)
# model = LanguageModel('EleutherAI/pythia-70m-deduped', device_map=DEVICE, dispatch=True)




# Load and prepare dataset
def load_and_prepare_dataset(ds_name):
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
    plt.xlabel('Label')
    plt.ylabel('Number of Samples')
    plt.title('Number of Samples per Label')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


#%%
ds_names = [
    "LabHC/bias_in_bios",
]

dataset, df = load_and_prepare_dataset(ds_names[0])
plot_label_distribution(df)
# %%
# Amazon

from datasets import load_dataset

amazon_categories ={
    0: "All_Beauty",
    1: "Amazon_Fashion",
    2: "Appliances",
    3: "Arts_Crafts_and_Sewing",
    4: "Automotive",
    5: "Baby_Products",
    6: "Beauty_and_Personal_Care",
    7: "Books",
    8: "CDs_and_Vinyl",
    9: "Cell_Phones_and_Accessories",
    10: "Clothing_Shoes_and_Jewelry",
    11: "Digital_Music",
    12: "Electronics",
    13: "Gift_Cards",
    14: "Grocery_and_Gourmet_Food",
    15: "Handmade_Products",
    16: "Health_and_Household",
    17: "Health_and_Personal_Care",
    18: "Home_and_Kitchen",
    19: "Industrial_and_Scientific",
    20: "Kindle_Store",
    21: "Magazine_Subscriptions",
    22: "Movies_and_TV",
    23: "Musical_Instruments",
    24: "Office_Products",
    25: "Patio_Lawn_and_Garden",
    26: "Pet_Supplies",
    27: "Software",
    28: "Sports_and_Outdoors",
    29: "Subscription_Boxes",
    30: "Tools_and_Home_Improvement",
    31: "Toys_and_Games",
    32: "Video_Games",
    33: "Unknown"
}

n_samples = 1000
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{amazon_categories[2]}", trust_remote_code=True, split=f"full[:{n_samples}]")
len(dataset)

# %%

# This is not a multiclass classification but rather question answering
from datasets import load_dataset

dataset = load_dataset("sciq")

# Check the features of the dataset
print(dataset['train'].features)

# Look at a few examples to see if there are any domain labels
for i in range(5):
    print(dataset['train'][i])


#%%

from datasets import load_dataset, load_from_disk

dataset_group_dir = "./sp_datasets/"
# get all folder names in dataset dir
dataset_group_names = [d for d in os.listdir(dataset_group_dir) if os.path.isdir(os.path.join(dataset_group_dir, d))]
dataset_dir = os.path.join(dataset_group_dir, dataset_group_names[0])
dataset = load_from_disk(dataset_dir)

# %%
# Assuming your dataset is loaded and named 'dataset'

# Use the .take() method to get the first sample
first_sample = dataset.take(1)

# Print the first sample
print("First sample:")
for key, value in first_sample[0].items():
    print(f"{key}: {value}")

# If you want to access a specific field
print("\nTokens of the first sample:")
print(first_sample[0]['tokens'])

# To see multiple samples, you can do:
print("\nFirst 5 samples:")
for i, sample in enumerate(dataset.take(5)):
    print(f"\nSample {i + 1}:")
    for key, value in sample.items():
        print(f"{key}: {value}")

# To see the data types
print("\nData types:")
print(dataset.features)

# %%

from datasets import load_dataset, load_from_disk
import os

dataset_group_dir = "./sp_datasets/"
# get all folder names in dataset dir
dataset_group_names = [d for d in os.listdir(dataset_group_dir) if os.path.isdir(os.path.join(dataset_group_dir, d))]
dataset_dir = os.path.join(dataset_group_dir, dataset_group_names[0])
dataset = load_from_disk(dataset_dir)

# Function to safely print a sample
def print_sample(sample):
    print("Sample:")
    for key, value in sample.items():
        print(f"{key}: {value}")

# Print the first sample
print("First sample:")
first_sample = next(iter(dataset))
print_sample(first_sample)

# Print the first 5 samples
print("\nFirst 5 samples:")
for i, sample in enumerate(dataset.iter(batch_size=1)):
    if i >= 5:
        break
    print(f"\nSample {i + 1}:")
    print_sample(sample)

# Print data types
print("\nData types:")
print(dataset.features)
# %%
