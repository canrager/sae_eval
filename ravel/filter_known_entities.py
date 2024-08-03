"""
Which entitiy-attribute pairs are known to the model?
Input: RAVEL dataset: entity_to_attribute_data and attribute_to_prompts
Output: entity_to attribute: Dict[Entity -> Dict[Attribute -> Known/Unknown]]

# (Deal with special cases using an LLM judge)
# Squad doesn't work well beyond string matching.

# Measure perplexity of the full completion
# Every E-A_E pair, every prompt.
"""

# %%
import os
import json
import torch as t
from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from torch.utils.data import DataLoader
from typing import List
from tqdm import tqdm

# Model setup
device = "cuda:0"

model = GPTNeoXForCausalLM.from_pretrained(
    "EleutherAI/pythia-70m-deduped",
    revision="step3000",
    cache_dir="./model/pythia-70m-deduped/step3000",
    device_map=device,
)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-70m-deduped",
    revision="step3000",
    cache_dir="./models/pythia-70m-deduped/step3000",
)
model_name = 'pythia-70m-deduped'


# model = AutoModelForCausalLM.from_pretrained("gpt2", device_map=device)
# tokenizer = AutoTokenizer.from_pretrained("gpt2", device_map=device)
# model_name = 'gpt-2-small'

# with open('/share/u/can/src/hf.txt', 'r') as f:
#     hf_token = f.read().strip()

# tokenizer = AutoTokenizer.from_pretrained(
#     "google/gemma-2-2b",
#     cache_dir="./models/gemma-2-2b",
#     use_auth_token=hf_token,
# )

# model = AutoModelForCausalLM.from_pretrained(
#     "google/gemma-2-2b",
#     cache_dir="./models/gemma-2-2b",
#     device_map=device,
#     use_auth_token=hf_token,
# )
# model_name = 'gemma-2-2b'


tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Dataset loading
DATA_DIR = "/share/u/can/sae_eval/ravel/data/"
entity_class = "city"

with open(os.path.join(DATA_DIR, f"base/ravel_{entity_class}_entity_attributes.json")) as f:
    entity_to_attributes = json.load(f)

with open(os.path.join(DATA_DIR, f"base/ravel_{entity_class}_attribute_to_prompts.json")) as f:
    attribute_to_prompts = json.load(f)

entities = list(entity_to_attributes.keys())
attribute_classes = list(attribute_to_prompts.keys())
attribute_classes = [attribute_classes[0]]

n_entities = len(entities)
n_attribute_classes = len(attribute_classes)
n_pairs = n_entities * n_attribute_classes

# Helper function to tokenize and format data
def tokenize_prompts(
    attribute_class: str, entities: List[str]
) -> tuple[t.Tensor, List[str], List[str]]:
    """
    Tokenizes and formats prompts based on a given attribute class and entities.

    Args:
        attribute_class: Attribute category ("Country")
        entities: Entities to generate prompts

    Returns:
        tokenized_prompts: Tokenized prompt formatted with entities, shape (n_entities * n_prompts, max_length)
        correct_attributes: Correct attribute to be predicted, shape (n_entities * n_prompts)
        entity_idxs: entity_to_attributes index of entity used in prompt, shape (n_entities * n_prompts)
    """
    entity_indices = []
    correct_attributes = []
    entity_attribute_prompts = []
    for entity_idx, entity in enumerate(entities):
        attribute = entity_to_attributes[entity][attribute_class]
        for prompt in attribute_to_prompts[attribute_class]:
            formatted_prompt = prompt % entity
            entity_attribute_prompts.append(formatted_prompt)
            entity_indices.append(entity_idx)
            correct_attributes.append(attribute)
    tokenized_prompts = tokenizer(
        entity_attribute_prompts, return_tensors="pt", padding=True, truncation=True
    ).input_ids
    return tokenized_prompts, correct_attributes, entity_indices


# Create Hugging Face dataset
data = []
for attr_idx, attr_class in enumerate(attribute_classes):
    prompts, attributes, entity_indices = tokenize_prompts(attr_class, entities)
    for p, a, entity_idx in zip(prompts, attributes, entity_indices):
        data.append({"prompt": p.tolist(), "attribute": a, "entity_idx": entity_idx, "attribute_idx": attr_idx})

hf_dataset = Dataset.from_list(data)


# Function to convert data for DataLoader
def collate_fn(batch):
    prompt_ids = t.tensor([item["prompt"] for item in batch])
    attribute_strs = [item["attribute"] for item in batch]
    entity_idxs = t.tensor([item["entity_idx"] for item in batch])
    attr_idxs = t.tensor([item["attribute_idx"] for item in batch])
    return prompt_ids, attribute_strs, entity_idxs, attr_idxs


# %%
# from evaluate import load
# squad_metric = load("squad")

# Enable batching
batch_size = 256
n_gen_tokens = 3
dataloader = DataLoader(hf_dataset, batch_size=batch_size, collate_fn=collate_fn)

# Example usage: Loop over batches


correct_attribute_cnt = t.zeros((n_entities, n_attribute_classes), dtype=t.int)

for batch in tqdm(dataloader):
    prompt_ids, attribute_strs, entity_idxs, attr_idxs = batch
    prompt_ids = prompt_ids.to(device)

    # Generate next tokens using Hugging Face's built-in method
    generated_outputs = model.generate(
        prompt_ids,
        max_length=prompt_ids.shape[1] + n_gen_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Extract only the generated tokens
    generated_tokens = generated_outputs[:, prompt_ids.shape[1] :]
    generated_strings = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)

    # Count matches
    matches = t.tensor([a.lower() in p.lower() for p, a in zip(generated_strings, attribute_strs)], dtype=t.int)
    indices = (entity_idxs, attr_idxs)
    correct_attribute_cnt.index_put_(indices, matches, accumulate=True)
    # Alternative: Use SQuAD F1 metric, 
    # predictions = [{'prediction_text': p, 'id': str(i)} for i, p in enumerate(generated_strings)]
    # references = [{'answers': {'answer_start': [0], 'text': [a]}, 'id': str(i)} for i, a in enumerate(attribute_strs)]
    # results = squad_metric.compute(predictions=predictions, references=references)

    # for g, a in zip(generated_strings, attribute_strs):
    #     print(f'Generated: {g.strip()}\t Correct: {a}\n')
    # break
# %%
n_prompts_per_attribute_class = t.tensor([len(attribute_to_prompts[ac]) for ac in attribute_classes])
correct_attribute_cnt = correct_attribute_cnt / n_prompts_per_attribute_class

#%%
known_entity_attribute_dict = {}
for entity_idx, entity in enumerate(entities):
    known_entity_attribute_dict[entity] = {}
    for attr_idx, attr_class in enumerate(attribute_classes):
        known_entity_attribute_dict[entity][attr_class] = correct_attribute_cnt[entity_idx, attr_idx].item()

with open(os.path.join(DATA_DIR, f'{model_name}/known_entity_attribute_pairs.json'), 'w') as f:
    json.dump(known_entity_attribute_dict, f, indent=4)


#%%
# Plot the distribution of correct attribute counts

with open(os.path.join(DATA_DIR, f'{model_name}/known_entity_attribute_pairs.json'), 'r') as f:
    known_entity_attribute_dict = json.load(f)

assert n_entities == len(known_entity_attribute_dict.keys())
assert n_attribute_classes == len(known_entity_attribute_dict[entities[0]].keys())

correct_attribute_cnt = t.zeros((n_entities, n_attribute_classes), dtype=t.float)
for entity_idx, entity in enumerate(entities):
    for attr_idx, attr_class in enumerate(attribute_classes):
        correct_attribute_cnt[entity_idx, attr_idx] = known_entity_attribute_dict[entity][attr_class]

import matplotlib.pyplot as plt
plt.hist(correct_attribute_cnt.cpu().numpy())
plt.xlabel('Fraction of prompts answered correctly per E - A_E pair')
plt.ylabel('Count of E - A_E pairs')
plt.title(f'Knowledge of E - A_E pairs in {model_name}\n Entity class: {entity_class}, #{n_entities} entities, #{n_attribute_classes} attributes')
plt.savefig(os.path.join(DATA_DIR, f'{model_name}/known_entity_attribute_pairs_hist.png'))
# %%
