# %%
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gc
from collections import defaultdict
parent_dir = os.path.abspath('..')
sys.path.append(parent_dir)

from datasets import load_dataset
import random
from nnsight import LanguageModel
import torch as t
from torch import nn
from attribution import patching_effect
from dictionary_learning import AutoEncoder, ActivationBuffer
from dictionary_learning.dictionary import IdentityDict
from dictionary_learning.interp import examine_dimension
from dictionary_learning.utils import hf_dataset_to_generator

from experiments.bib_multiclass import (
    load_and_prepare_dataset, 
    get_train_test_data, 
    get_class_nonclass_samples, 
    test_probe,
    prepare_probe_data,
    get_all_activations,
    Probe
)

DEBUGGING = False

if DEBUGGING:
    tracer_kwargs = dict(scan=True, validate=True)
else:
    tracer_kwargs = dict(scan=False, validate=False)





# loading dictionaries
def load_submodules_and_dictionaries(model, probe_layer, activation_dim, load_embed=True, load_attn=True, load_mlp=True, load_resid=True, DEVICE='cpu'):
    dict_id = 10
    expansion_factor = 64
    dictionary_size = expansion_factor * activation_dim

    submodules = []
    dictionaries = {}

    if load_embed:
        submodules.append(model.gpt_neox.embed_in)
        dictionaries[model.gpt_neox.embed_in] = AutoEncoder.from_pretrained(
            f'../dictionary_learning/dictionaries/pythia-70m-deduped/embed/{dict_id}_{dictionary_size}/ae.pt',
            device=DEVICE
        )
    for i in range(probe_layer + 1):
        if load_attn:
            submodules.append(model.gpt_neox.layers[i].attention)
            dictionaries[model.gpt_neox.layers[i].attention] = AutoEncoder.from_pretrained(
                f'../dictionary_learning/dictionaries/pythia-70m-deduped/attn_out_layer{i}/{dict_id}_{dictionary_size}/ae.pt',
                device=DEVICE
            )

        if load_mlp:
            submodules.append(model.gpt_neox.layers[i].mlp)
            dictionaries[model.gpt_neox.layers[i].mlp] = AutoEncoder.from_pretrained(
                f'../dictionary_learning/dictionaries/pythia-70m-deduped/mlp_out_layer{i}/{dict_id}_{dictionary_size}/ae.pt',
                device=DEVICE
            )

        if load_resid:
            submodules.append(model.gpt_neox.layers[i])
            dictionaries[model.gpt_neox.layers[i]] = AutoEncoder.from_pretrained(
                f'../dictionary_learning/dictionaries/pythia-70m-deduped/resid_out_layer{i}/{dict_id}_{dictionary_size}/ae.pt',
                device=DEVICE
            )
    return submodules, dictionaries



# Metric function effectively maximizing the logit difference between the classes: selected, and nonclass

# labels[0] = true labels (profession)
# labels[1] = spurious labels (gender)

def metric_fn(model, labels=None, probe=None):
    attn_mask = model.input[1]['attention_mask']
    acts = model.gpt_neox.layers[layer].output[0]
    acts = acts * attn_mask[:, :, None]
    acts = acts.sum(1) / attn_mask.sum(1)[:, None]
    
    return t.where(
        labels == 0,
        probe(acts),
        - probe(acts)
    )

# Attribution Patching

def get_effects_per_class(model, submodules, dictionaries, probes, class_idx, train_bios, n_batches=None, batch_size=10):
    probe = probes[class_idx]
    texts_train, labels_train = get_class_nonclass_samples(train_bios, class_idx, batch_size)
    if n_batches is not None:
        if len(texts_train) > n_batches:
            texts_train = texts_train[:n_batches]
            labels_train = labels_train[:n_batches]

    running_total = 0
    running_nodes = None    

    for batch_idx, (clean, labels) in tqdm(enumerate(zip(texts_train, labels_train)), total=n_batches):
        if batch_idx == n_batches:
            break

        effects, _, _, _ = patching_effect(
            clean,
            None,
            model,
            submodules,
            dictionaries,
            metric_fn,
            metric_kwargs=dict(labels=labels, probe=probe),
            method='ig'
        )
        with t.no_grad():
            if running_nodes is None:
                running_nodes = {k : len(clean) * v.sum(dim=1).mean(dim=0) for k, v in effects.items()}
            else:
                for k, v in effects.items():
                    running_nodes[k] += len(clean) * v.sum(dim=1).mean(dim=0)
            running_total += len(clean)
        del effects, _
        gc.collect()

    nodes = {k : v / running_total for k, v in running_nodes.items()}
    # Convert SparseAct to Tensor
    nodes = {k : v.act for k, v in nodes.items()}
    return nodes


# Get the output activations for the submodule where some saes are ablated
def get_acts_ablated(
    text,
    model,
    submodules,
    dictionaries,
    to_ablate
):
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
            f[...,feat_idxs] = 0. # zero ablation
            if is_tuple[submodule]:
                submodule.output[0][:] = dictionary.decode(f) + res
            else:
                submodule.output = dictionary.decode(f) + res
        attn_mask = model.input[1]['attention_mask']
        act = model.gpt_neox.layers[layer].output[0]
        act = act * attn_mask[:, :, None]
        act = act.sum(1) / attn_mask.sum(1)[:, None]
        act = act.save()

    t.cuda.empty_cache()
    gc.collect()

    return act.value


# Get the output activations for the submodule where some saes are ablated
def get_all_acts_ablated(
    text_batches: list[list[str]],
    model,
    submodules,
    dictionaries,
    to_ablate
):
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
                f[...,feat_idxs] = 0. # zero ablation
                if is_tuple[submodule]:
                    submodule.output[0][:] = dictionary.decode(f) + res
                else:
                    submodule.output = dictionary.decode(f) + res
            attn_mask = model.input[1]['attention_mask']
            act = model.gpt_neox.layers[layer].output[0]
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


def select_significant_features(submodules, nodes, activation_dim, T_effect=0.05): 
    top_feats_to_ablate = {}
    total_features = 0
    for component_idx, effect in enumerate(nodes.values()):
        print(f"Component {component_idx}:")
        top_feats_to_ablate[submodules[component_idx]] = []
        for idx in (effect > T_effect).nonzero():
            print(idx.item(), effect[idx].item())
            top_feats_to_ablate[submodules[component_idx]].append(idx.item())
            total_features += 1
    print(f"total features: {total_features}")

    top_feats_to_ablate = {
        submodule : n_hot(feats, activation_dim) for submodule, feats in top_feats_to_ablate.items()
    }
    return top_feats_to_ablate

## Plotting functions

def plot_feature_effects_above_threshold(nodes, threshold=0.05):
        all_values = []
        for key in nodes.keys():
            all_values.append(nodes[key].cpu().numpy().reshape(-1))
        all_values = [x for sublist in all_values for x in sublist]
        all_values = [x for x in all_values if x > threshold]

        all_values = sorted(all_values, reverse=True)
        plt.scatter(range(len(all_values)), all_values)
        plt.title('all_values')
        plt.show()

def plot_accuracy_comparison(test_accuracies):
    # Get unique probe_idx values
    probe_indices = test_accuracies[-1].keys()

    # Get ablated_classes values
    ablated_class_indices = list(test_accuracies.keys())

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set the width of each bar and the positions of the bars
    width = 1/(len(ablated_class_indices)+1)
    x = np.arange(len(probe_indices))

    labels = ['Original Model'] + [f'Ablated class {class_idx}' for class_idx in ablated_class_indices[1:]]

    # Create bars for each class
    for i, class_idx in enumerate(ablated_class_indices):
        values = [test_accuracies[class_idx].get(idx, 0)[0] for idx in probe_indices]
        print(values)
        print(probe_indices)
        ax.bar(x + i*width, values, width, label=labels[i])

    # Customize the plot
    ax.set_xlabel('Probe Index')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Bar Plot of Values by Probe Index and Class')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(probe_indices)
    ax.legend(loc='lower right')

    # Add some padding to the x-axis
    plt.xlim(-width, len(probe_indices) - width/2)

    # Show the plot
    plt.tight_layout()
    plt.show()





#%%
# Load model and dictionaries
DEVICE = 'cuda:0'
layer = 4 # model layer for attaching linear classification head
SEED = 42
model = LanguageModel('EleutherAI/pythia-70m-deduped', device_map=DEVICE, dispatch=True)
activation_dim = 512

submodules, dictionaries = load_submodules_and_dictionaries(model, layer, activation_dim, DEVICE=DEVICE)




#%%
# Load datset and probes
train_set_size = 500
test_set_size = 100
batch_size_act_cache = 500

dataset, _ = load_and_prepare_dataset()
train_bios, test_bios = get_train_test_data(dataset, train_set_size, test_set_size)
test_accuracies = {} # class ablated, class probe accuracy

probes = t.load('trained_bib_probes/probes_0705.pt')
all_classes_list = list(probes.keys())[:5]


### Get activations for original model, all classes
test_acts = {}
for class_idx in all_classes_list:
    class_test_acts = get_all_activations(test_bios[class_idx], model)
    test_acts[class_idx] = class_test_acts

test_accuracies[-1] = defaultdict(list)
for class_idx in all_classes_list:
    batch_test_acts, batch_test_labels = prepare_probe_data(test_acts, class_idx, batch_size_act_cache)
    test_acc_probe = test_probe(batch_test_acts, batch_test_labels, probes[class_idx], precomputed_acts=True)
    print(f'class {class_idx} test accuracy: {test_acc_probe}')
    test_accuracies[-1][class_idx].append(test_acc_probe)

### Get activations for ablated models
# ablating the top features for each class

N_EVAL_BATCHES = 3
T_effect = 0.05
batch_size_patching = 10

for ablated_class_idx in all_classes_list:
    test_accuracies[ablated_class_idx] = defaultdict(list)

    nodes = get_effects_per_class(
        model,
        submodules,
        dictionaries,
        probes, 
        ablated_class_idx, 
        train_bios, 
        N_EVAL_BATCHES, 
        batch_size=batch_size_patching
    )

    plot_feature_effects_above_threshold(nodes, threshold=T_effect)

    top_feats_to_ablate = select_significant_features(submodules, nodes, activation_dim*64, T_effect=T_effect)
    del nodes
    t.cuda.empty_cache()
    gc.collect()

    test_acts_ablated = {}
    for class_idx in all_classes_list:
        test_acts_ablated[class_idx] = get_all_acts_ablated(
            test_bios[class_idx],
            model,
            submodules,
            dictionaries,
            top_feats_to_ablate,
        )

    for class_idx in all_classes_list:
        batch_test_acts, batch_test_labels = prepare_probe_data(test_acts_ablated, class_idx, batch_size_act_cache)
        test_acc_probe = test_probe(batch_test_acts, batch_test_labels, probes[class_idx], precomputed_acts=True)
        print(f'class {class_idx} test accuracy: {test_acc_probe}')
        test_accuracies[ablated_class_idx][class_idx].append(test_acc_probe)


plot_accuracy_comparison(test_accuracies)

