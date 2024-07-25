# SAE eval
by measuring the effect of SAE features on probes.

`setup.sh` installs dependencies and downloads our pretrained dictionaries.
Most other functionality we added lives in `/experiments`

1. Train linear probes on the professions from the **bias in bios datset**. Saving probes in `/experiments/trained_bib_probes`
2. Identify "relevance scores" for each feature to each probe by running `/experiments/bib_interventions.py`. Saving these scores as a dict `effects[probe_name] = importance_scores_tensor of shape d_SAE` in the file `/dictionary_learning/dictionaries/path_to_the_evaluated_sae/node_effects.pkl`.
3. We're now measuring the effect of zero ablating features with an effect above `threshold` on all probes (not just the one we computed the effect for). Saving the probe accuraries as a dict `accs[ablated_probe][importance_score_threshold][evaluated_probe] = test_accuracy of probe after ablating features above importance threshold for ablated_probe` in the file `/dictionary_learning/dictionaries/path_to_the_evaluated_sae/class_accuracies.pkl`. This is done in the same `/experiments/bib_interventions.py` execution.
4. Run benchmark SAE metrics (L_0, Loss Recovered) in `/experiments/eval_saes.py`. Saving results in `/dictionary_learning/dictionaries/path_to_the_evaluated_sae/eval_results.json`
5. Plotting results in `/experiments/graph_3var.ipynb`.

All experiments follow this convention for declaring the SAEs to evaluate. Example:
```
ae_sweep_paths = {
    "pythia70m_sweep_standard_ctx128_0712": {
        "resid_post_layer_3": {"trainer_ids": [1, 7, 11, 18]}
    }
}
```

More comprehensively:
```
# Example of sweeping over all SAEs in a sweep
ae_sweep_paths = {"pythia70m_test_sae": None}

# Example of sweeping over all SAEs in a submodule
ae_sweep_paths = {"pythia70m_test_sae": {"resid_post_layer_3": {"trainer_ids": None}}}

# Example of sweeping over a single SAE
ae_sweep_paths = {"pythia70m_test_sae": {"resid_post_layer_3": {"trainer_ids": [0]}}}
```




# Fork of: Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models

This repository contains code, data, and links to autoencoders for replicating the experiments of [Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models](https://arxiv.org/abs/2403.19647). 

## Demos and Links
- We provide an interface for exploring and downloading clusters [here](https://feature-circuits.xyz).
- View our SAEs and their features on [Neuronpedia](https://www.neuronpedia.org/p70d-sm).

## Installation

Use python >= 3.10. To install dependencies, use
```
pip install -r requirements.txt
```

You will also need to clone the [dictionary learning repository](https://github.com/saprmarks/dictionary_learning). Run this command from the root directory of this repository to get that code:
```
git submodule update --init
```


## Data
### Subject–Verb Agreement
We create modified versions of the stimuli from [Finlayson et al. (2021)](https://aclanthology.org/2021.acl-long.144/) (code [here](https://github.com/mattf1n/lm-intervention)). Specifically, we use the same nouns and structures, but modify the verb sets to only include those whose singular and plural inflections are single tokens in Pythia. Our data may be found in `data/`.

We use different splits for discovering circuits and evaluating faithfulness/completeness. The `*_train` files are those we use to discover circuits (typically with 100-example subsamples); the `*_test` files are those we use for evaluation.

### Bias in Bios
We download the Bias in Bios dataset from [Huggingface](https://huggingface.co/datasets/LabHC/bias_in_bios). We write functions to subsample the data for our classifier experiments; see [the Bias in Bios experiment notebook](experiments/bib_shift.ipynb).

### Cluster Data
We provide an online interface for observing and downloading clusters [here](https://feature-circuits.xyz).

### Autoencoders
To run our experiments, you will need to either train or download sparse autoencoders for each layer of Pythia 70M. You can download dictionaries using the script provided at our [dictionary learning repository](https://github.com/saprmarks/dictionary_learning). Running that script from the `feature-circuits` home directory should download the dictionaries to `dictionaries/pythia-70m-deduped/`, which is where this repo expects to find them.

### Annotations
We provide feature annotations in `annotations/10_32768.jsonl`. These are primarily used in `circuit_plotting.py`.

## Experiments
Here, we provide instructions for replicating the results from our paper.

### Subject–Verb Agreement
To discover a circuit, use the following command:
```
scripts/get_circuit.sh <data_type> <node_threshold> <edge_threshold> <aggregation> <example_length> <dict_id>
```
For example, to discover a sparse feature circuit for agreement across a relative clause using node threshold 0.1 and edge threshold 0.01, and with no aggregation across token positions, run this command:
```
scripts/get_circuit.sh rc_train 0.1 0.01 none 6 10
```
If you would like a circuit composed of model components instead of sparse features, replace "10" with "id".

By default, this will save a circuit in `circuits/`, and a circuit plot in `circuits/figures/`.

To evaluate the **faithfulness** and **completeness** of circuits across a variety of thresholds, see [experiments/faithfulness.ipynb](experiments/faithfulness.ipynb). To evaluate just a single circuit, use the following command:
```
scripts/evaluate_circuit.sh <circuit_path> <data_type> <node_threshold> <example_length> <dict_id>
```
For example, to evaluate the faithfulness and completeness if the agreement across RC circuit with node threshold 0.1, you can run
```
scripts/evaluate_circuit.sh circuits/rc_train_dict10_node0.1_edge0.01_n100_aggnone.pt rc_test 0.1 6 10
```

### Bias in Bios
All code for replicating our data processing, classifier training, and SHIFT method (including all baselines and skylines) can be found in [experiments/bib_shift.ipynb](experiments/bib_shift.ipynb).

To generate a circuit for the BiB classifier, use [experiments/bib_circuit.ipynb](experiments/bib_circuit.ipynb)

### Clusters
After downloading a cluster, run this script:
```
scripts/get_circuit_nopair.sh <data_path> <node_threshold> <edge_threshold> <dict_id>
```
`data_path` should be the full path to a cluster .json in the same format as those that can be downloaded [here](https://feature-circuits.xyz). By default, this will save a circuit in `circuits/` and a circuit plot in `circuits/figures/`.


## General utilties
The following files contain utilities which are generally useful for our circuit discovery methods:
* `attribution.py` implements methods for attributing model behaviors to SAE features and error terms.
* `activation_utils.py` defines the `SparseAct` object, which bundles together feature activations and error terms in a convenient way and provides utilities for working with them in a unified way.
* `ablation.py` implements general methods useful for performing SAE feature ablations
* `circuit.py` contains our circuit discovery code
* `circuit_plotting.py` contains our code for plotting circuits, once discovered.
* `loading_utils.py` contains utilities for working with our subject-verb agreement datasets and clusters.

## Citation
If you use any of the code or ideas presented here, please cite our paper:
```
@article{marks-etal-2024-feature,
    author={Samuel Marks and Can Rager and Eric J. Michaud and Yonatan Belinkov and David Bau and Aaron Mueller},
    title={Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models},
    year={2024},
    journal={Computing Research Repository},
    volume={arXiv:2403.19647},
    url={https://arxiv.org/abs/2403.19647}
}
```


## License
We release source code for this work under an MIT license.
