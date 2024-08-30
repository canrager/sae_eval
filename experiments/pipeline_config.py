from dataclasses import dataclass
import torch


@dataclass
class PipelineConfig:
    max_activations_collection_n_inputs: int = 5000
    top_k_activating_inputs: int = 10

    probe_train_set_size: int = 4000
    probe_test_set_size: int = 500

    # Load datset and probes
    train_set_size: int = 100
    test_set_size: int = 200

    eval_saes_n_inputs: int = 250

    probe_batch_size: int = min(500, probe_test_set_size)
    probe_epochs: int = 10

    model_dtype: torch.dtype = torch.bfloat16

    reduced_GPU_memory: bool = False
    include_gender: bool = True

    dictionaries_path: str = "../dictionary_learning/dictionaries"
    probes_dir: str = "trained_bib_probes"
