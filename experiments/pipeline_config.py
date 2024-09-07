from dataclasses import dataclass, asdict
from typing import Optional
from enum import Enum
import torch


class FeatureSelection(Enum):
    unique = 1
    above_threshold = 2
    top_n = 3


@dataclass
class PipelineConfig:
    dataset_name: str = "bias_in_bios"
    column1_vals = ("professor", "nurse")
    column2_vals = ("male", "female")

    max_activations_collection_n_inputs: int = 10000
    top_k_inputs_act_collect: int = 5

    probe_train_set_size: int = 4000
    probe_test_set_size: int = 500

    # Load datset and probes
    train_set_size: int = 100
    test_set_size: int = 200

    eval_saes_n_inputs: int = 250

    probe_batch_size: int = min(500, test_set_size)
    probe_epochs: int = 10

    model_dtype: torch.dtype = torch.bfloat16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reduced_GPU_memory: bool = False
    include_gender: bool = True

    force_eval_results_recompute: bool = False
    force_max_activations_recompute: bool = False
    force_probe_recompute: bool = False
    force_node_effects_recompute: bool = False
    force_autointerp_recompute: bool = False
    force_ablations_recompute: bool = False

    dictionaries_path: str = "../dictionary_learning/dictionaries"
    probes_dir: str = "trained_bib_probes"

    # If == "sae_layer", then the probe layer is the same as the SAE layer
    # else it should be an int, representing the layer number
    # This only applies to Gemma-2-2b
    gemma_probe_layer: str | int = "sae_layer"

    attribution_patching_method: str = "attrib"
    ig_steps: int = 10

    # selection_method = FeatureSelection.above_threshold
    selection_method = FeatureSelection.top_n

    attrib_t_effects = [2, 5, 10, 20, 50, 100, 500, 1000, 2000]
    autointerp_t_effects = [2, 5, 10, 20]

    # This is for spurrious correlation removal
    chosen_class_indices = [
        "male / female",
        "professor / nurse",
        "male_professor / female_nurse",
        "biased_male / biased_female",
    ]

    # This is for targeted probe perturbation
    # chosen_class_indices = [
    #     0,
    #     1,
    #     2,
    #     6,
    # ]

    # Autointerp stuff

    use_autointerp: bool = True

    api_llm: str = "claude-3-5-sonnet-20240620"
    # api_llm: str = "claude-3-haiku-20240307"

    autointerp_api_total_token_per_minute_limit: int = 400_000
    autointerp_api_total_requests_per_minute_limit: int = 4_000
    num_allowed_tokens_per_minute: int = int(0.3 * autointerp_api_total_token_per_minute_limit)
    num_allowed_requests_per_minute: int = int(0.3 * autointerp_api_total_requests_per_minute_limit)
    num_tokens_system_prompt: Optional[int] = None  # Will be set during llm_query

    prompt_dir: str = "llm_autointerp/"
    node_effects_attrib_filename: str = "node_effects.pkl"
    autointerp_filename: str = "node_effects_auto_interp.pkl"
    bias_shift_dir1_filename: str = "node_effects_bias_shift_dir1.pkl"
    bias_shift_dir2_filename: str = "node_effects_bias_shift_dir2.pkl"

    saving_exclude_files = ["max_activating_inputs.pkl", "ae.pt"]

    # This will be shared if you have multiple configs open, so don't do that. Don't modify it while running either.
    node_effect_filenames = [
        node_effects_attrib_filename,
        autointerp_filename,
        bias_shift_dir1_filename,
        bias_shift_dir2_filename,
    ]

    autointerp_context_length: int = 128
    num_top_emphasized_tokens: int = 5
    num_top_inputs_autointerp: int = top_k_inputs_act_collect
    num_top_features_per_class: int = 20
    # num_top_features_per_class: int = 1

    llm_judge_min_scale: int = 0
    llm_judge_max_scale: int = 4
    llm_judge_binary_threshold: float = 0.5

    include_activation_values_in_prompt: bool = True

    chosen_autointerp_class_names = None

    def to_dict(self):
        return {k: str(v) if isinstance(v, torch.dtype) else v for k, v in asdict(self).items()}
