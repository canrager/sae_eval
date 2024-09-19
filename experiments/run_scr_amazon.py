from experiments.bib_intervention import *
from experiments.dataset_info import *
from experiments.pipeline_config import PipelineConfig
from experiments import utils
import time
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


p_config = PipelineConfig()
p_config.spurious_corr = True
p_config.include_gender = True
p_config.probe_layer = "sae_layer"

column1_names = ("Software", "Beauty_and_Personal_Care")
column2_names = (1.0, 5.0)
p_config.column2_name = "rating"

p_config.column1_vals = column1_names
p_config.column2_vals = column2_names

p_config.sweep_output_dir = "09_18_gemma2-2b_scr_amazon_topk_standard_jump_autointerp"
p_config.probes_dir = "trained_amazon_probes"

p_config.api_llm = "gpt-4o-mini-2024-07-18"
p_config.max_percentage_of_num_allowed_tokens_per_minute = 0.4
p_config.max_percentage_of_num_allowed_requests_per_minute = 0.4
p_config.autointerp_t_effects = [2, 5, 10, 20, 50]
p_config.num_top_features_per_class = 50

p_config.force_ablations_recompute = True
p_config.force_node_effects_recompute = True
p_config.force_autointerp_recompute = True
p_config.use_autointerp = True

p_config.dataset_name = "amazon_reviews_1and5"

# chosen_classes = [
#     "Beauty_and_Personal_Care",
#     "Books",
#     "Automotive",
#     "Musical_Instruments",
#     "Software",
# ]
# p_config.chosen_class_indices = [
#     amazon_category_dict[i] for i in chosen_classes
# ]
p_config.chosen_class_indices = [
    "male / female",
    "professor / nurse",
    "male_professor / female_nurse",
]

trainer_ids = [0, 2, 4]

ae_sweep_paths = {
    "gemma-2-2b_sweep_topk_ctx128_ef8_0824": {
        "resid_post_layer_19": {"trainer_ids": trainer_ids},
    },
    "gemma-2-2b_sweep_standard_ctx128_ef8_0824": {
        "resid_post_layer_19": {"trainer_ids": trainer_ids},
    },
    "gemma-2-2b_sweep_jumprelu_0902": {
        "resid_post_layer_19": {"trainer_ids": trainer_ids},
    },
}

column1_vals_list = [
    ("Books", "CDs_and_Vinyl"),
    ("Software", "Electronics"),
    ("Pet Supplies", "Office_Products"),
    ("Industrial_and_Scientific", "Toys_and_Games"),
]

# column1_vals_list = [
#     (24, 26),
#     (30, 6),
#     (18, 11),
#     (3, 1),
# ]

# Set manually in console
if p_config.use_autointerp:
    llm_utils.set_api_key(p_config.api_llm, "../")

# This will look for any empty folders in any ae_path and raise an error if it finds any
for sweep_name, submodule_trainers in ae_sweep_paths.items():
    ae_group_paths = utils.get_ae_group_paths(
        p_config.dictionaries_path, sweep_name, submodule_trainers
    )

for column1_vals in column1_vals_list:
    p_config.column1_vals = column1_vals

    start_time = time.time()

    for sweep_name, submodule_trainers in ae_sweep_paths.items():
        run_interventions(
            submodule_trainers,
            p_config,
            sweep_name,
            random_seed=p_config.random_seed,
            verbose=True,
        )

    end_time = time.time()

    print(f"Time taken: {end_time - start_time} seconds")
