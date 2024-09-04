from experiments.bib_intervention import *
from experiments.utils_bib_dataset import profession_int_to_str


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

selection_method = FeatureSelection.top_n

random_seed = 42

chosen_class_indices = [
    0,
    1,
    2,
    6,
    9,
    11,
    13,
    14,
    18,
    19,
    20,
    21,
    22,
    25,
    26,
]

pipeline_config = PipelineConfig()
pipeline_config.chosen_autointerp_class_names = [
    profession_int_to_str[i] for i in chosen_class_indices
]

top_n_features = [100]
T_effects_all_classes = [0.1, 0.01]
T_effects_unique_class = [1e-4, 1e-8]

if selection_method == FeatureSelection.top_n:
    T_effects = top_n_features
elif selection_method == FeatureSelection.above_threshold:
    T_effects = T_effects_all_classes
elif selection_method == FeatureSelection.unique:
    T_effects = T_effects_unique_class
else:
    raise ValueError("Invalid selection method")

T_max_sideeffect = 5e-3

trainer_ids = [2, 6, 10, 14, 18]

ae_sweep_paths = {
    "pythia70m_sweep_standard_ctx128_0712": {
        #     # "resid_post_layer_0": {"trainer_ids": None},
        #     # "resid_post_layer_1": {"trainer_ids": None},
        #     # "resid_post_layer_2": {"trainer_ids": None},
        "resid_post_layer_3": {"trainer_ids": trainer_ids},
        #     "resid_post_layer_4": {"trainer_ids": None},
    },
    # "pythia70m_sweep_gated_ctx128_0730": {
    #     # "resid_post_layer_0": {"trainer_ids": None},
    #     # "resid_post_layer_1": {"trainer_ids": None},
    #     # "resid_post_layer_2": {"trainer_ids": None},
    #     "resid_post_layer_3": {"trainer_ids": [2, 6, 10, 18]},
    #     # "resid_post_layer_4": {"trainer_ids": None},
    # },
    # "pythia70m_sweep_panneal_ctx128_0730": {
    #     # "resid_post_layer_0": {"trainer_ids": None},
    #     # "resid_post_layer_1": {"trainer_ids": None},
    #     # "resid_post_layer_2": {"trainer_ids": None},
    #     "resid_post_layer_3": {"trainer_ids": trainer_ids},
    #     # "resid_post_layer_4": {"trainer_ids": None},
    # },
    "pythia70m_sweep_topk_ctx128_0730": {
        # "resid_post_layer_0": {"trainer_ids": None},
        # "resid_post_layer_1": {"trainer_ids": None},
        # "resid_post_layer_2": {"trainer_ids": None},
        "resid_post_layer_3": {"trainer_ids": trainer_ids},
        # "resid_post_layer_4": {"trainer_ids": None},
    },
}

# trainer_ids = [0, 2, 4, 5]

# ae_sweep_paths = {
#     "gemma-2-2b_sweep_topk_ctx128_ef8_0824": {
#         # "resid_post_layer_3": {"trainer_ids": trainer_ids},
#         # "resid_post_layer_7": {"trainer_ids": trainer_ids},
#         "resid_post_layer_11": {"trainer_ids": trainer_ids},
#         # "resid_post_layer_15": {"trainer_ids": trainer_ids},
#         # "resid_post_layer_19": {"trainer_ids": trainer_ids},
# },
#     "gemma-2-2b_sweep_standard_ctx128_ef8_0824": {
#         # "resid_post_layer_3": {"trainer_ids": trainer_ids},
#         # "resid_post_layer_7": {"trainer_ids": trainer_ids},
#         "resid_post_layer_11": {"trainer_ids": trainer_ids},
#         # "resid_post_layer_15": {"trainer_ids": trainer_ids},
#         # "resid_post_layer_19": {"trainer_ids": trainer_ids},
#     },
# }

# Set manually in console
# if pipeline_config.use_autointerp:
#     with open("../anthropic_api_key.txt", "r") as f:
#         api_key = f.read().strip()

#     os.environ["ANTHROPIC_API_KEY"] = api_key

# This will look for any empty folders in any ae_path and raise an error if it finds any
for sweep_name, submodule_trainers in ae_sweep_paths.items():
    ae_group_paths = utils.get_ae_group_paths(
        pipeline_config.dictionaries_path, sweep_name, submodule_trainers
    )

start_time = time.time()

for sweep_name, submodule_trainers in ae_sweep_paths.items():
    run_interventions(
        submodule_trainers,
        pipeline_config,
        sweep_name,
        selection_method,
        T_effects,  # NOTE: I moved this to p_config
        T_max_sideeffect,
        random_seed,
        chosen_class_indices=chosen_class_indices,
        verbose=True,
    )

end_time = time.time()

print(f"Time taken: {end_time - start_time} seconds")
