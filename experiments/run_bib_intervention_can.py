from experiments.bib_intervention import *
from experiments.dataset_info import profession_int_to_str
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



p_config = PipelineConfig()
p_config.spurious_corr = False
p_config.include_gender = False
p_config.gemma_probe_layer = 24

p_config.attrib_t_effects = [100]
p_config.autointerp_t_effects = [20]
p_config.num_top_features_per_class = 100



# For debugging with pythia
# p_config.model_dtype = torch.float32
# p_config.train_set_size = 500
# p_config.test_set_size = 500
p_config.force_autointerp_recompute = True
p_config.force_node_effects_recompute = True

p_config.chosen_class_indices = [
    0,
    1,
    2,
]
p_config.chosen_autointerp_class_names = [
    profession_int_to_str[i] for i in p_config.chosen_class_indices
]


# trainer_ids = [2, 10, 18]
# trainer_ids = [10]

# ae_sweep_paths = {
#     "pythia70m_sweep_standard_ctx128_0712": {
#         #     # "resid_post_layer_0": {"trainer_ids": None},
#         #     # "resid_post_layer_1": {"trainer_ids": None},
#         #     # "resid_post_layer_2": {"trainer_ids": None},
#         "resid_post_layer_3": {"trainer_ids": trainer_ids},
#         #     "resid_post_layer_4": {"trainer_ids": None},
#     },
#     # "pythia70m_sweep_gated_ctx128_0730": {
#     #     # "resid_post_layer_0": {"trainer_ids": None},
#     #     # "resid_post_layer_1": {"trainer_ids": None},
#     #     # "resid_post_layer_2": {"trainer_ids": None},
#     #     "resid_post_layer_3": {"trainer_ids": [2, 6, 10, 18]},
#     #     # "resid_post_layer_4": {"trainer_ids": None},
#     # },
#     # "pythia70m_sweep_panneal_ctx128_0730": {
#     #     # "resid_post_layer_0": {"trainer_ids": None},
#     #     # "resid_post_layer_1": {"trainer_ids": None},
#     #     # "resid_post_layer_2": {"trainer_ids": None},
#     #     "resid_post_layer_3": {"trainer_ids": trainer_ids},
#     #     # "resid_post_layer_4": {"trainer_ids": None},
#     # },
#     "pythia70m_sweep_topk_ctx128_0730": {
#         # "resid_post_layer_0": {"trainer_ids": None},
#         # "resid_post_layer_1": {"trainer_ids": None},
#         # "resid_post_layer_2": {"trainer_ids": None},
#         "resid_post_layer_3": {"trainer_ids": trainer_ids},
#         # "resid_post_layer_4": {"trainer_ids": None},
#     },
# }

trainer_ids = [0, 3, 5]

ae_sweep_paths = {
    "gemma-2-2b_sweep_topk_ctx128_ef8_0824": {
        # "resid_post_layer_3": {"trainer_ids": trainer_ids},
        # "resid_post_layer_7": {"trainer_ids": trainer_ids},
        "resid_post_layer_11": {"trainer_ids": trainer_ids},
        # "resid_post_layer_15": {"trainer_ids": trainer_ids},
        # "resid_post_layer_19": {"trainer_ids": trainer_ids},
},
    "gemma-2-2b_sweep_standard_ctx128_ef8_0824": {
        # "resid_post_layer_3": {"trainer_ids": trainer_ids},
        # "resid_post_layer_7": {"trainer_ids": trainer_ids},
        "resid_post_layer_11": {"trainer_ids": trainer_ids},
        # "resid_post_layer_15": {"trainer_ids": trainer_ids},
        # "resid_post_layer_19": {"trainer_ids": trainer_ids},
    },
}

# Set manually in console
# if pipeline_config.use_autointerp:
#     with open("../anthropic_api_key.txt", "r") as f:
#         api_key = f.read().strip()

#     os.environ["ANTHROPIC_API_KEY"] = api_key

# This will look for any empty folders in any ae_path and raise an error if it finds any
for sweep_name, submodule_trainers in ae_sweep_paths.items():
    ae_group_paths = utils.get_ae_group_paths(
        p_config.dictionaries_path, sweep_name, submodule_trainers
    )

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
