import os
import time

from experiments.pipeline_config import PipelineConfig
import experiments.utils as utils
import experiments.llm_autointerp.llm_utils as llm_utils
import experiments.bib_intervention as bib_intervention

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    random_seed = 42

    # Use for debugging / any time you need to run from root dir
    # dictionaries_path = "dictionary_learning/dictionaries"
    # probes_dir = "experiments/trained_bib_probes"

    # Example of sweeping over all SAEs in a sweep
    ae_sweep_paths = {"pythia70m_test_sae": None}

    # Example of sweeping over all SAEs in a submodule
    ae_sweep_paths = {"pythia70m_test_sae": {"resid_post_layer_3": {"trainer_ids": None}}}

    # Example of sweeping over a single SAE
    ae_sweep_paths_list = [{"pythia70m_test_sae": {"resid_post_layer_3": {"trainer_ids": [0]}}}]

    trainer_ids = [2, 6, 10, 14, 18]

    ae_sweep_paths_list = [
        {
            "pythia70m_sweep_standard_ctx128_0712": {
                #     # "resid_post_layer_0": {"trainer_ids": None},
                #     # "resid_post_layer_1": {"trainer_ids": None},
                #     # "resid_post_layer_2": {"trainer_ids": None},
                # "resid_post_layer_3": {"trainer_ids": [6]},
                "resid_post_layer_4": {"trainer_ids": None},
            },
            "pythia70m_sweep_gated_ctx128_0730": {
                # "resid_post_layer_0": {"trainer_ids": None},
                # "resid_post_layer_1": {"trainer_ids": None},
                # "resid_post_layer_2": {"trainer_ids": None},
                # "resid_post_layer_3": {"trainer_ids": [2, 6, 10, 18]},
                "resid_post_layer_4": {"trainer_ids": None},
            },
            "pythia70m_sweep_panneal_ctx128_0730": {
                # "resid_post_layer_0": {"trainer_ids": None},
                # "resid_post_layer_1": {"trainer_ids": None},
                # "resid_post_layer_2": {"trainer_ids": None},
                # "resid_post_layer_3": {"trainer_ids": trainer_ids},
                "resid_post_layer_4": {"trainer_ids": None},
            },
            "pythia70m_sweep_topk_ctx128_0730": {
                # "resid_post_layer_0": {"trainer_ids": None},
                # "resid_post_layer_1": {"trainer_ids": None},
                # "resid_post_layer_2": {"trainer_ids": None},
                # "resid_post_layer_3": {"trainer_ids": [2, 6, 10, 18]},
                "resid_post_layer_4": {"trainer_ids": None},
            },
        }
    ]

    trainer_ids = None
    trainer_ids = [0, 2, 4]

    ae_sweep_paths_list = [
        {
            "gemma-2-2b_sweep_standard_ctx128_ef8_0824": {
                "resid_post_layer_11": {"trainer_ids": trainer_ids},
            },
            "gemma-2-2b_sweep_topk_ctx128_ef8_0824": {
                "resid_post_layer_11": {"trainer_ids": trainer_ids},
            },
            "gemma-2-2b_sweep_jumprelu_0902": {
                "resid_post_layer_11": {"trainer_ids": trainer_ids},
            },
        },
        {
            "gemma-2-2b_sweep_standard_ctx128_ef8_0824": {
                "resid_post_layer_19": {"trainer_ids": trainer_ids},
            },
            "gemma-2-2b_sweep_topk_ctx128_ef8_0824": {
                "resid_post_layer_19": {"trainer_ids": trainer_ids},
            },
            "gemma-2-2b_sweep_jumprelu_0902": {
                "resid_post_layer_19": {"trainer_ids": trainer_ids},
            },
        },
    ]

    column1_vals_list = [
        ("professor", "nurse"),
        ("architect", "journalist"),
        # ("surgeon", "psychologist"),
        # ("attorney", "teacher"),
    ]

    for ae_sweep_paths in ae_sweep_paths_list:
        for column1_vals in column1_vals_list:
            pipeline_config = PipelineConfig()

            pipeline_config.sweep_output_dir = "09_17_gemma_spurious_autointerp"

            # trainer_nums = [0, 2, 3, 5]
            # step_nums = [4882, 19528, 48828]

            # sae_name_filters = []
            # for trainer_num in trainer_nums:
            #     for step_num in step_nums:
            #         sae_name_filters.append(f"trainer_{trainer_num}_step_{step_num}")

            # sae_name_filters.append("trainer_0_step_0")

            # print(f"Only analyzing {sae_name_filters}")

            # pipeline_config.sae_name_filters = sae_name_filters
            pipeline_config.sae_name_filters = []

            pipeline_config.api_llm = "gpt-4o-mini-2024-07-18"
            pipeline_config.autointerp_t_effects = [2, 5, 10, 20, 50]
            pipeline_config.num_top_features_per_class = 50

            pipeline_config.force_ablations_recompute = True
            pipeline_config.force_node_effects_recompute = True
            pipeline_config.force_autointerp_recompute = True

            pipeline_config.use_autointerp = True

            pipeline_config.spurious_corr = True
            pipeline_config.column1_vals = column1_vals

            if pipeline_config.use_autointerp:
                llm_utils.set_api_key(pipeline_config.api_llm, "../")

            # This will look for any empty folders in any ae_path and raise an error if it finds any
            for sweep_name, submodule_trainers in ae_sweep_paths.items():
                ae_group_paths = utils.get_ae_group_paths(
                    pipeline_config.dictionaries_path, sweep_name, submodule_trainers
                )

            start_time = time.time()

            for sweep_name, submodule_trainers in ae_sweep_paths.items():
                bib_intervention.run_interventions(
                    submodule_trainers,
                    pipeline_config,
                    sweep_name,
                    random_seed,
                    verbose=True,
                )

            end_time = time.time()

            print(f"Time taken: {end_time - start_time} seconds")
