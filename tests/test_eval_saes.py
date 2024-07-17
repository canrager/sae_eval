from nnsight import LanguageModel
import torch

import experiments.eval_saes as eval_saes
import experiments.utils as utils


def test_eval_saes():

    l0_tolerance = 1
    tolerance = 0.1

    DEVICE = "cuda"

    llm_batch_size = 20  # Approx 16GB VRAM on pythia70m with 128 context length
    # TODO: Don't hardcode context length
    context_length = 128
    n_inputs = 10000

    seed = 42
    torch.manual_seed(seed)

    ae_sweep_paths = {"pythia70m_test_sae": {"resid_post_layer_3": {"trainer_ids": [0]}}}

    for sweep_name, submodule_trainers in ae_sweep_paths.items():

        dictionaries_path = "dictionary_learning/dictionaries"

        model_eval_config = utils.ModelEvalConfig.from_sweep_name(sweep_name)
        model_name = model_eval_config.full_model_name

        model = LanguageModel(model_name, device_map=DEVICE, dispatch=True)

        ae_group_paths = utils.get_ae_group_paths(dictionaries_path, sweep_name, submodule_trainers)
        ae_paths = utils.get_ae_paths(ae_group_paths)

        eval_results = eval_saes.eval_saes(
            model,
            ae_paths,
            n_inputs,
            context_length,
            llm_batch_size,
            DEVICE,
            overwrite_prev_results=True,
        )

        expected_l0 = 80.80000305175781
        expected_frac_recovered = 0.952869713306427

        l0_difference = abs(eval_results["l0"] - expected_l0)
        assert l0_difference < l0_tolerance

        difference = abs(eval_results["frac_recovered"] - expected_frac_recovered)
        assert difference < tolerance