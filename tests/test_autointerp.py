import torch
import os
from transformers import AutoTokenizer

from experiments.pipeline_config import PipelineConfig
import experiments.utils as utils
import experiments.llm_autointerp.llm_query as llm_query
import experiments.llm_autointerp.llm_utils as llm_utils


# def test_decoding():
#     input_FKL = torch.randint(0, 1000, (10, 10, 128))

#     pythia_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")

#     output1 = utils.list_decode(input_FKL, pythia_tokenizer)
#     output2 = utils.batch_decode_to_tokens(input_FKL, pythia_tokenizer)

#     assert output1 == output2


# We comment these tests by default because they use API tokens when running
# Use sonnet 3.5 for more reliable results


def test_llm_query():
    ae_path = "dictionary_learning/dictionaries/pythia70m_test_sae/resid_post_layer_3/trainer_0"

    debug_mode = True

    pythia_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")

    p_config = PipelineConfig()

    p_config.api_llm = "gpt-4o-mini-2024-07-18"

    llm_utils.set_api_key(p_config.api_llm, "")

    p_config.spurious_corr = True

    p_config.num_top_features_per_class = 2
    p_config.prompt_dir = "experiments/llm_autointerp/"
    p_config.force_autointerp_recompute = True

    p_config.chosen_autointerp_class_names = ["gender", "professor", "nurse"]

    node_effects_auto_interp, node_effects_bias_shift_dir1, node_effects_bias_shift_dir2 = (
        llm_query.perform_llm_autointerp(pythia_tokenizer, p_config, ae_path, debug_mode=debug_mode)
    )

    # It's a bit janky, but sonnet returns 4.0 for all of these values
    assert node_effects_auto_interp["male / female"][3243] > 0.0
    assert node_effects_auto_interp["professor / nurse"][1987] > 0.0

    assert node_effects_bias_shift_dir1["male_professor / female_nurse"][3243] > 0.0
    assert node_effects_bias_shift_dir2["male_professor / female_nurse"][1987] > 0.0

    assert node_effects_bias_shift_dir1["male_professor / female_nurse"][1987] == 0.0
    assert node_effects_bias_shift_dir2["male_professor / female_nurse"][3243] == 0.0


# def test_llm_query():
#     chosen_class_names = [
#         "gender",
#         "professor",
#         "nurse",
#         "accountant",
#         "architect",
#         "attorney",
#         "dentist",
#         "filmmaker",
#     ]

#     p_config = PipelineConfig()
#     p_config.api_llm = "gpt-4o-mini-2024-07-18"

#     llm_utils.set_api_key(p_config.api_llm, "")

#     p_config.prompt_dir = "experiments/llm_autointerp/"
#     p_config.spurious_corr = False

#     # IMPORTANT NOTE: We are using prompt caching. Before running on many prompts, run a single prompt
#     # two times with number_of_test_examples = 1 and verify that
#     # the cache_creation_input_tokens is 0 and cache_read_input_tokens is > 3000 on the second call.
#     # Then you can run on many prompts with number_of_test_examples > 1.
#     number_of_test_examples = 2

#     result = llm_query.test_llm_vs_manual_labels(
#         p_config=p_config,
#         chosen_class_names=chosen_class_names,
#         number_of_test_examples=number_of_test_examples,
#         output_filename="llm_test_results.json",
#     )

#     assert result["1"][1]["dentist"] > 0


# test_llm_query()
