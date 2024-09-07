import torch
import os
from transformers import AutoTokenizer

from experiments.pipeline_config import PipelineConfig
import experiments.utils as utils
import experiments.llm_autointerp.llm_query as llm_query


# def test_decoding():
#     input_FKL = torch.randint(0, 1000, (10, 10, 128))

#     pythia_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")

#     output1 = utils.list_decode(input_FKL, pythia_tokenizer)
#     output2 = utils.batch_decode_to_tokens(input_FKL, pythia_tokenizer)

#     assert output1 == output2

# We comment these tests by default because they use API tokens when running
# Use sonnet 3.5 for more reliable results


def test_llm_query():
    ae_path = "dictionary_learning/dictionaries/autointerp_test_data/pythia70m_sweep_topk_ctx128_0730/resid_post_layer_3/trainer_2"

    with open("anthropic_api_key.txt", "r") as f:
        api_key = f.read().strip()

    os.environ["ANTHROPIC_API_KEY"] = api_key

    debug_mode = True

    pythia_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")

    p_config = PipelineConfig()

    p_config.num_top_features_per_class = 2
    p_config.prompt_dir = "experiments/llm_autointerp/"
    p_config.force_autointerp_recompute = True

    node_effects_auto_interp, node_effects_bias_shift_dir1, node_effects_bias_shift_dir2 = (
        llm_query.perform_llm_autointerp(pythia_tokenizer, p_config, ae_path, debug_mode=debug_mode)
    )

    # It's a bit janky, but sonnet returns 4.0 for all of these values
    assert node_effects_auto_interp[2][12558] > 0.0
    assert node_effects_auto_interp["male / female"][11871] > 0.0
    assert node_effects_auto_interp["professor / nurse"][9743] > 0.0

    assert node_effects_bias_shift_dir1["male_professor / female_nurse"][11871] > 0.0
    assert node_effects_bias_shift_dir2["male_professor / female_nurse"][9743] > 0.0

    assert node_effects_bias_shift_dir1["male_professor / female_nurse"][9743] == 0.0
    assert node_effects_bias_shift_dir2["male_professor / female_nurse"][11871] == 0.0


# def test_llm_query():
#     with open("anthropic_api_key.txt", "r") as f:
#         api_key = f.read().strip()

#     os.environ["ANTHROPIC_API_KEY"] = api_key

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

#     p_config.prompt_dir = "experiments/llm_autointerp/"

#     # IMPORTANT NOTE: We are using prompt caching. Before running on many prompts, run a single prompt
#     # two times with number_of_test_examples = 1 and verify that
#     # the cache_creation_input_tokens is 0 and cache_read_input_tokens is > 3000 on the second call.
#     # Then you can run on many prompts with number_of_test_examples > 1.
#     number_of_test_examples = 4

#     result = llm_query.test_llm_vs_manual_labels(
#         p_config=p_config,
#         chosen_class_names=chosen_class_names,
#         number_of_test_examples=number_of_test_examples,
#         output_filename="llm_test_results.json",
#     )

#     expected_result = 2

#     assert result["1"][1]["dentist"] == expected_result
