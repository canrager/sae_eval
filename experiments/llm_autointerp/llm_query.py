import asyncio
import anthropic
from tenacity import retry, stop_after_attempt
import json
import pickle
import os
from tqdm.asyncio import tqdm
import torch
from transformers import AutoTokenizer

import experiments.llm_autointerp.llm_utils as llm_utils
import experiments.llm_autointerp.prompts as prompts
import experiments.autointerp as autointerp
import experiments.utils as utils
import experiments.utils_bib_dataset as utils_bib_dataset

from experiments.pipeline_config import PipelineConfig


@retry(stop=stop_after_attempt(3))
async def anthropic_request_prompt_caching(
    client: anthropic.Anthropic,
    system_prompt: str,
    few_shot_examples: str,
    test_prompt: str,
    model: str,
) -> str:
    message = await client.beta.prompt_caching.messages.create(
        model=model,
        max_tokens=500,
        temperature=0,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": few_shot_examples,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": test_prompt},
                ],
            }
        ],
    )
    # After this is called a second time, cache_creation_input_tokens should be 0
    # cache_read_input_tokens should be > 3000 (len(few_shot_examples) + len(system_prompt))
    print(f"Cache creation tokens: {message.usage.cache_creation_input_tokens}")
    print(f"Cache read tokens: {message.usage.cache_read_input_tokens}")
    return message.content[0].text


@retry(stop=stop_after_attempt(1))
async def anthropic_request(
    client: anthropic.Anthropic,
    system_prompt: str,
    few_shot_examples: str,
    test_prompt: str,
    model: str,
) -> str:
    message = await client.messages.create(
        model=model,
        max_tokens=500,
        temperature=0,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": few_shot_examples,
                    },
                    {"type": "text", "text": test_prompt},
                ],
            }
        ],
    )
    return message.content[0].text


@retry(stop=stop_after_attempt(3))
async def fill_anthropic_prompt_cache(
    client: anthropic.Anthropic,
    system_prompt: str,
    few_shot_examples: str,
    model: str,
) -> str:
    message = await client.beta.prompt_caching.messages.create(
        model=model,
        max_tokens=5,
        temperature=0,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": few_shot_examples,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": "This is a test prompt to fill the prompt cache."},
                ],
            }
        ],
    )
    # After this is called a second time, cache_creation_input_tokens should be 0
    # cache_read_input_tokens should be > 3000 (len(few_shot_examples) + len(system_prompt))
    print(f"Cache creation tokens: {message.usage.cache_creation_input_tokens}")
    print(f"Cache read tokens: {message.usage.cache_read_input_tokens}")
    return message.content[0].text


async def process_prompt(
    client: anthropic.Anthropic,
    model: str,
    system_prompt: str,
    test_prompt: str,
    prompt_index: int,
    few_shot_examples: str,
    min_scale: int,
    max_scale: int,
    chosen_class_names: list[str],
) -> tuple[int, str, dict, bool, str]:
    llm_response = await anthropic_request_prompt_caching(
        client, system_prompt, few_shot_examples, test_prompt, model
    )
    json_response = llm_utils.extract_and_validate_json(llm_response)
    good_json, verification_message = llm_utils.verify_json_response(
        json_response, min_scale, max_scale, chosen_class_names
    )
    return prompt_index, llm_response, json_response, good_json, verification_message, test_prompt


async def run_all_prompts(
    number_of_test_examples: int,
    client: anthropic.Anthropic,
    api_llm: str,
    system_prompt: str,
    test_prompts: dict[int, str],
    few_shot_examples: str,
    min_scale: int,
    max_scale: int,
    chosen_class_names: list[str],
    debug_mode: bool = False,
) -> dict[int, tuple[str, dict, bool, str]]:
    tasks = [
        process_prompt(
            client,
            api_llm,
            system_prompt[0]["text"],
            test_prompts[idx],
            idx,
            few_shot_examples,
            min_scale,
            max_scale,
            chosen_class_names,
        )
        for idx in sorted(test_prompts.keys())[:number_of_test_examples]
    ]

    results = {}
    async for result in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Processing prompts"
    ):
        (
            prompt_index,
            llm_response,
            json_response,
            good_json,
            verification_message,
            test_prompt,
        ) = await result

        results[prompt_index] = (
            llm_response,
            json_response,
            good_json,
            verification_message,
            test_prompt,
        )

        print(len(results) - 1, good_json, verification_message)

    if debug_mode:
        with open("llm_debug_results.json", "w") as f:
            json.dump(results, f)

    return results


def test_llm_vs_manual_labels(
    p_config: PipelineConfig,
    chosen_class_names: list[str],
    number_of_test_examples: int,
    output_filename: str = "llm_results.pkl",
):
    client = anthropic.AsyncAnthropic()

    with open(f"{p_config.prompt_dir}/manual_labels_can_final.json", "r") as f:
        manual_test_labels = json.load(f)

    few_shot_examples = prompts.create_few_shot_examples(prompt_dir=p_config.prompt_dir)
    system_prompt = prompts.build_system_prompt(
        concepts=chosen_class_names,
        min_scale=p_config.llm_judge_min_scale,
        max_scale=p_config.llm_judge_max_scale,
    )

    print(f"Few shot example is using {llm_utils.count_tokens(few_shot_examples)} tokens")
    print(f"System prompt is using {llm_utils.count_tokens(system_prompt[0]['text'])} tokens")

    test_prompts, test_prompt_metadata = prompts.create_test_prompts(manual_test_labels)

    test_prompts_tokens = 0

    for idx, test_prompt in test_prompts.items():
        test_prompts_tokens += llm_utils.count_tokens(test_prompt)

    average_test_prompts_tokens = test_prompts_tokens / len(test_prompts)
    print(
        f"Test prompts are using {test_prompts_tokens} tokens total, {average_test_prompts_tokens} tokens on average"
    )

    results = asyncio.run(
        run_all_prompts(
            number_of_test_examples,
            client,
            p_config.api_llm,
            system_prompt,
            test_prompts,
            few_shot_examples,
            p_config.llm_judge_min_scale,
            p_config.llm_judge_max_scale,
            chosen_class_names,
        )
    )

    with open(output_filename, "wb") as f:
        pickle.dump(results, f)


def construct_llm_features_prompts(
    ae_path: str,
    tokenizer: AutoTokenizer,
    p_config: PipelineConfig,
) -> dict[int, str]:
    with open(f"{ae_path}/max_activating_inputs.pkl", "rb") as f:
        max_activating_inputs = pickle.load(f)

    print(max_activating_inputs.keys())

    max_token_idxs_FKL = max_activating_inputs["max_tokens_FKL"]
    max_activations_FKL = max_activating_inputs["max_activations_FKL"]
    top_dla_token_idxs_FK = max_activating_inputs["dla_results_FK"]

    with open(f"{ae_path}/node_effects.pkl", "rb") as f:
        node_effects_attrib_patching = pickle.load(f)

    unique_feature_indices = []
    for class_name in node_effects_attrib_patching.keys():
        _, top_k_indices = torch.topk(
            node_effects_attrib_patching[class_name], p_config.num_top_features_per_class
        )
        unique_feature_indices.append(top_k_indices)

    all_unique_feature_indices = torch.cat(unique_feature_indices).unique()

    feature_prompts = {}

    for feature_idx in all_unique_feature_indices:
        feature_token_idxs_1KL = max_token_idxs_FKL[
            feature_idx, : p_config.num_top_inputs_per_feature, :
        ].unsqueeze(0)
        feature_activations_1KL = max_activations_FKL[
            feature_idx, : p_config.num_top_inputs_per_feature, :
        ].unsqueeze(0)
        feature_dla_token_idxs_K = top_dla_token_idxs_FK[feature_idx]

        example_prompt = autointerp.format_examples(
            tokenizer,
            feature_token_idxs_1KL,
            feature_activations_1KL,
            p_config.num_top_emphasized_tokens,
            p_config.include_activation_values_in_prompt,
        )[0]
        tokens_list = utils.list_decode(feature_dla_token_idxs_K, tokenizer)
        tokens_string = ", ".join(tokens_list)

        feature_prompts[feature_idx.item()] = prompts.create_feature_prompt(
            example_prompt, tokens_string
        )

    return feature_prompts


def filter_node_effects_with_autointerp(
    node_effects_autointerp: dict[int, torch.Tensor],
    node_effects: dict[int, torch.Tensor],
    llm_score_threshold: float,
) -> dict[int, torch.Tensor]:
    for class_name, tensor in node_effects_autointerp.items():
        node_effects_autointerp[class_name] = (tensor > llm_score_threshold).float()

    # Now perform element-wise multiplication
    filtered_node_effects = {}
    for class_name in node_effects_autointerp.keys():
        if class_name in node_effects:
            filtered_node_effects[class_name] = (
                node_effects_autointerp[class_name] * node_effects[class_name]
            )
        else:
            raise ValueError(f"Class name {class_name} not found in node_effects")

    return filtered_node_effects


def llm_json_response_to_node_effects(
    llm_json_response: dict[int, dict[int | str, int]],
    ae_path: str,
    p_config: PipelineConfig,
):
    """Challenges with this function that should be addressed:
    node_effects.pkl has keys of ints 0-27 + utils.PAIRED_CLASS_KEYS
    llm_json_response has keys of 'doctor', 'accountant', etc. + professor, nurse, and gender.
    Now we have to stitch these together."""

    with open(f"{ae_path}/node_effects.pkl", "rb") as f:
        node_effects_attrib_patching = pickle.load(f)

    node_effects_auto_interp = {}

    first_node_effect = next(iter(node_effects_attrib_patching.values()))
    dict_size = first_node_effect.size(0)

    for class_name in node_effects_attrib_patching.keys():
        node_effects_auto_interp[class_name] = torch.zeros(dict_size)

    node_effects_bias_shift_dir1 = {
        "male_professor / female_nurse": torch.zeros(dict_size),
        "biased_male / biased_female": torch.zeros(dict_size),
    }

    node_effects_bias_shift_dir2 = node_effects_bias_shift_dir1.copy()

    # Step 1: Regular class names, no bias shift
    for sae_feature_idx in llm_json_response:
        # This will happen if the LLM returns bad json for this feature idx
        if llm_json_response[sae_feature_idx] is None:
            continue

        for class_idx in node_effects_attrib_patching.keys():
            if class_idx in utils.PAIRED_CLASS_KEYS:
                continue
            class_name = utils_bib_dataset.profession_int_to_str[class_idx]
            node_effects_auto_interp[class_idx][sae_feature_idx] = llm_json_response[
                sae_feature_idx
            ][class_name]

    # Step 2: Paired class names, no bias shift

    for sae_feature_idx in llm_json_response:
        # This will happen if the LLM returns bad json for this feature idx
        if llm_json_response[sae_feature_idx] is None:
            continue

        for class_name in node_effects_attrib_patching.keys():
            if class_name not in utils.PAIRED_CLASS_KEYS:
                continue

            professor_value = llm_json_response[sae_feature_idx]["professor"]
            nurse_value = llm_json_response[sae_feature_idx]["nurse"]
            gender_value = llm_json_response[sae_feature_idx]["gender"]

            professor_nurse_value = max(professor_value, nurse_value)
            professor_nurse_gender_value = max(professor_nurse_value, gender_value)

            if class_name == "male / female":
                node_effects_auto_interp[class_name][sae_feature_idx] = gender_value
            elif class_name == "professor / nurse":
                node_effects_auto_interp[class_name][sae_feature_idx] = professor_nurse_value
            elif (
                class_name == "male_professor / female_nurse"
                or class_name == "biased_male / biased_female"
            ):
                node_effects_auto_interp[class_name][sae_feature_idx] = professor_nurse_gender_value
            else:
                raise ValueError(f"Unknown class name: {class_name}")

    # Step 3: bias shift
    for sae_feature_idx in llm_json_response:
        # This will happen if the LLM returns bad json for this feature idx
        if llm_json_response[sae_feature_idx] is None:
            continue

        professor_value = llm_json_response[sae_feature_idx]["professor"]
        nurse_value = llm_json_response[sae_feature_idx]["nurse"]
        gender_value = llm_json_response[sae_feature_idx]["gender"]

        professor_nurse_value = max(professor_value, nurse_value)

        for class_name in node_effects_bias_shift_dir1.keys():
            if (
                class_name == "male_professor / female_nurse"
                or class_name == "biased_male / biased_female"
            ):
                node_effects_bias_shift_dir1[class_name][sae_feature_idx] = gender_value
                node_effects_bias_shift_dir2[class_name][sae_feature_idx] = professor_nurse_value

    node_effects_auto_interp = filter_node_effects_with_autointerp(
        node_effects_auto_interp, node_effects_attrib_patching, p_config.llm_judge_binary_threshold
    )

    node_effects_bias_shift_dir1 = filter_node_effects_with_autointerp(
        node_effects_bias_shift_dir1,
        node_effects_attrib_patching,
        p_config.llm_judge_binary_threshold,
    )

    node_effects_bias_shift_dir2 = filter_node_effects_with_autointerp(
        node_effects_bias_shift_dir2,
        node_effects_attrib_patching,
        p_config.llm_judge_binary_threshold,
    )

    with open(f"{ae_path}/{p_config.autointerp_filename}", "wb") as f:
        pickle.dump(node_effects_auto_interp, f)

    with open(f"{ae_path}/{p_config.bias_shift_dir1_filename}", "wb") as f:
        pickle.dump(node_effects_bias_shift_dir1, f)

    with open(f"{ae_path}/{p_config.bias_shift_dir2_filename}", "wb") as f:
        pickle.dump(node_effects_bias_shift_dir2, f)


def perform_llm_autointerp(
    tokenizer: AutoTokenizer,
    p_config: PipelineConfig,
    chosen_class_names: list[str],
    ae_path: str,
    debug_mode: bool = False,
):
    client = anthropic.AsyncAnthropic()

    few_shot_examples = prompts.create_few_shot_examples(prompt_dir=p_config.prompt_dir)
    system_prompt = prompts.build_system_prompt(
        concepts=chosen_class_names,
        min_scale=p_config.llm_judge_min_scale,
        max_scale=p_config.llm_judge_max_scale,
    )

    asyncio.run(
        fill_anthropic_prompt_cache(
            client, system_prompt[0]["text"], few_shot_examples, p_config.api_llm
        )
    )

    features_prompts = construct_llm_features_prompts(ae_path, tokenizer, p_config)

    results = asyncio.run(
        run_all_prompts(
            number_of_test_examples=len(features_prompts),
            client=client,
            api_llm=p_config.api_llm,
            system_prompt=system_prompt,
            test_prompts=features_prompts,
            few_shot_examples=few_shot_examples,
            min_scale=p_config.llm_judge_min_scale,
            max_scale=p_config.llm_judge_max_scale,
            chosen_class_names=chosen_class_names,
            debug_mode=debug_mode,
        )
    )

    # 1 is the index of the extracted json from the llm's response
    json_results = {idx: result[1] for idx, result in results.items()}

    llm_json_response_to_node_effects(json_results, ae_path, p_config)


if __name__ == "__main__":
    with open("../anthropic_api_key.txt", "r") as f:
        api_key = f.read().strip()

    os.environ["ANTHROPIC_API_KEY"] = api_key

    chosen_class_names = [
        "gender",
        "professor",
        "nurse",
        "accountant",
        "architect",
        "attorney",
        "dentist",
        "filmmaker",
    ]
    PROMPT_DIR = "llm_autointerp"

    min_scale = 0
    max_scale = 4
    api_llm = "claude-3-5-sonnet-20240620"
    api_llm = "claude-3-haiku-20240307"

    # IMPORTANT NOTE: We are using prompt caching. Before running on many prompts, run a single prompt
    # two times with number_of_test_examples = 1 and verify that
    # the cache_creation_input_tokens is 0 and cache_read_input_tokens is > 3000 on the second call.
    # Then you can run on many prompts with number_of_test_examples > 1.
    number_of_test_examples = 1

    debug_mode = True

    ae_path = "../dictionary_learning/dictionaries/autointerp_test_data/pythia70m_sweep_topk_ctx128_0730/resid_post_layer_3/trainer_2"
    pythia_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")

    p_config = PipelineConfig()

    perform_llm_autointerp(
        pythia_tokenizer, p_config, chosen_class_names, ae_path, debug_mode=debug_mode
    )

# if __name__ == "__main__":
#     with open("../anthropic_api_key.txt", "r") as f:
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
#     PROMPT_DIR = "llm_autointerp"

#     min_scale = 0
#     max_scale = 4
#     api_llm = "claude-3-5-sonnet-20240620"
#     api_llm = "claude-3-haiku-20240307"

#     # IMPORTANT NOTE: We are using prompt caching. Before running on many prompts, run a single prompt
#     # two times with number_of_test_examples = 1 and verify that
#     # the cache_creation_input_tokens is 0 and cache_read_input_tokens is > 3000 on the second call.
#     # Then you can run on many prompts with number_of_test_examples > 1.
#     number_of_test_examples = 1

#     test_llm_vs_manual_labels(
#         api_llm=api_llm,
#         chosen_class_names=chosen_class_names,
#         min_scale=min_scale,
#         max_scale=max_scale,
#         number_of_test_examples=number_of_test_examples,
#         prompt_dir=PROMPT_DIR,
#         output_filename="llm_results.pkl",
#     )
