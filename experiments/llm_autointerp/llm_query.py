import asyncio
import time
import anthropic
import openai
from tenacity import retry, stop_after_attempt, RetryCallState
import random
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
import experiments.dataset_info as dataset_info

from experiments.pipeline_config import PipelineConfig


def exponential_backoff(retry_state: RetryCallState):
    # Base wait time (in seconds)
    base_wait = 5
    # Maximum wait time (in seconds)
    max_wait = 1000
    # Exponential factor
    factor = 2

    wait = min(base_wait * (factor ** (retry_state.attempt_number - 1)), max_wait)
    # Add jitter to avoid thundering herd problem
    jitter = random.uniform(0, 0.1 * wait)
    return wait + jitter


@retry(
    stop=stop_after_attempt(9),
    wait=exponential_backoff,
)
async def anthropic_request_prompt_caching(
    client: anthropic.Anthropic,
    system_prompt: str,
    few_shot_examples: str,
    test_prompt: str,
    model: str,
    verbose: bool = False,
) -> str:
    print("Attempting API call")
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
    if verbose:
        # After this is called a second time, cache_creation_input_tokens should be 0
        # cache_read_input_tokens should be > 3000 (len(few_shot_examples) + len(system_prompt))
        print(f"Cache creation tokens: {message.usage.cache_creation_input_tokens}")
        print(f"Cache read tokens: {message.usage.cache_read_input_tokens}")
    return message.content[0].text


@retry(
    stop=stop_after_attempt(9),
    wait=exponential_backoff,
)
async def openai_request(
    client: openai.OpenAI,
    system_prompt: str,
    few_shot_examples: str,
    test_prompt: str,
    model: str,
    verbose: bool = False,
) -> str:
    if verbose:
        print("Attempting OpenAI API call")
    message = await client.chat.completions.create(
        model=model,
        max_tokens=500,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": few_shot_examples + test_prompt,
            },
        ],
    )
    with open("openai_response.json", "w") as f:
        json.dump(message.choices[0].message.content, f, indent=4)
    return message.choices[0].message.content


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


@retry(
    stop=stop_after_attempt(9),
    wait=exponential_backoff,
)
async def fill_anthropic_prompt_cache(
    client: anthropic.Anthropic,
    system_prompt: str,
    few_shot_examples: str,
    model: str,
) -> str:
    print("Attempting prompt fill API call")
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
    client: anthropic.Anthropic | openai.OpenAI,
    model: str,
    system_prompt: str,
    test_prompt: str,
    prompt_index: int,
    few_shot_examples: str,
    min_scale: int,
    max_scale: int,
    chosen_class_names: list[str],
) -> tuple[int, str, dict, bool, str]:
    if "claude" in model:
        llm_response = await anthropic_request_prompt_caching(
            client, system_prompt, few_shot_examples, test_prompt, model
        )
    elif "gpt" in model:
        llm_response = await openai_request(
            client, system_prompt, few_shot_examples, test_prompt, model
        )
    else:
        raise ValueError("Model name must contain 'claude' or 'gpt'")
    json_response = llm_utils.extract_and_validate_json(llm_response)
    good_json, verification_message = llm_utils.verify_json_response(
        json_response, min_scale, max_scale, chosen_class_names
    )
    if good_json:
        json_response = llm_utils.zero_out_non_max_values_in_json_response(json_response)
    else:
        json_response = None
    return prompt_index, llm_response, json_response, good_json, verification_message, test_prompt


async def run_all_prompts(
    number_of_test_examples: int,
    client: anthropic.Anthropic | openai.OpenAI,
    api_llm: str,
    system_prompt: str,
    test_prompts: dict[int, str],
    few_shot_examples: str,
    min_scale: int,
    max_scale: int,
    chosen_class_names: list[str],
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

    return results


def test_llm_vs_manual_labels(
    p_config: PipelineConfig,
    chosen_class_names: list[str],
    number_of_test_examples: int,
    output_filename: str = "llm_results.json",
) -> dict[int, tuple[str, dict, bool, str]]:
    client = llm_utils.get_async_client(p_config.api_llm)

    with open(os.path.join(p_config.prompt_dir, "manual_labels_can_final.json"), "r") as f:
        manual_test_labels = json.load(f)

    few_shot_examples = prompts.load_few_shot_examples(
        prompt_dir=p_config.prompt_dir, spurious_corr=p_config.spurious_corr
    )
    system_prompt = prompts.build_system_prompt(
        concepts=chosen_class_names,
        min_scale=p_config.llm_judge_min_scale,
        max_scale=p_config.llm_judge_max_scale,
    )

    print(f"Few shot example is using {llm_utils.count_tokens(few_shot_examples)} tokens")
    print(f"System prompt is using {llm_utils.count_tokens(system_prompt[0]['text'])} tokens")

    test_prompts = prompts.create_test_prompts(manual_test_labels, chosen_class_names)

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

    with open(output_filename, "w") as f:
        json.dump(results, f, indent=4)

    return results


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

    with open(f"{ae_path}/{p_config.node_effects_attrib_filename}", "rb") as f:
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
            feature_idx, : p_config.num_top_inputs_autointerp, :
        ].unsqueeze(0)
        feature_activations_1KL = max_activations_FKL[
            feature_idx, : p_config.num_top_inputs_autointerp, :
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
            example_prompt, tokens_string, p_config.chosen_autointerp_class_names
        )

    return feature_prompts


def filter_node_effects_with_autointerp(
    node_effects_autointerp: torch.Tensor,
    node_effects: torch.Tensor,
    top_k_features: int,
    llm_score_threshold: float,
) -> torch.Tensor:
    node_effects_autointerp = (node_effects_autointerp > llm_score_threshold).float()

    top_k_values, top_k_indices = torch.topk(node_effects, top_k_features)
    masked_node_effects = torch.zeros_like(node_effects)
    masked_node_effects[top_k_indices] = top_k_values

    filtered_node_effects = node_effects_autointerp * masked_node_effects

    return filtered_node_effects


def get_node_effects_auto_interp_tpp(
    node_effects_attrib_patching: dict[int, torch.Tensor],
    dict_size: int,
    llm_json_response: dict[int, dict[int, int]],
) -> dict[int, torch.Tensor]:
    node_effects_auto_interp = {}

    for class_name in node_effects_attrib_patching.keys():
        node_effects_auto_interp[class_name] = torch.zeros(dict_size, device="cpu")

    for sae_feature_idx in llm_json_response:
        # This will happen if the LLM returns bad json for this feature idx
        if llm_json_response[sae_feature_idx] is None:
            continue

        for class_idx in node_effects_attrib_patching.keys():
            assert isinstance(class_idx, int)
            class_name = dataset_info.profession_int_to_str[class_idx]
            node_effects_auto_interp[class_idx][sae_feature_idx] = llm_json_response[
                sae_feature_idx
            ][class_name]

    return node_effects_auto_interp


def get_node_effects_auto_interp_spurious(
    node_effects_attrib_patching: dict[str, torch.Tensor],
    dict_size: int,
    llm_json_response: dict[int, dict[str, int]],
    column1_vals: tuple[str, str],
    column2_autointerp_name: str,
) -> dict[str, torch.Tensor]:
    node_effects_auto_interp = {}

    for class_name in node_effects_attrib_patching.keys():
        node_effects_auto_interp[class_name] = torch.zeros(dict_size, device="cpu")

    for sae_feature_idx in llm_json_response:
        # This will happen if the LLM returns bad json for this feature idx
        if llm_json_response[sae_feature_idx] is None:
            continue

        for class_name in node_effects_attrib_patching.keys():
            # assert isinstance(class_name, str)
            if isinstance(class_name, int):
                continue

            # Note that getting professor / nurse indexing right doesn't really matter, as we take the max
            professor_value = llm_json_response[sae_feature_idx][column1_vals[0]]
            nurse_value = llm_json_response[sae_feature_idx][column1_vals[1]]
            gender_value = llm_json_response[sae_feature_idx][column2_autointerp_name]

            professor_nurse_value = max(professor_value, nurse_value)
            professor_nurse_gender_value = max(professor_nurse_value, gender_value)

            if class_name == "male / female":
                node_effects_auto_interp[class_name][sae_feature_idx] = gender_value
            elif class_name == "professor / nurse":
                node_effects_auto_interp[class_name][sae_feature_idx] = professor_nurse_value
            elif class_name == "male_professor / female_nurse":
                node_effects_auto_interp[class_name][sae_feature_idx] = professor_nurse_gender_value
            else:
                raise ValueError(f"Unknown class name: {class_name}")

    return node_effects_auto_interp


def get_node_effects_bias_shift(
    node_effects_attrib_patching: dict[str, torch.Tensor],
    dict_size: int,
    llm_json_response: dict[int, dict[str, int]],
    column1_vals: tuple[str, str],
    column2_autointerp_name: str,
) -> dict[str, torch.Tensor]:
    node_effects_bias_shift_dir1 = {
        "male_professor / female_nurse": torch.zeros(dict_size, device="cpu"),
        "male / female": torch.zeros(dict_size, device="cpu"),
        "professor / nurse": torch.zeros(dict_size, device="cpu"),
    }

    node_effects_bias_shift_dir2 = {
        "male_professor / female_nurse": torch.zeros(dict_size, device="cpu"),
        "male / female": torch.zeros(dict_size, device="cpu"),
        "professor / nurse": torch.zeros(dict_size, device="cpu"),
    }

    class1_key = column1_vals[0]
    class2_key = column1_vals[1]

    for sae_feature_idx in llm_json_response:
        # This will happen if the LLM returns bad json for this feature idx
        if llm_json_response[sae_feature_idx] is None:
            continue

        for class_name in node_effects_attrib_patching.keys():
            assert isinstance(class_name, str)

            professor_value = llm_json_response[sae_feature_idx][class1_key]
            nurse_value = llm_json_response[sae_feature_idx][class2_key]
            gender_value = llm_json_response[sae_feature_idx][column2_autointerp_name]

            professor_nurse_value = max(professor_value, nurse_value)

            if class_name == "male / female":
                node_effects_bias_shift_dir1[class_name][sae_feature_idx] = gender_value
                node_effects_bias_shift_dir2[class_name][sae_feature_idx] = gender_value
            elif class_name == "professor / nurse":
                node_effects_bias_shift_dir1[class_name][sae_feature_idx] = professor_nurse_value
                node_effects_bias_shift_dir2[class_name][sae_feature_idx] = professor_nurse_value

    # Step 3: bias shift
    for sae_feature_idx in llm_json_response:
        # This will happen if the LLM returns bad json for this feature idx
        if llm_json_response[sae_feature_idx] is None:
            continue

        professor_value = llm_json_response[sae_feature_idx][class1_key]
        nurse_value = llm_json_response[sae_feature_idx][class2_key]
        gender_value = llm_json_response[sae_feature_idx][column2_autointerp_name]

        professor_nurse_value = max(professor_value, nurse_value)

        unrelated_gender_value = 4
        unrelated_professor_nurse_value = 4

        if gender_value > 0:
            unrelated_gender_value = 0

        if professor_nurse_value > 0:
            unrelated_professor_nurse_value = 0

        for class_name in node_effects_bias_shift_dir1.keys():
            if class_name == "male_professor / female_nurse":
                node_effects_bias_shift_dir1[class_name][sae_feature_idx] = (
                    unrelated_professor_nurse_value
                )
                node_effects_bias_shift_dir2[class_name][sae_feature_idx] = unrelated_gender_value

    return node_effects_bias_shift_dir1, node_effects_bias_shift_dir2


def llm_json_response_to_node_effects(
    llm_json_response: dict[int, dict[int | str, int]],
    ae_path: str,
    p_config: PipelineConfig,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Challenges with this function that should be addressed:
    node_effects.pkl has keys of ints 0-27 + utils.PAIRED_CLASS_KEYS
    llm_json_response has keys of 'doctor', 'accountant', etc. + professor, nurse, and gender.
    Now we have to stitch these together."""

    with open(f"{ae_path}/node_effects.pkl", "rb") as f:
        node_effects_attrib_patching = pickle.load(f)

    first_node_effect = next(iter(node_effects_attrib_patching.values()))
    dict_size = first_node_effect.size(0)

    if p_config.spurious_corr:
        column2_autointerp_name = dataset_info.dataset_metadata[p_config.dataset_name][
            "column2_autointerp_name"
        ]

        node_effects_auto_interp = get_node_effects_auto_interp_spurious(
            node_effects_attrib_patching,
            dict_size,
            llm_json_response,
            p_config.column1_vals,
            column2_autointerp_name,
        )
        node_effects_bias_shift_dir1, node_effects_bias_shift_dir2 = get_node_effects_bias_shift(
            node_effects_attrib_patching,
            dict_size,
            llm_json_response,
            p_config.column1_vals,
            column2_autointerp_name,
        )

        with open(f"{ae_path}/{p_config.bias_shift_dir1_filename}", "wb") as f:
            pickle.dump(node_effects_bias_shift_dir1, f)

        with open(f"{ae_path}/{p_config.bias_shift_dir2_filename}", "wb") as f:
            pickle.dump(node_effects_bias_shift_dir2, f)
    else:
        node_effects_auto_interp = get_node_effects_auto_interp_tpp(
            node_effects_attrib_patching, dict_size, llm_json_response
        )
        node_effects_bias_shift_dir1 = {}
        node_effects_bias_shift_dir2 = {}

    with open(f"{ae_path}/{p_config.autointerp_filename}", "wb") as f:
        pickle.dump(node_effects_auto_interp, f)

    with open(os.path.join(ae_path, "autointerp_pipeline_config.pkl"), "wb") as f:
        pickle.dump(p_config, f)

    return node_effects_auto_interp, node_effects_bias_shift_dir1, node_effects_bias_shift_dir2


def perform_llm_autointerp(
    tokenizer: AutoTokenizer,
    p_config: PipelineConfig,
    ae_path: str,
    debug_mode: bool = False,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    if not p_config.force_autointerp_recompute and os.path.exists(
        f"{ae_path}/{p_config.autointerp_filename}"
    ):
        print("Loading auto interp results from file")
        with open(f"{ae_path}/{p_config.autointerp_filename}", "rb") as f:
            node_effects_auto_interp = pickle.load(f)

        with open(f"{ae_path}/{p_config.bias_shift_dir1_filename}", "rb") as f:
            node_effects_bias_shift_dir1 = pickle.load(f)

        with open(f"{ae_path}/{p_config.bias_shift_dir2_filename}", "rb") as f:
            node_effects_bias_shift_dir2 = pickle.load(f)

        return node_effects_auto_interp, node_effects_bias_shift_dir1, node_effects_bias_shift_dir2

    elif not os.path.exists(f"{ae_path}/{p_config.autointerp_filename}"):
        print("Auto interp results not found, performing LLM query")
    elif p_config.force_autointerp_recompute:
        print("Recomputing auto interp results")

    client = llm_utils.get_async_client(p_config.api_llm)

    few_shot_examples = prompts.load_few_shot_examples(
        prompt_dir=p_config.prompt_dir, spurious_corr=p_config.spurious_corr
    )
    system_prompt = prompts.build_system_prompt(
        concepts=p_config.chosen_autointerp_class_names,
        min_scale=p_config.llm_judge_min_scale,
        max_scale=p_config.llm_judge_max_scale,
    )
    system_prompt_tokens = llm_utils.count_tokens(system_prompt[0]["text"])
    few_shot_examples_tokens = llm_utils.count_tokens(few_shot_examples)
    p_config.num_tokens_system_prompt = system_prompt_tokens + few_shot_examples_tokens

    if "claude" in p_config.api_llm:
        asyncio.run(
            fill_anthropic_prompt_cache(
                client, system_prompt[0]["text"], few_shot_examples, p_config.api_llm
            )
        )

    features_prompts = construct_llm_features_prompts(ae_path, tokenizer, p_config)

    if debug_mode:
        with open(os.path.join(ae_path, "input_prompts.json"), "w") as f:
            json.dump(features_prompts, f)

    batches_prompt_indices = llm_utils.get_prompt_batch_indices(features_prompts, p_config)

    results = {}

    for b, batch_indices in enumerate(batches_prompt_indices):
        test_prompts = {idx: features_prompts[idx] for idx in batch_indices}
        results.update(
            asyncio.run(
                run_all_prompts(
                    len(test_prompts),
                    client,
                    p_config.api_llm,
                    system_prompt,
                    test_prompts,
                    few_shot_examples,
                    p_config.llm_judge_min_scale,
                    p_config.llm_judge_max_scale,
                    p_config.chosen_autointerp_class_names,
                )
            )
        )
        num_batches_left = len(batches_prompt_indices) - b - 1
        if num_batches_left > 0:
            seconds_delay = 65
            print(
                f"Finished batch of {len(test_prompts)} prompts. Waiting for {seconds_delay} seconds to obey API limits... There are {num_batches_left} batches left."
            )
            time.sleep(seconds_delay)

    # 1 is the index of the extracted json from the llm's response
    json_results = {idx: result[1] for idx, result in results.items()}

    if debug_mode:
        with open(os.path.join(ae_path, "raw_llm_outputs.json"), "w") as f:
            json.dump(results, f)

        with open(os.path.join(ae_path, "extracted_json_llm_outputs.json"), "w") as f:
            json.dump(json_results, f)

    return llm_json_response_to_node_effects(json_results, ae_path, p_config)


if __name__ == "__main__":
    debug_mode = True

    ae_path = "../dictionary_learning/dictionaries/autointerp_test_data/pythia70m_sweep_topk_ctx128_0730/resid_post_layer_3/trainer_2"
    pythia_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")

    p_config = PipelineConfig()

    llm_utils.set_api_key(p_config.api_llm, "../")

    p_config.chosen_autointerp_class_names = ["gender", "professor", "nurse"]
    p_config.spurious_corr = True

    perform_llm_autointerp(pythia_tokenizer, p_config, ae_path, debug_mode=debug_mode)

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

#     p_config = PipelineConfig()
#     p_config.spurious_corr = False

#     # IMPORTANT NOTE: We are using prompt caching. Before running on many prompts, run a single prompt
#     # two times with number_of_test_examples = 1 and verify that
#     # the cache_creation_input_tokens is 0 and cache_read_input_tokens is > 3000 on the second call.
#     # Then you can run on many prompts with number_of_test_examples > 1.
#     number_of_test_examples = 1

#     test_llm_vs_manual_labels(
#         p_config=p_config,
#         chosen_class_names=chosen_class_names,
#         number_of_test_examples=number_of_test_examples,
#         output_filename="llm_results.json",
#     )
