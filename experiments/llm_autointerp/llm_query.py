import asyncio
import anthropic
from tenacity import retry, stop_after_attempt
import json
import pickle
import os
from tqdm.asyncio import tqdm

import experiments.llm_autointerp.llm_utils as llm_utils
import experiments.llm_autointerp.prompts as prompts


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
    return prompt_index, llm_response, json_response, good_json, verification_message


async def run_all_prompts(
    number_of_test_examples: int,
    client: anthropic.Anthropic,
    model: str,
    system_prompt: str,
    test_prompts: list[str],
    few_shot_examples: str,
    min_scale: int,
    max_scale: int,
    chosen_class_names: list[str],
) -> list[tuple[str, dict, bool, str]]:
    tasks = [
        process_prompt(
            client,
            model,
            system_prompt[0]["text"],
            test_prompts[i][0],
            test_prompts[i][4],
            few_shot_examples,
            min_scale,
            max_scale,
            chosen_class_names,
        )
        for i in range(number_of_test_examples)
    ]

    results = []
    async for result in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Processing prompts"
    ):
        prompt_index, llm_response, json_response, good_json, verification_message = await result
        results.append((prompt_index, llm_response, json_response, good_json, verification_message))
        print(len(results) - 1, good_json, verification_message)

    return results


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
    model = "claude-3-5-sonnet-20240620"
    # model = "claude-3-haiku-20240307"

    # IMPORTANT NOTE: We are using prompt caching. Before running on many prompts, run a single prompt
    # two times with number_of_test_examples = 1 and verify that
    # the cache_creation_input_tokens is 0 and cache_read_input_tokens is > 3000 on the second call.
    # Then you can run on many prompts with number_of_test_examples > 1.
    number_of_test_examples = 1

    with open(f"{PROMPT_DIR}/manual_labels_few_shot.json", "r") as f:
        few_shot_manual_labels = json.load(f)

    with open(f"{PROMPT_DIR}/manual_labels_can_final.json", "r") as f:
        manual_test_labels = json.load(f)

    few_shot_examples = prompts.create_few_shot_examples(few_shot_manual_labels)
    system_prompt = prompts.build_system_prompt(
        concepts=chosen_class_names, min_scale=min_scale, max_scale=max_scale
    )

    print(f"Few shot example is using {llm_utils.count_tokens(few_shot_examples)} tokens")
    print(f"System prompt is using {llm_utils.count_tokens(system_prompt[0]['text'])} tokens")

    test_prompts = prompts.create_test_prompts(manual_test_labels)

    client = anthropic.AsyncAnthropic()

    results = asyncio.run(
        run_all_prompts(
            number_of_test_examples,
            client,
            model,
            system_prompt,
            test_prompts,
            few_shot_examples,
            min_scale,
            max_scale,
            chosen_class_names,
        )
    )

    with open("llm_results.pkl", "wb") as f:
        pickle.dump(results, f)
