import tiktoken
import re
import json
import os
from typing import Optional
import anthropic
import openai

from experiments.pipeline_config import PipelineConfig


def extract_and_validate_json(text: str) -> Optional[dict[str, int]]:
    # Regex pattern to match JSON block
    pattern = r"```json\s*(.*?)\s*```"

    # Search for the pattern
    match = re.search(pattern, text, re.DOTALL)

    if match:
        json_str = match.group(1)
        try:
            # Attempt to parse the JSON
            json_obj = json.loads(json_str)
            return json_obj
        except json.JSONDecodeError:
            # If JSON is invalid, return None
            print("WARNING: Invalid JSON block")
            return None
    else:
        # If no JSON block found, return None
        print("WARNING: No JSON block found")
        return None


def set_api_key(model_name: str, current_dir: str):
    """Please include the `/` at the end of the current_dir."""
    if "claude" in model_name:
        with open(f"{current_dir}anthropic_api_key.txt", "r") as f:
            api_key = f.read().strip()
        os.environ["ANTHROPIC_API_KEY"] = api_key
    elif "gpt" in model_name:
        with open(f"{current_dir}openai_api_key.txt", "r") as f:
            api_key = f.read().strip()
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        raise ValueError("Model name must contain 'claude' or 'gpt'")


def get_async_client(model_name: str) -> anthropic.AsyncAnthropic | openai.AsyncOpenAI:
    if "claude" in model_name:
        return anthropic.AsyncAnthropic()
    elif "gpt" in model_name:
        return openai.AsyncOpenAI()
    else:
        raise ValueError("Model name must contain 'claude' or 'gpt'")


def verify_json_response(
    json_response: dict[str, int], min_val: int, max_val: int, chosen_class_names: list[str]
) -> tuple[bool, str]:
    if json_response is None or not isinstance(json_response, dict):
        return False, "Invalid JSON response"

    # Check if every chosen class name is present in the JSON response
    if set(json_response.keys()) != set(chosen_class_names):
        return False, "Mismatch between JSON keys and chosen class names"

    # Check if every value is an int from min_val to max_val
    for value in json_response.values():
        if not isinstance(value, int) or value < min_val or value > max_val:
            return False, f"Invalid value: all values must be integers from {min_val} to {max_val}"

    return True, "Verification passed"


def zero_out_non_max_values_in_json_response(json_response: dict[str, int]) -> dict[str, int]:
    """If the response is {professor: 4, gender: 1}, this should only count the professor class."""
    max_value = max(json_response.values())
    return {k: v if v == max_value else 0 for k, v in json_response.items()}


def count_tokens(prompt: str, model: str = "gpt-4") -> int:
    """For some reason Anthropic doesn't provide token counts, so we need to use gpt-4's tokenizer."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(prompt))
    return num_tokens


def get_rate_limits(model_name: str) -> tuple[int, int]:
    if "claude" in model_name:
        tokens_per_min = 400_000
        requests_per_min = 4_000
    elif "gpt" in model_name:
        tokens_per_min = 4_000_000
        requests_per_min = 5_000
    else:
        raise ValueError("Model name must contain 'claude' or 'gpt'")

    return tokens_per_min, requests_per_min


def get_prompt_batch_indices(prompts: dict[str, str], p_config: PipelineConfig):
    """Given a dictionary of prompts, return a list of lists of indices of the prompts to be queried in each batch."""
    assert (
        p_config.num_tokens_system_prompt is not None
    ), "num_tokens_system_prompt must be set in the config during the pipeline"
    prompts_num_tokens = {
        k: (count_tokens(v) + p_config.num_tokens_system_prompt) for k, v in prompts.items()
    }

    tokens_per_min, requests_per_min = get_rate_limits(p_config.api_llm)
    tokens_per_min *= p_config.max_percentage_of_num_allowed_tokens_per_minute
    requests_per_min *= p_config.max_percentage_of_num_allowed_requests_per_minute
    tokens_per_min = int(tokens_per_min)
    requests_per_min = int(requests_per_min)

    running_token_count = 0
    running_feat_idx_batch = []
    api_call_feat_idx_batches = []
    for feat_idx, num_tokens in prompts_num_tokens.items():
        if (len(running_feat_idx_batch) > requests_per_min) or (
            running_token_count + num_tokens > tokens_per_min
        ):
            api_call_feat_idx_batches.append(running_feat_idx_batch)
            running_feat_idx_batch = [feat_idx]
            running_token_count = num_tokens
        else:
            running_feat_idx_batch.append(feat_idx)
            running_token_count += num_tokens

    api_call_feat_idx_batches.append(running_feat_idx_batch)
    return api_call_feat_idx_batches
