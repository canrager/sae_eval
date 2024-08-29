import tiktoken
import re
import json
from typing import Optional


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


def count_tokens(prompt: str, model: str = "gpt-4") -> int:
    """For some reason Anthropic doesn't provide token counts, so we need to use gpt-4's tokenizer."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(prompt))
    return num_tokens
