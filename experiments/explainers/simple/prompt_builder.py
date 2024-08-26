from typing import List

from .prompts import example, system


def build_examples(
    **kwargs,
):
    examples = []

    for i in range(1, 4):
        prompt, response = example(i, **kwargs)

        messages = [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": response,
            },
        ]

        examples.extend(messages)

    return examples


def build_prompt(
    examples,
    concept: str,
    cot: bool = False,
    few_shot: bool = False,
    activations: bool = False,
    top_logits: List[str] = None,
):
    logits = True if top_logits is not None else False

    system_prompt = system(
        concept=concept,
        cot=cot,
        logits=logits,
        activations=activations,
    )

    few_shot_examples = build_examples(
        concept=concept,
        cot=cot,
        logits=logits,
        activations=activations,
    )

    messages = few_shot_examples

    user_start = f"\n{examples}\n"

    if logits:
        user_start += f"\nTop_logits: {top_logits}"

    messages.append(
        {
            "role": "user",
            "content": user_start,
        }
    )

    return system_prompt, messages
