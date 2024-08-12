import torch
from nnsight import LanguageModel
from datasets import load_dataset
import json
import os

import experiments.utils as utils
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.evaluation import evaluate

DEBUGGING = False

if DEBUGGING:
    tracer_kwargs = dict(scan=True, validate=True)
else:
    tracer_kwargs = dict(scan=False, validate=False)


@torch.no_grad()
def eval_saes(
    model: LanguageModel,
    ae_paths: list[str],
    n_inputs: int,
    llm_batch_size: int,
    device: str,
    overwrite_prev_results: bool = False,
    transcoder: bool = False,
) -> dict:
    buffer_size = min(512, n_inputs)

    if transcoder:
        io = "in_and_out"
    else:
        io = "out"

    pile_dataset = load_dataset("NeelNanda/pile-10k", streaming=False)

    input_strings = []

    for i, example in enumerate(pile_dataset["train"]["text"]):
        if i == n_inputs:
            break
        input_strings.append(example)

    eval_results = {}

    for ae_path in ae_paths:
        output_filename = f"{ae_path}/eval_results.json"
        if not overwrite_prev_results:
            if os.path.exists(output_filename):
                print(f"Skipping {ae_path} as eval results already exist")
                continue

        submodule, dictionary, config = utils.load_dictionary(model, ae_path, device)

        activation_dim = config["trainer"]["activation_dim"]
        # TODO: Think about how to handle context length... should we instead use the same context length for all dictionaries?
        context_length = config["buffer"]["ctx_len"]

        activation_buffer_data = iter(input_strings)

        activation_buffer = ActivationBuffer(
            activation_buffer_data,
            model,
            submodule,
            n_ctxs=buffer_size,
            ctx_len=context_length,
            refresh_batch_size=llm_batch_size,
            out_batch_size=llm_batch_size,
            io=io,
            d_submodule=activation_dim,
            device=device,
        )

        eval_results = evaluate(
            dictionary, activation_buffer, context_length, llm_batch_size, io=io, device=device
        )

        hyperparameters = {
            # TODO: Add batching so n_inputs is actually n_inputs
            "n_inputs": llm_batch_size,
            "context_length": context_length,
        }
        eval_results["hyperparameters"] = hyperparameters

        print(eval_results)

        with open(output_filename, "w") as f:
            json.dump(eval_results, f)

    # return the final eval_results for testing purposes
    return eval_results


if __name__ == "__main__":
    DEVICE = "cuda"

    llm_batch_size = 10  # Approx 1.5GB VRAM on pythia70m with 128 context length
    n_inputs = 10000

    submodule_trainers = {"resid_post_layer_3": {"trainer_ids": [0]}}

    dictionaries_path = "../dictionary_learning/dictionaries"

    sweep_name = "pythia70m_test_sae"

    # Current recommended way to generate graphs. You can copy paste ae_sweep_paths directly from bib_intervention.py
    ae_sweep_paths = {"pythia70m_sweep_standard_ctx128_0712": None}
    sweep_name = list(ae_sweep_paths.keys())[0]
    submodule_trainers = ae_sweep_paths[sweep_name]

    model_eval_config = utils.ModelEvalConfig.from_sweep_name(sweep_name)
    model_name = model_eval_config.full_model_name

    model = LanguageModel(model_name, device_map=DEVICE, dispatch=True)

    ae_group_paths = utils.get_ae_group_paths(dictionaries_path, sweep_name, submodule_trainers)
    ae_paths = utils.get_ae_paths(ae_group_paths)

    eval_results = eval_saes(
        model,
        ae_paths,
        n_inputs,
        llm_batch_size,
        DEVICE,
    )

    print(f"Final eval results: {eval_results}")
