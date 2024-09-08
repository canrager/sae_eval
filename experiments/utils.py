import torch
import json
import os
import shutil
from typing import List, TypeAlias, Any, Optional
from tqdm import tqdm, trange
from transformers import AutoTokenizer

from dictionary_learning import AutoEncoder, ActivationBuffer
from dictionary_learning.dictionary import (
    IdentityDict,
    GatedAutoEncoder,
    AutoEncoderNew,
    JumpReluAutoEncoder,
)
from dictionary_learning.trainers.top_k import AutoEncoderTopK
from dictionary_learning.interp import examine_dimension

submodule_alias: TypeAlias = Any

PAIRED_CLASS_KEYS = {
    "male / female": "female_data_only",
    "professor / nurse": "nurse_data_only",
    "male_professor / female_nurse": "female_nurse_data_only",
    "biased_male / biased_female": "biased_female_data_only",
}

POSITIVE_CLASS_LABEL = 0
NEGATIVE_CLASS_LABEL = 1


# Activation dim technically isn't needed as it can be accessed from HuggingFace model config.
# The key to access it changes between model architecures, so we would have to add that key per architecure plus some logic.
class ModelEvalConfig:
    CONFIGS = {
        "pythia70m": {
            "full_model_name": "EleutherAI/pythia-70m-deduped",
            "activation_dim": 512,
            "probe_layer": 4,
            "llm_batch_size": 500,
            "attribution_patching_batch_size": 250,
            "eval_results_batch_size": 100,
        },
        "pythia160m": {
            "full_model_name": "EleutherAI/pythia-160m-deduped",
            "activation_dim": 768,
            "probe_layer": 10,
            "llm_batch_size": 125,
            "attribution_patching_batch_size": 50,
            "eval_results_batch_size": 50,
        },
        "gemma-2-2b": {
            "full_model_name": "google/gemma-2-2b",
            "activation_dim": 2304,
            "probe_layer": 20,
            "llm_batch_size": 32,
            "attribution_patching_batch_size": 8,
            "eval_results_batch_size": 24,
        },
    }

    def __init__(self, model_name):
        config = self.CONFIGS.get(model_name)
        if config is None:
            raise ValueError(f"Unknown model: {model_name}")

        self.model_name = model_name
        self.full_model_name = config["full_model_name"]
        self.activation_dim = config["activation_dim"]
        self.probe_layer = config["probe_layer"]
        self.llm_batch_size = config["llm_batch_size"]
        self.attribution_patching_batch_size = config["attribution_patching_batch_size"]
        self.eval_results_batch_size = config["eval_results_batch_size"]

    @classmethod
    def from_sweep_name(cls, sweep_name):
        matching_configs = [
            model_name for model_name in cls.CONFIGS.keys() if model_name in sweep_name
        ]
        if len(matching_configs) != 1:
            raise ValueError(
                f"Expected exactly one matching model for sweep {sweep_name}, found {len(matching_configs)}"
            )
        return cls(matching_configs[0])

    @classmethod
    def from_full_model_name(cls, full_model_name):
        for model_name, config in cls.CONFIGS.items():
            if config["full_model_name"] == full_model_name:
                return cls(model_name)
        raise ValueError(f"Unknown full model name: {full_model_name}")


def get_ae_group_paths(
    dictionaries_path: str, sweep_name: str, submodule_trainers: Optional[dict]
) -> list[str]:
    if submodule_trainers is None:
        return [f"{dictionaries_path}/{sweep_name}"]

    ae_group_paths = []

    for submodule in submodule_trainers.keys():
        trainer_ids = submodule_trainers[submodule]["trainer_ids"]

        base_filename = f"{dictionaries_path}/{sweep_name}/{submodule}"

        if trainer_ids is None:
            ae_group_paths.append(base_filename)
        else:
            for trainer_id in trainer_ids:
                ae_group_paths.append(f"{base_filename}/trainer_{trainer_id}")

    check_for_empty_folders(ae_group_paths)

    return ae_group_paths


def get_ae_paths(ae_group_paths: list[str]) -> list[str]:
    ae_paths = []
    for ae_group_path in ae_group_paths:
        ae_paths.extend(get_nested_folders(ae_group_path))
    return ae_paths


def extract_results(
    src_folder: str, dst_folder: str, exclude_files: list[str], ae_paths: Optional[list[str]] = None
):
    """If ae_paths is not None, only copy folders that are in ae_paths."""
    for root, dirs, files in os.walk(src_folder):
        # Create corresponding directory in destination
        rel_path = os.path.relpath(root, src_folder)
        dst_dir = os.path.join(dst_folder, rel_path)

        if ae_paths is not None:
            if root not in ae_paths:
                continue

        os.makedirs(dst_dir, exist_ok=True)

        # Copy files, excluding those in exclude_files
        for file in files:
            if file not in exclude_files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst_dir, file)
                shutil.copy2(src_file, dst_file)


def check_if_spurious_correlation_removal(chosen_class_indices: list[str | int]) -> bool:
    in_paired_keys = [index in PAIRED_CLASS_KEYS.keys() for index in chosen_class_indices]
    all_in = all(in_paired_keys)
    none_in = not any(in_paired_keys)

    assert all_in or none_in, "All or none of the chosen class indices must be in PAIRED_CLASS_KEYS"

    spurious_correlation_removal = all_in
    return spurious_correlation_removal


def get_batch_sizes(
    model_eval_config: ModelEvalConfig,
    reduced_GPU_memory: bool,
    train_set_size: int,
    test_set_size: Optional[int] = None,
    probe_train_set_size: Optional[int] = None,
    probe_test_set_size: Optional[int] = None,
) -> tuple[int, int, int]:
    llm_batch_size = model_eval_config.llm_batch_size
    patching_batch_size = model_eval_config.attribution_patching_batch_size
    eval_results_batch_size = model_eval_config.eval_results_batch_size

    if reduced_GPU_memory:
        llm_batch_size //= 5
        llm_batch_size //= 5
        patching_batch_size //= 5

    assert train_set_size >= llm_batch_size

    if test_set_size is not None:
        assert test_set_size >= llm_batch_size
    if probe_train_set_size is not None:
        assert probe_train_set_size >= llm_batch_size
    if probe_test_set_size is not None:
        assert probe_test_set_size >= llm_batch_size

    return llm_batch_size, patching_batch_size, eval_results_batch_size


def to_device(data, device):
    """
    Recursively move tensors in a nested dictionary to desired device.
    """
    with torch.no_grad():
        if isinstance(data, dict):
            # If it's a dictionary, apply recursively to each value
            return {key: to_device(value, device) for key, value in data.items()}
        elif isinstance(data, list):
            # If it's a list, apply recursively to each element
            return [to_device(item, device) for item in data]
        elif isinstance(data, torch.Tensor):
            # If it's a tensor, move it to CPU
            return data.to(device)
        else:
            # If it's neither, return it as is
            return data


def get_nested_folders(path: str) -> list[str]:
    """
    Recursively get a list of folders that contain an ae.pt file, starting the search from the given path
    """
    folder_names = []

    # We use config.json so it also works for data folders
    for root, dirs, files in os.walk(path):
        if "config.json" in files:
            folder_names.append(root)

    return folder_names


def check_for_empty_folders(ae_group_paths: list[str]) -> bool:
    """So your run doesn't crash / do nothing interesting because folder 13 is empty."""
    for ae_group_path in ae_group_paths:
        if len(get_nested_folders(ae_group_path)) == 0:
            raise ValueError(f"No folders found in {ae_group_path}")
    return True


# TODO: Use model.device instead of device?
def load_dictionary(model, base_path: str, device: str, verbose: bool = True):
    if verbose:
        print(f"Loading dictionary from {base_path}")
    ae_path = f"{base_path}/ae.pt"
    config_path = f"{base_path}/config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    submodule_str = config["trainer"]["submodule_name"]
    layer = config["trainer"]["layer"]
    model_name = config["trainer"]["lm_name"]
    dict_class = config["trainer"]["dict_class"]

    first_model_name = model.config._name_or_path
    assert type(first_model_name) == str, "Model name must be a string"

    assert (
        model_name == first_model_name
    ), f"Model name {model_name} does not match first model name {first_model_name}"

    submodule = get_submodule(model, submodule_str, layer)

    if dict_class == "AutoEncoder":
        dictionary = AutoEncoder.from_pretrained(ae_path, device=device)
    # elif dict_class == "IdentityDict":
    #     dictionary = IdentityDict.from_pretrained(ae_path, device=device)
    elif dict_class == "GatedAutoEncoder":
        dictionary = GatedAutoEncoder.from_pretrained(ae_path, device=device)
    elif dict_class == "AutoEncoderNew":
        dictionary = AutoEncoderNew.from_pretrained(ae_path, device=device)
    elif dict_class == "AutoEncoderTopK":
        k = config["trainer"]["k"]
        dictionary = AutoEncoderTopK.from_pretrained(ae_path, k=k, device=device)
    elif dict_class == "JumpReluAutoEncoder":
        dictionary = JumpReluAutoEncoder.from_pretrained(ae_path, device=device)
    else:
        raise ValueError(f"Dictionary class {dict_class} not supported")

    return submodule, dictionary, config


def get_submodule(model, submodule_str: str, layer: int):
    allowed_submodules = ["attention_out", "mlp_out", "resid_post", "unembed"]

    model_architecture = model.config.architectures[0]

    assert type(model_architecture) == str, "Model architecture must be a string"

    if model_architecture == "GPTNeoXForCausalLM":
        if "attention_out" in submodule_str:
            submodule = model.gpt_neox.layers[layer].attention
        elif "mlp_out" in submodule_str:
            submodule = model.gpt_neox.layers[layer].mlp
        elif "resid_post" in submodule_str:
            submodule = model.gpt_neox.layers[layer]
        elif "unembed" in submodule_str:
            submodule = model.embed_out
        else:
            raise ValueError(f"submodule_str must contain one of {allowed_submodules}")
    elif model_architecture == "Gemma2ForCausalLM":
        if "resid_post" in submodule_str:
            submodule = model.model.layers[layer]
        elif "unembed" in submodule_str:
            submodule = model.lm_head
        else:
            raise ValueError(f"submodule_str must contain one of {allowed_submodules}")
    else:
        raise ValueError(f"Model architecture {model_architecture} not supported")

    return submodule


def batch_inputs(inputs, batch_size: int):
    if isinstance(inputs, list) and isinstance(inputs[0], str):
        return [inputs[i : i + batch_size] for i in range(0, len(inputs), batch_size)]
    elif isinstance(inputs, list) and isinstance(inputs[0], int):
        return [torch.tensor(inputs[i : i + batch_size]) for i in range(0, len(inputs), batch_size)]
    elif isinstance(inputs, torch.Tensor):
        return [inputs[i : i + batch_size] for i in range(0, len(inputs), batch_size)]
    elif (
        isinstance(inputs, dict)
        and "input_ids" in inputs.keys()
        and "attention_mask" in inputs.keys()
    ):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        return [
            {
                "input_ids": input_ids[i : i + batch_size],
                "attention_mask": attention_mask[i : i + batch_size],
            }
            for i in range(0, len(input_ids), batch_size)
        ]
    else:
        raise ValueError("Unsupported input type")


def tokenize_data(
    data: dict[int, list[str]], tokenizer, max_length: int, device: str
) -> dict[int, dict]:
    tokenized_data = {}
    for key, texts in tqdm(data.items(), desc="Tokenizing data"):
        # .data so we have a dict, not a BatchEncoding
        tokenized_data[key] = (
            tokenizer(
                texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            )
            .to(device)
            .data
        )
    return tokenized_data


def get_ctx_length(ae_paths: list[str]) -> int:
    first_ctx_len = None

    for path in ae_paths:
        config_path = os.path.join(path, "config.json")

        with open(config_path, "r") as f:
            config = json.load(f)

        ctx_len = config["buffer"]["ctx_len"]

        if first_ctx_len is None:
            first_ctx_len = ctx_len
            print(f"Context length of the first path: {first_ctx_len}")
        else:
            assert (
                ctx_len == first_ctx_len
            ), f"Mismatch in ctx_len at {path}. Expected {first_ctx_len}, got {ctx_len}"

    if first_ctx_len is None:
        raise ValueError("No paths found.")
    else:
        print("All context lengths are the same.")
    return first_ctx_len


def get_sae_layer(ae_paths: list[str]) -> int:
    """This function is used if you want to evaluate the probe on the same layer as the saes.
    This means that if calling this function you will have to specify every layer in `ae_sweep_paths`.
    That is, this:
    ae_sweep_paths = {
        "pythia70m_sweep_standard_ctx128_0712": {
            "resid_post_layer_3": {"trainer_ids": None},
        }
    }

    ae_sweep_paths = {
        "pythia70m_sweep_standard_ctx128_0712": {
            "resid_post_layer_4": {"trainer_ids": None},
        }
    }

    ae_sweep_paths = {
        "pythia70m_sweep_standard_ctx128_0712": {
            "resid_post_layer_3": {"trainer_ids": None},
            "resid_post_layer_4": {"trainer_ids": None},
        }
    }
    """
    first_sae_layer = None

    for path in ae_paths:
        config_path = os.path.join(path, "config.json")

        with open(config_path, "r") as f:
            config = json.load(f)

        sae_layer = config["trainer"]["layer"]

        if first_sae_layer is None:
            first_sae_layer = sae_layer
            print(f"Context length of the first path: {first_sae_layer}")
        else:
            assert (
                sae_layer == first_sae_layer
            ), f"Mismatch in ctx_len at {path}. Expected {first_sae_layer}, got {sae_layer}"

    if first_sae_layer is None:
        raise ValueError("No paths found.")
    else:
        print("All context lengths are the same.")
    return first_sae_layer


def list_decode(x, tokenizer):
    if len(x.shape) == 0:
        return tokenizer.decode(x, skip_special_tokens=True)
    else:
        return [list_decode(y, tokenizer) for y in x]


# Assert list_decode(x) == batch_decode_to_tokens(x) is currently failing, I haven't investigated further.

# def batch_decode_to_tokens(x: torch.Tensor, tokenizer: AutoTokenizer, batch_size: int = 10000) -> List[List[str]]:
#     # Flatten the tensor
#     flat_x = x.reshape(-1)

#     # Decode each token individually
#     decoded = []
#     for i in tqdm(range(0, len(flat_x), batch_size), desc="Decoding token ids"):
#         decoded.append(tokenizer.batch_decode(flat_x[i:i+batch_size], skip_special_tokens=True))
#         # print(f'decoded: {decoded}')
#         # break

#     # Reshape the result back to the original shape
#     return [decoded[i:i+x.shape[1]] for i in range(0, len(decoded), x.shape[1])]

# def batch_decode_to_tokens(x: torch.Tensor, tokenizer: AutoTokenizer) -> List[List[str]]:
#     total_sequences = x.shape[0]
#     context_length = x.shape[-1]

#     # Iterate over features
#     all_decoded_tokens = []
#     for i in trange(total_sequences, desc="Decoding token idxs"):
#         flat_batch = x[i].reshape(-1)
#         decoded_batch = tokenizer.batch_decode(flat_batch.unsqueeze(-1), skip_special_tokens=True)

#         feature_decoded_tokens = []
#         # Reshape and filter out empty tokens
#         for j in range(0, len(decoded_batch), context_length):
#             sequence = [token for token in decoded_batch[j:j+context_length] if token]
#             feature_decoded_tokens.append(sequence)
#         all_decoded_tokens.append(feature_decoded_tokens)

#     return all_decoded_tokens
