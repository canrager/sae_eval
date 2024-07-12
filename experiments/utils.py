import torch
import json
import os
from typing import TypeAlias, Any

from dictionary_learning import AutoEncoder, ActivationBuffer
from dictionary_learning.dictionary import (
    IdentityDict,
    GatedAutoEncoder,
    AutoEncoderNew,
)
from dictionary_learning.trainers.top_k import AutoEncoderTopK
from dictionary_learning.interp import examine_dimension

submodule_alias: TypeAlias = Any


def get_ae_group_paths(
    dictionaries_path: str, model_location: str, sweep_name: str, submodule_trainers: dict
) -> list[str]:
    for submodule in submodule_trainers.keys():
        submodule_trainers[submodule]["model_location"] = model_location
        submodule_trainers[submodule]["sweep_name"] = sweep_name

    ae_group_paths = []

    for submodule in submodule_trainers.keys():
        trainer_ids = submodule_trainers[submodule]["trainer_ids"]
        model_location = submodule_trainers[submodule]["model_location"]
        sweep_name = submodule_trainers[submodule]["sweep_name"]

        base_filename = f"{dictionaries_path}/{model_location}{sweep_name}/{submodule}"

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

    for root, dirs, files in os.walk(path):
        if "ae.pt" in files:
            folder_names.append(root + "/")

    return folder_names


def check_for_empty_folders(ae_group_paths: list[str]) -> bool:
    """So your run doesn't crash / do nothing interesting because folder 13 is empty."""
    for ae_group_path in ae_group_paths:
        if len(get_nested_folders(ae_group_path)) == 0:
            raise ValueError(f"No folders found in {ae_group_path}")
    return True


def load_dictionary(model, first_model_name: str, base_path: str, device: str):
    print(f"Loading dictionary from {base_path}")
    ae_path = f"{base_path}ae.pt"
    config_path = f"{base_path}config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    submodule_str = config["trainer"]["submodule_name"]
    layer = config["trainer"]["layer"]
    model_name = config["trainer"]["lm_name"]
    dict_class = config["trainer"]["dict_class"]

    assert (
        model_name == first_model_name
    ), f"Model name {model_name} does not match first model name {first_model_name}"

    submodule = get_submodule(model, model_name, submodule_str, layer)

    if dict_class == "AutoEncoder":
        dictionary = AutoEncoder.from_pretrained(ae_path, device=device)
    # elif dict_class == "IdentityDict":
    #     dictionary = IdentityDict.from_pretrained(ae_path, device=device)
    elif dict_class == "GatedAutoEncoder":
        dictionary = GatedAutoEncoder.from_pretrained(ae_path, device=device)
    elif dict_class == "AutoEncoderNew":
        dictionary = AutoEncoderNew.from_pretrained(ae_path, device=device)
    elif dict_class == "AutoEncoderTopK":
        dictionary = AutoEncoderTopK.from_pretrained(ae_path, device=device)
    else:
        raise ValueError(f"Dictionary class {dict_class} not supported")

    return submodule, dictionary, config


def get_submodule(model, model_name: str, submodule_str: str, layer: int):
    allowed_submodules = ["attention_out", "mlp_out", "resid_post"]
    allowed_model_names = ["EleutherAI/pythia-70m-deduped"]

    if model_name not in allowed_model_names:
        raise ValueError(f"model_name must be one of {allowed_model_names}")

    if model_name == "EleutherAI/pythia-70m-deduped":
        if "attention_out" in submodule_str:
            submodule = model.gpt_neox.layers[layer].attention
        elif "mlp_out" in submodule_str:
            submodule = model.gpt_neox.layers[layer].mlp
        elif "resid_post" in submodule_str:
            submodule = model.gpt_neox.layers[layer]
        else:
            raise ValueError(f"submodule_str must contain one of {allowed_submodules}")

    return submodule


def batch_dict_lists(
    input_dict: dict[int, list[str]], batch_size: int
) -> dict[int, list[list[str]]]:
    for key in input_dict.keys():
        input_dict[key] = batch_list(input_dict[key], batch_size)
    return input_dict


def batch_list(input_list: list[str], batch_size: int) -> list[list[str]]:
    return [input_list[i : i + batch_size] for i in range(0, len(input_list), batch_size)]
