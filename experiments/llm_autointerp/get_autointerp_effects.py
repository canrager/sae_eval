import os
import torch as t
import json
import anthropic
from typing import List, Tuple, Dict
from experiments.llm_autointerp.prompts import (
    build_system_prompt,
    load_few_shot_examples,
    create_unlabeled_prompts,
)
from experiments.llm_autointerp import llm_query
from experiments.llm_autointerp import llm_utils
from experiments.autointerp import get_autointerp_inputs_for_all_saes, format_examples
from experiments import utils_bib_dataset


def extract_scores_llm(data: List[Tuple[str, Dict[str, int], bool, str]]) -> Dict[str, List[int]]:
    result = {}
    for item in data:
        scores = item[1]  # The scores dictionary is the second element of each tuple
        if scores is None:
            raise NotImplementedError("incorrect json formatting not handled")
        else:
            for category, score in scores.items():
                if category not in result:
                    result[category] = []
                result[category].append(score)
    return result


def node_effects_autointerp(
    model,
    cfg,
    node_effects_classprobe,
    max_token_idxs_FKL,
    max_activations_FKL,
    top_dla_token_idxs_FK,
    few_shot_manual_labels,
):
    # Select top features for each class
    topk_feature_idxs = {}
    all_features_set = set()
    for class_idx in cfg["chosen_class_idxs"]:
        effects = node_effects_classprobe[class_idx]
        top_k_indices = t.topk(effects, cfg["num_top_features_per_class"]).indices
        topk_feature_idxs[class_idx] = top_k_indices
        all_features_set.update(top_k_indices.tolist())
    all_features_set = list(all_features_set)

    max_token_idxs_FKL = max_token_idxs_FKL[all_features_set]
    max_activations_FKL = max_activations_FKL[all_features_set]
    top_dla_token_idxs_FK = top_dla_token_idxs_FK[all_features_set]

    # Static prompts
    system_prompt = build_system_prompt(
        concepts=list(utils_bib_dataset.profession_dict.keys()),
    )
    few_shot_examples = load_few_shot_examples(few_shot_manual_labels)

    print(f"Few shot example is using {llm_utils.count_tokens(few_shot_examples)} tokens")
    print(f"System prompt is using {llm_utils.count_tokens(system_prompt[0]['text'])} tokens")

    # Unlabeled prompts
    unlabeled_examples = format_examples(
        model.tokenizer, max_token_idxs_FKL, max_activations_FKL, cfg["num_top_emphasized_tokens"]
    )
    top_dla_token_strs_FK = utils.list_decode(top_dla_token_idxs_FK, model.tokenizer)
    unlabeled_prompts = create_unlabeled_prompts(unlabeled_examples, top_dla_token_strs_FK)
    unlabeled_prompt_idxs = list(range(len(unlabeled_prompts)))

    # Run LLM
    client = anthropic.AsyncAnthropic()

    llm_out = llm_query.run_all_prompts(
        number_of_test_examples=len(unlabeled_prompts),
        api_llm=cfg["llm_judge_name"],
        system_prompt=system_prompt,
        test_prompts=unlabeled_prompts,
        test_prompts_indices=unlabeled_prompt_idxs,
        few_shot_examples=few_shot_examples,
        client=client,
        min_scale=cfg["judge_min_scale"],
        max_scale=cfg["judge_max_scale"],
        chosen_class_names=cfg["chosen_class_strs"],
    )

    # Extract scores
    # TODO handle that only a subset of features is used
    llm_labels = extract_scores_llm(llm_out)

    node_effects = {}
    for labels in llm_labels:
        node_effects[utils_bib_dataset.profession_dict[labels]] = t.tensor(llm_labels[labels])

    # save as llm_autointerp/node_effects_autointerp.pkl
    with open(os.path.join("llm_autointerp", cfg["output_filename"]), "wb") as f:
        pickle.dump(node_effects, f)

    return None


def get_default_cfg():
    default_cfg = {
        "num_top_emphasized_tokens": 5,
        "num_top_inputs_per_feature": 5,
        "num_top_features_per_class": 20,
        "num_total_contexts_max_act": 10000,
        "batch_size_max_act": 256,
        "context_length": 128,
        "dict_size": 16384,
        "llm_judge_name": "claude-3-5-sonnet-20240620",
        "judge_min_scale": 0,
        "judge_max_scale": 4,
    }
    return default_cfg


if __name__ == "__main__":
    import torch as t
    import os
    import pickle
    from nnsight import LanguageModel
    from experiments import utils
    from experiments.utils_bib_dataset import profession_int_to_str

    # Load configs

    cfg = get_default_cfg()
    cfg["output_filename"] = "node_effects_autointerp.pkl"

    # Load Model
    DEVICE = "cuda"
    model_name = "EleutherAI/pythia-70m-deduped"
    model_dtype = t.bfloat16
    model = LanguageModel(
        model_name,
        device_map=DEVICE,
        dispatch=True,
        attn_implementation="eager",
        torch_dtype=model_dtype,
    )
    model_unembed = model.embed_out  # For direct logit attribution

    # Fetch AE paths
    dictionaries_path = "../dictionary_learning/dictionaries"

    ae_sweep_paths = {
        "pythia70m_sweep_standard_ctx128_0712": {"resid_post_layer_3": {"trainer_ids": [6]}},
        # "pythia70m_sweep_gated_ctx128_0730": {"resid_post_layer_3": {"trainer_ids": [9]}},
        # "pythia70m_sweep_topk_ctx128_0730": {"resid_post_layer_3": {"trainer_ids": [10]}},
        # "gemma-2-2b_sweep_topk_ctx128_0817": {"resid_post_layer_12": {"trainer_ids": [2]}},
    }
    sweep_name = list(ae_sweep_paths.keys())[0]
    submodule_trainers = ae_sweep_paths[sweep_name]

    ae_group_paths = utils.get_ae_group_paths(dictionaries_path, sweep_name, submodule_trainers)
    ae_paths = utils.get_ae_paths(ae_group_paths)

    # Load (or precompute) max activating inputs
    get_autointerp_inputs_for_all_saes(
        model,
        n_inputs=cfg["num_total_contexts_max_act"],
        batch_size=cfg["batch_size_max_act"],
        context_length=cfg["context_length"],
        top_k_inputs=cfg["num_top_inputs_per_feature"],
        ae_paths=ae_paths,
        force_rerun=False,
    )
    t.cuda.empty_cache()

    # For specific AE
    ae_path = ae_paths[0]

    with open(os.path.join(ae_path, "max_activating_inputs.pkl"), "rb") as f:
        file = pickle.load(f)

    max_token_idxs_FKL = file["max_tokens_FKL"]
    max_activations_FKL = file["max_activations_FKL"]
    top_dla_token_idxs_FK = file["dla_results_FK"]

    # Load manual labels
    with open(f"llm_autointerp/manual_labels_few_shot.json", "r") as f:
        few_shot_manual_labels = json.load(f)

    # Load node_effects_classprobe
    with open(os.path.join(ae_path, "node_effects.pkl"), "rb") as f:
        node_effects_classprobe = pickle.load(f)

    # Testing
    cfg["chosen_class_idxs"] = [0]

    node_effects_classprobe = {
        k: v for k, v in node_effects_classprobe.items() if k in cfg["chosen_class_idxs"]
    }
    cfg["chosen_class_strs"] = [profession_int_to_str[i] for i in node_effects_classprobe.keys()]

    # Run LLM
    llm_out = node_effects_autointerp(
        model,
        cfg,
        node_effects_classprobe,
        max_token_idxs_FKL,
        max_activations_FKL,
        top_dla_token_idxs_FK,
        few_shot_manual_labels,
    )
    print(llm_out)
