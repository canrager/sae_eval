# Need to set ANTHROPIC_API_KEY

import os
from transformers import AutoTokenizer
from experiments.pipeline_config import PipelineConfig
from experiments.llm_autointerp.llm_query import perform_llm_autointerp

with open("../anthropic_api_key.txt", "r") as f:
    api_key = f.read().strip()

os.environ["ANTHROPIC_API_KEY"] = api_key

debug_mode = True
ae_path = "../dictionary_learning/dictionaries/pythia70m_sweep_topk_ctx128_0730/resid_post_layer_3/trainer_10"
pythia_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")

cfg = PipelineConfig()
cfg.prompt_dir = "llm_autointerp/"
cfg.force_autointerp_recompute = True
cfg.chosen_autointerp_class_names = [
    "gender",
    "professor",
    "nurse",
    "accountant",
    "architect",
    "attorney",
    "dentist",
]

# cfg.node_effects_attrib_filename = "node_effects_dist_diff.pkl"

cfg.num_top_features_per_class = 20
# for k, v in vars(cfg).items():
#     print(f"{k}: {v}")

perform_llm_autointerp(pythia_tokenizer, cfg, ae_path, debug_mode=debug_mode)
