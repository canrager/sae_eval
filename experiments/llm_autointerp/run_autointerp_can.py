# Need to set ANTHROPIC_API_KEY

import os
from transformers import AutoTokenizer
from experiments.pipeline_config import PipelineConfig
from experiments.llm_autointerp.llm_query import perform_llm_autointerp


debug_mode = True
ae_path = "../dictionary_learning/dictionaries/autointerp_test_data/pythia70m_sweep_topk_ctx128_0730/resid_post_layer_3/trainer_2"
pythia_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")

cfg = PipelineConfig()
cfg.prompt_dir = "llm_autointerp/"
cfg.force_autointerp_recompute = True

# for k, v in vars(cfg).items():
#     print(f"{k}: {v}")

perform_llm_autointerp(pythia_tokenizer, cfg, ae_path, debug_mode=debug_mode)