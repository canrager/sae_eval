# %%
# Imports
import torch as t
from nnsight import LanguageModel
import argparse
import itertools
import gc

from dictionary_learning.training import trainSAE
from dictionary_learning.trainers.standard import StandardTrainer
from dictionary_learning.trainers.top_k import TrainerTopK, AutoEncoderTopK
from dictionary_learning.trainers.gdm import GatedSAETrainer
from dictionary_learning.trainers.p_anneal import PAnnealTrainer
from dictionary_learning.utils import zst_to_generator, hf_dataset_to_generator
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.dictionary import AutoEncoder, GatedAutoEncoder

# %%
DEVICE = "cuda:0"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True, help="where to store sweep")
    parser.add_argument("--no_wandb_logging", action="store_true", help="omit wandb logging")
    parser.add_argument("--dry_run", action="store_true", help="dry run sweep")
    parser.add_argument(
        "--layers", type=int, nargs="+", required=True, help="layers to train SAE on"
    )
    args = parser.parse_args()
    return args


def run_sae_training(
    layer: int,
    save_dir: str,
    device: str,
    dry_run: bool = False,
    no_wandb_logging: bool = False,
):
    # model and data parameters
    model_name = "google/gemma-2-2b"
    dataset_name = "/share/data/datasets/pile/the-eye.eu/public/AI/pile/train/00.jsonl.zst"
    context_length = 128

    buffer_size = int(2e3)
    llm_batch_size = 32  # 32 on a 24GB RTX 3090
    sae_batch_size = 4096
    num_tokens = 200_000_000

    # sae training parameters
    # random_seeds = t.arange(10).tolist()
    random_seeds = [0]
    initial_sparsity_penalties = [0.02, 0.025, 0.035, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
    initial_sparsity_penalties = [0.01, 0.05, 0.075, 0.1, 0.15]
    ks = [20, 40, 80, 160, 320]

    assert len(initial_sparsity_penalties) == len(ks) # This is pretty janky but it can't fail silently with the assert
    ks = {p: ks[i] for i, p in enumerate(initial_sparsity_penalties)}
    expansion_factors = [8, 32]
    expansion_factors = [8]

    # PAnneal sparsity penalties for pythia 70m. Had poor coverage of L0 100-400. Recommend adding 0.0333 and 0.0366
    # initial_sparsity_penalties = [
    #     0.01,
    #     0.02,
    #     0.03,
    #     0.04,
    #     0.05,
    #     0.075,
    #     0.1,
    # ]

    # Gated sparsity penalties for pythia 70m. No coverage from 10-50. Recommending adding 1.0 and 1.1.
    # initial_sparsity_penalties = [0.1, 0.3, 0.5, 0.7, 0.9]

    steps = int(num_tokens / sae_batch_size)  # Total number of batches to train
    save_steps = None
    warmup_steps = 1000  # Warmup period at start of training and after each resample
    resample_steps = None

    # standard sae training parameters
    learning_rates = [3e-4]

    # topk sae training parameters
    decay_start = 24000
    auxk_alpha = 1 / 32

    # p_anneal sae training parameters
    p_start = 1
    p_end = 0.2
    anneal_end = None  # steps - int(steps/10)
    sparsity_queue_length = 10
    anneal_start = 10000
    n_sparsity_updates = 10

    log_steps = 25  # Log the training on wandb
    if no_wandb_logging:
        log_steps = None

    model = LanguageModel(
        model_name,
        # token=hf_token, # I had to use huggingface-cli login for some reason
        device_map=device,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        torch_dtype=t.bfloat16,
    )
    submodule = model.model.layers[layer]
    submodule_name = f"resid_post_layer_{layer}"
    io = "out"
    activation_dim = model.config.hidden_size

    # generator = zst_to_generator(dataset_name)
    generator = hf_dataset_to_generator("monology/pile-uncopyrighted")

    activation_buffer = ActivationBuffer(
        generator,
        model,
        submodule,
        n_ctxs=buffer_size,
        ctx_len=context_length,
        refresh_batch_size=llm_batch_size,
        out_batch_size=sae_batch_size,
        io=io,
        d_submodule=activation_dim,
        device=device,
    )

    # create the list of configs
    trainer_configs = []
    for seed, initial_sparsity_penalty, expansion_factor, learning_rate in itertools.product(
        random_seeds, initial_sparsity_penalties, expansion_factors, learning_rates
    ):
        trainer_configs.extend(
            [
                # {
                #     "trainer": PAnnealTrainer,
                #     "dict_class": AutoEncoder,
                #     "activation_dim": activation_dim,
                #     "dict_size": expansion_factor * activation_dim,
                #     "lr": learning_rate,
                #     "sparsity_function": "Lp^p",
                #     "initial_sparsity_penalty": initial_sparsity_penalty,
                #     "p_start": p_start,
                #     "p_end": p_end,
                #     "anneal_start": int(anneal_start),
                #     "anneal_end": anneal_end,
                #     "sparsity_queue_length": sparsity_queue_length,
                #     "n_sparsity_updates": n_sparsity_updates,
                #     "warmup_steps": warmup_steps,
                #     "resample_steps": resample_steps,
                #     "steps": steps,
                #     "seed": seed,
                #     "wandb_name": f"PAnnealTrainer-pythia70m-{layer}",
                #     "layer": layer,
                #     "lm_name": model_name,
                #     "device": device,
                #     "submodule_name": submodule_name,
                # },
                # {
                #     "trainer": StandardTrainer,
                #     "dict_class": AutoEncoder,
                #     "activation_dim": activation_dim,
                #     "dict_size": expansion_factor * activation_dim,
                #     "lr": learning_rate,
                #     "l1_penalty": initial_sparsity_penalty,
                #     "warmup_steps": warmup_steps,
                #     "resample_steps": resample_steps,
                #     "seed": seed,
                #     "wandb_name": f"StandardTrainer-{model_name}-{submodule_name}",
                #     "layer": layer,
                #     "lm_name": model_name,
                #     "device": device,
                #     "submodule_name": submodule_name,
                # },
                {
                    "trainer": TrainerTopK,
                    "dict_class": AutoEncoderTopK,
                    "activation_dim": activation_dim,
                    "dict_size": expansion_factor * activation_dim,
                    "k": ks[initial_sparsity_penalty],
                    "auxk_alpha": auxk_alpha,  # see Appendix A.2
                    "decay_start": decay_start,  # when does the lr decay start
                    "steps": steps,  # when when does training end
                    "seed": seed,
                    "wandb_name": f"TopKTrainer-{model_name}-{submodule_name}",
                    "device": device,
                    "layer": layer,
                    "lm_name": model_name,
                    "submodule_name": submodule_name,
                },
                # {
                #     "trainer": GatedSAETrainer,
                #     "dict_class": GatedAutoEncoder,
                #     "activation_dim": activation_dim,
                #     "dict_size": expansion_factor * activation_dim,
                #     "lr": learning_rate,
                #     "l1_penalty": initial_sparsity_penalty,
                #     "warmup_steps": warmup_steps,
                #     "resample_steps": resample_steps,
                #     "seed": seed,
                #     "wandb_name": f"GatedSAETrainer-{model_name}-{submodule_name}",
                #     "device": device,
                #     "layer": layer,
                #     "lm_name": model_name,
                #     "submodule_name": submodule_name,
                # },
            ]
        )

    print(f"len trainer configs: {len(trainer_configs)}")
    save_dir = f"{save_dir}/{submodule_name}"

    if not dry_run:
        # actually run the sweep
        trainSAE(
            data=activation_buffer,
            trainer_configs=trainer_configs,
            steps=steps,
            save_steps=save_steps,
            save_dir=save_dir,
            log_steps=log_steps,
            use_wandb=not no_wandb_logging,
            wandb_entity="adam-karvonen",
            wandb_project="topk_sae_sweep",
        )


if __name__ == "__main__":
    args = get_args()
    for layer in args.layers:
        run_sae_training(
            layer=layer,
            save_dir=args.save_dir,
            device="cuda:0",
            dry_run=args.dry_run,
            no_wandb_logging=args.no_wandb_logging,
        )
