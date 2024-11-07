# Imports
import torch as t
from nnsight import LanguageModel
import argparse
import itertools
import gc
import os
from datetime import datetime

from dictionary_learning.training import trainSAE
from dictionary_learning.trainers.standard import StandardTrainer
from dictionary_learning.trainers.top_k import TrainerTopK, AutoEncoderTopK
from dictionary_learning.trainers.gdm import GatedSAETrainer
from dictionary_learning.trainers.p_anneal import PAnnealTrainer
from dictionary_learning.utils import zst_to_generator, hf_dataset_to_generator
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.dictionary import AutoEncoder, GatedAutoEncoder


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True, help="where to store sweep")
    parser.add_argument("--no_wandb_logging", action="store_true", help="omit wandb logging")
    parser.add_argument("--dry_run", action="store_true", help="dry run sweep")
    parser.add_argument("--num_tokens", type=int, required=True, help="total number of training tokens")
    parser.add_argument(
        "--layers", type=int, nargs="+", required=True, help="layers to train SAE on"
    )
    parser.add_argument(
        "--width_exponents", type=int, nargs="+", required=True, help="power of 2 for total number of SAE latents"
    )
    parser.add_argument(
        "--architecture", type=str, required=True, choices=["vanilla", "topk"], help="architecture of the SAE"
    )
    parser.add_argument("--device", type=str, help="device to train on")
    args = parser.parse_args()
    return args


def run_sae_training(
    layer: int,
    width_exponents: list[int],
    num_tokens: int,
    architecture: str,
    save_dir: str,
    device: str,
    dry_run: bool = False,
    no_wandb_logging: bool = False,
):
    # model and data parameters
    model_name = "google/gemma-2-2b"
    context_length = 128

    buffer_size = int(2048)
    llm_batch_size = 32  # 32 on a 24GB RTX 3090
    sae_batch_size = 2048  # 2048 on a 24GB RTX 3090

    # sae training parameters
    # random_seeds = t.arange(10).tolist()
    random_seeds = [0]
    initial_sparsity_penalties = [0.025, 0.035, 0.04, 0.05, 0.06, 0.07]
    # initial_sparsity_penalties = [0.01, 0.05, 0.075, 0.1, 0.15]
    ks = [20, 40, 80, 160, 320, 640]

    assert len(initial_sparsity_penalties) == len(
        ks
    )  # This is pretty janky but it can't fail silently with the assert
    ks = {p: ks[i] for i, p in enumerate(initial_sparsity_penalties)}
    dict_sizes = [int(2**i) for i in width_exponents]

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
    warmup_steps = 1000  # Warmup period at start of training and after each resample
    resample_steps = None

    desired_checkpoints = t.logspace(-3, 0, 7).tolist()
    desired_checkpoints = [0.0] + desired_checkpoints[:-1]
    desired_checkpoints.sort()

    save_steps = [int(steps * step) for step in desired_checkpoints]
    save_steps.sort()

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

    log_steps = 100  # Log the training on wandb
    if no_wandb_logging:
        log_steps = None

    cache_dir = "/data/huggingface/"
    model = LanguageModel(
        model_name,
        device_map=device,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        torch_dtype=t.bfloat16,
        cache_dir=cache_dir,
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
    for seed, initial_sparsity_penalty, dict_size, learning_rate in itertools.product(
        random_seeds, initial_sparsity_penalties, dict_sizes, learning_rates
    ):
        if architecture == "vanilla":
            trainer_configs.extend([{
                    "trainer": StandardTrainer,
                    "dict_class": AutoEncoder,
                    "activation_dim": activation_dim,
                    "dict_size": dict_size,
                    "lr": learning_rate,
                    "l1_penalty": initial_sparsity_penalty,
                    "warmup_steps": warmup_steps,
                    "resample_steps": resample_steps,
                    "seed": seed,
                    "wandb_name": f"StandardTrainer-{model_name}-{submodule_name}",
                    "layer": layer,
                    "lm_name": model_name,
                    "device": device,
                    "submodule_name": submodule_name,
            }])

        if architecture == "topk":
            trainer_configs.extend([{
                "trainer": TrainerTopK,
                "dict_class": AutoEncoderTopK,
                "activation_dim": activation_dim,
                "dict_size": dict_size,
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
            }])

            #     # {
            #     #     "trainer": PAnnealTrainer,
            #     #     "dict_class": AutoEncoder,
            #     #     "activation_dim": activation_dim,
            #     #     "dict_size": expansion_factor * activation_dim,
            #     #     "lr": learning_rate,
            #     #     "sparsity_function": "Lp^p",
            #     #     "initial_sparsity_penalty": initial_sparsity_penalty,
            #     #     "p_start": p_start,
            #     #     "p_end": p_end,
            #     #     "anneal_start": int(anneal_start),
            #     #     "anneal_end": anneal_end,
            #     #     "sparsity_queue_length": sparsity_queue_length,
            #     #     "n_sparsity_updates": n_sparsity_updates,
            #     #     "warmup_steps": warmup_steps,
            #     #     "resample_steps": resample_steps,
            #     #     "steps": steps,
            #     #     "seed": seed,
            #     #     "wandb_name": f"PAnnealTrainer-pythia70m-{layer}",
            #     #     "layer": layer,
            #     #     "lm_name": model_name,
            #     #     "device": device,
            #     #     "submodule_name": submodule_name,
            #     # },
            #     # {
            #     #     "trainer": GatedSAETrainer,
            #     #     "dict_class": GatedAutoEncoder,
            #     #     "activation_dim": activation_dim,
            #     #     "dict_size": expansion_factor * activation_dim,
            #     #     "lr": learning_rate,
            #     #     "l1_penalty": initial_sparsity_penalty,
            #     #     "warmup_steps": warmup_steps,
            #     #     "resample_steps": resample_steps,
            #     #     "seed": seed,
            #     #     "wandb_name": f"GatedSAETrainer-{model_name}-{submodule_name}",
            #     #     "device": device,
            #     #     "layer": layer,
            #     #     "lm_name": model_name,
            #     #     "submodule_name": submodule_name,
            #     # },
            # ]

    mmdd = datetime.now().strftime('%m%d')
    model_id = model_name.split('/')[1]
    width_str = "_".join([str(i) for i in width_exponents])
    save_name = f"{model_id}_{architecture}_layer-{layer}_width-2pow{width_str}_date-{mmdd}"
    save_dir = os.path.join(save_dir, save_name)
    print(f"save_dir: {save_dir}")
    print(f"desired_checkpoints: {desired_checkpoints}")
    print(f"save_steps: {save_steps}")
    print(f"num_tokens: {num_tokens}")
    print(f"len trainer configs: {len(trainer_configs)}")
    print(f"trainer_configs: {trainer_configs}")
   

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
            wandb_entity="canrager",
            wandb_project="checkpoint_sae_sweep",
        )


if __name__ == "__main__":
    args = get_args()
    for layer in args.layers:
        run_sae_training(
            layer=layer,
            save_dir=args.save_dir,
            num_tokens=args.num_tokens,
            width_exponents=args.width_exponents,
            architecture=args.architecture,
            device=args.device,
            dry_run=args.dry_run,
            no_wandb_logging=args.no_wandb_logging,
        )
