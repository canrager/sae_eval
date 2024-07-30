#%%
from attribution import patching_effect
from dictionary_learning.trainers.top_k import AutoEncoderTopK
from nnsight import LanguageModel

device = 'cuda'
model_name = "EleutherAI/pythia-70m-deduped"
model = LanguageModel(model_name, device_map=device, dispatch=True)

submodules = [model.gpt_neox.layers[2], model.gpt_neox.layers[2].mlp]

ae_config = {
    'activation_dim': 512,
    'dict_size': 8192,
    'k': 30,
}
dictionaries = {submodule: AutoEncoderTopK(**ae_config) for submodule in submodules}

clean_prompt = "Hello"

def metric_fn(logits):
    return logits.sum()

effects, _, _, _ = patching_effect(
            clean=clean_prompt,
            patch=None,
            model=model,
            submodules=submodules,
            dictionaries=dictionaries,
            metric_fn=metric_fn,
            metric_kwargs={},
            method='attrib',
            # steps=10,
        )

# %%
