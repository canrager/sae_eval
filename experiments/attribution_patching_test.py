from attribution import patching_effect
from dictionary_learning.trainers.top_k import AutoEncoderTopK
from nnsight import LanguageModel

device = "cuda:0"
model_name = "EleutherAI/pythia-70m-deduped"
model = LanguageModel(model_name, device_map=device, dispatch=True)

submodules = [model.gpt_neox.layers[2]]

ae_config = {
    "activation_dim": 512,
    "dict_size": 8192,
    "k": 30,
}
dictionaries = {submodule: AutoEncoderTopK(**ae_config) for submodule in submodules}

clean_prompt = "Hello"


def metric_fn(model):
    # last_token_logits = model.logits[:, -1, :]  # Shape: [batch_size, vocab_size]
    # return last_token_logits.sum()
    acts = model.gpt_neox.layers[4].output[0]
    return acts.sum()


# def metric_fn(model, labels=None):
#     attn_mask = model.input[1]['attention_mask']
#     acts = model.gpt_neox.layers[layer].output[0]
#     acts = acts * attn_mask[:, :, None]
#     acts = acts.sum(1) / attn_mask.sum(1)[:, None]

#     return t.where(
#         labels == 0,
#         probe(acts),
#         - probe(acts)
#     )

effects, _, _, _ = patching_effect(
    clean=clean_prompt,
    patch=None,
    model=model,
    submodules=submodules,
    dictionaries=dictionaries,
    metric_fn=metric_fn,
    metric_kwargs={},
    method="attrib",
    # steps=10,
)

print(effects)
