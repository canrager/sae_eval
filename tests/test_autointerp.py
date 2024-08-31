import torch
from transformers import AutoTokenizer

import experiments.utils as utils


def test_decoding():
    input_FKL = torch.randint(0, 1000, (10, 10, 128))

    pythia_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")

    output1 = utils.list_decode(input_FKL, pythia_tokenizer)
    output2 = utils.batch_decode_to_tokens(input_FKL, pythia_tokenizer)

    assert output1 == output2
