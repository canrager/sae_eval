import torch
from nnsight import LanguageModel

import experiments.probe_training as probe_training


def test_probing():
    # If encountering errors, increase tolerance
    tolerance = 0.01
    device = "cuda"
    probe_output_filename = ""

    llm_model_name = "EleutherAI/pythia-70m-deduped"
    model = LanguageModel(
        llm_model_name, device_map=device, dispatch=True, torch_dtype=torch.bfloat16
    )

    test_accuracies = probe_training.train_probes(
        train_set_size=4000,
        test_set_size=1000,
        model=model,
        context_length=128,
        probe_batch_size=1000,
        llm_batch_size=500,
        llm_model_name="EleutherAI/pythia-70m-deduped",
        epochs=10,
        device=device,
        probe_output_filename=probe_output_filename,
        save_results=False,
        seed=42,
        include_gender=True,
    )

    expected_accuracies = {
        0: (0.8860000371932983, 0.8640000224113464, 0.9080000519752502, 0.27734375),
        1: (0.8840000629425049, 0.906000018119812, 0.862000048160553, 0.30859375),
        2: (0.9020000696182251, 0.8880000710487366, 0.9160000681877136, 0.26953125),
        6: (0.9330000281333923, 0.9340000152587891, 0.9320000410079956, 0.1923828125),
        9: (0.906000018119812, 0.8680000305175781, 0.9440000653266907, 0.2314453125),
        11: (0.8790000677108765, 0.89000004529953, 0.8680000305175781, 0.2890625),
        13: (0.8940000534057617, 0.8720000386238098, 0.9160000681877136, 0.28515625),
        14: (0.9030000567436218, 0.9320000410079956, 0.8740000128746033, 0.2431640625),
        18: (0.8560000658035278, 0.8540000319480896, 0.8580000400543213, 0.314453125),
        19: (0.859000027179718, 0.8640000224113464, 0.8540000319480896, 0.328125),
        20: (0.8980000615119934, 0.8940000534057617, 0.9020000696182251, 0.2431640625),
        21: (0.8730000257492065, 0.8660000562667847, 0.8800000548362732, 0.328125),
        22: (0.8560000658035278, 0.862000048160553, 0.8500000238418579, 0.380859375),
        25: (0.878000020980835, 0.89000004529953, 0.8660000562667847, 0.30078125),
        26: (0.8190000653266907, 0.784000039100647, 0.8540000319480896, 0.38671875),
        "male / female": (0.9920000433921814, 1.0, 0.984000027179718, 0.07470703125),
        "professor / nurse": (
            0.9350000619888306,
            0.9380000233650208,
            0.9320000410079956,
            0.1669921875,
        ),
        "male_professor / female_nurse": (
            0.9920000433921814,
            0.9960000514984131,
            0.9880000352859497,
            0.037353515625,
        ),
        "biased_male / biased_female": (
            0.9780000448226929,
            0.9760000705718994,
            0.9800000190734863,
            0.0830078125,
        ),
    }

    print(test_accuracies)

    for class_idx in expected_accuracies:
        difference = abs(test_accuracies[class_idx][0] - expected_accuracies[class_idx][0])

        assert difference < tolerance
