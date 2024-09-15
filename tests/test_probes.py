import torch
from nnsight import LanguageModel

import experiments.probe_training as probe_training


def test_probing_tpp():
    # If encountering errors, increase tolerance
    tolerance = 0.01
    device = "cuda"
    probe_output_filename = ""

    llm_model_name = "EleutherAI/pythia-70m-deduped"
    model = LanguageModel(
        llm_model_name, device_map=device, dispatch=True, torch_dtype=torch.bfloat16
    )

    chosen_class_indices = [0, 1, 2, 6, 9]

    test_accuracies = probe_training.train_probes(
        train_set_size=4000,
        test_set_size=1000,
        model=model,
        context_length=128,
        probe_batch_size=1000,
        llm_batch_size=500,
        llm_model_name=llm_model_name,
        epochs=10,
        device=device,
        probe_output_filename=probe_output_filename,
        spurious_correlation_removal=False,
        chosen_class_indices=chosen_class_indices,
        dataset_name="bias_in_bios",
        save_results=False,
        seed=42,
    )

    expected_accuracies = {
        0: (0.8870000243186951, 0.878000020980835, 0.8960000276565552, 0.29296875),
        1: (0.9270000457763672, 0.9280000329017639, 0.9260000586509705, 0.2314453125),
        2: (0.9010000228881836, 0.8880000710487366, 0.9140000343322754, 0.2490234375),
        6: (0.9750000238418579, 0.9600000381469727, 0.9900000691413879, 0.08935546875),
        9: (0.9480000734329224, 0.9360000491142273, 0.9600000381469727, 0.140625),
    }

    print(test_accuracies)

    for class_idx in expected_accuracies:
        difference = abs(test_accuracies[class_idx][0] - expected_accuracies[class_idx][0])

        assert difference < tolerance


def test_probing_spurious_correlation():
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
        llm_model_name=llm_model_name,
        epochs=10,
        device=device,
        probe_output_filename=probe_output_filename,
        spurious_correlation_removal=True,
        dataset_name="bias_in_bios",
        save_results=False,
        seed=42,
        column1_vals=("professor", "nurse"),
        column2_vals=("male", "female"),
    )

    expected_accuracies = {
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
