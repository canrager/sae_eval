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
        probe_train_batch_size=8,
        probe_test_batch_size=1000,
        llm_batch_size=500,
        llm_model_name=llm_model_name,
        epochs=2,
        device=device,
        probe_output_filename=probe_output_filename,
        spurious_correlation_removal=False,
        chosen_class_indices=chosen_class_indices,
        dataset_name="bias_in_bios",
        save_results=False,
        seed=42,
    )

    expected_accuracies = {
        0: (0.878000020980835, 0.8720000386238098, 0.8840000629425049, 0.3359375),
        1: (0.9160000681877136, 0.9180000424385071, 0.9140000343322754, 0.291015625),
        2: (0.8860000371932983, 0.9040000438690186, 0.8680000305175781, 0.30078125),
        6: (0.9750000238418579, 0.9580000638961792, 0.9920000433921814, 0.12109375),
        9: (0.9420000314712524, 0.9200000166893005, 0.9640000462532043, 0.1708984375),
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
        probe_train_batch_size=8,
        probe_test_batch_size=1000,
        llm_batch_size=500,
        llm_model_name=llm_model_name,
        epochs=2,
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
        "male / female": (0.987000048160553, 1.0, 0.9740000367164612, 0.1376953125),
        "professor / nurse": (
            0.9330000281333923,
            0.9280000329017639,
            0.9380000233650208,
            0.2001953125,
        ),
        "male_professor / female_nurse": (
            0.9910000562667847,
            0.9960000514984131,
            0.9860000610351562,
            0.0693359375,
        ),
    }

    print(test_accuracies)

    for class_idx in expected_accuracies:
        difference = abs(test_accuracies[class_idx][0] - expected_accuracies[class_idx][0])

        assert difference < tolerance
