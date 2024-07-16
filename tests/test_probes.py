import experiments.probe_training as probe_training


def test_probing():

    # If encountering errors, increase tolerance
    tolerance = 0.01

    test_accuracies = probe_training.train_probes(
        train_set_size=1000,
        test_set_size=1000,
        context_length=128,
        probe_batch_size=50,
        llm_batch_size=20,
        llm_model_name="EleutherAI/pythia-70m-deduped",
        epochs=10,
        device="cuda",
        save_results=False,
        seed=42,
    )

    expected_accuracies = {
        0: 0.8130000233650208,
        1: 0.8040000200271606,
        2: 0.8510000109672546,
        6: 0.9240000247955322,
        9: 0.8300000429153442,
        11: 0.8130000233650208,
        12: 0.8260000348091125,
        13: 0.8510000109672546,
        14: 0.8810000419616699,
        18: 0.8220000267028809,
        19: 0.859000027179718,
        20: 0.8390000462532043,
        21: 0.8420000672340393,
        22: 0.7990000247955322,
        24: 0.8800000548362732,
        25: 0.8480000495910645,
        26: 0.7570000290870667,
    }

    for class_idx in expected_accuracies:

        difference = abs(test_accuracies[class_idx] - expected_accuracies[class_idx])

        assert difference < tolerance
