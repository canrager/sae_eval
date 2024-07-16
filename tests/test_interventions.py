import pickle

import experiments.bib_intervention as bib_intervention
import experiments.utils as utils


def compare_dicts_within_tolerance(actual, expected, tolerance, path=""):
    """
    Recursively compare two nested dictionaries and assert that all numeric values
    are within the specified tolerance.

    :param actual: The actual dictionary of results
    :param expected: The expected dictionary of results
    :param tolerance: The allowed tolerance for floating point comparisons
    :param path: The current path in the nested structure (used for error messages)
    """
    assert type(actual) == type(
        expected
    ), f"Type mismatch at {path}: {type(actual)} != {type(expected)}"

    if isinstance(actual, dict):
        assert set(actual.keys()) == set(
            expected.keys()
        ), f"Key mismatch at {path}: {set(actual.keys())} != {set(expected.keys())}"
        for key in actual:
            new_path = f"{path}.{key}" if path else str(key)
            compare_dicts_within_tolerance(actual[key], expected[key], tolerance, new_path)
    elif isinstance(actual, (int, float)):
        assert (
            abs(actual - expected) <= tolerance
        ), f"Value mismatch at {path}: {actual} not within {tolerance} of {expected}"
    else:
        assert actual == expected, f"Value mismatch at {path}: {actual} != {expected}"


def test_run_interventions():

    selection_method = bib_intervention.FeatureSelection.above_threshold
    selection_method = bib_intervention.FeatureSelection.top_n

    probe_train_set_size = 5000
    probe_test_set_size = 1000

    # Load datset and probes
    train_set_size = 1000
    test_set_size = 1000
    probe_batch_size = 50
    # llm_batch_size = 250
    llm_batch_size = 10

    # Attribution patching variables
    n_eval_batches = 4
    patching_batch_size = 5

    num_classes = 3

    seed = 42

    top_n_features = [5, 500]

    if selection_method == bib_intervention.FeatureSelection.top_n:
        T_effects = top_n_features
    else:
        raise ValueError("Invalid selection method")

    T_max_sideeffect = 5e-3

    submodule_trainers = {"resid_post_layer_3": {"trainer_ids": [0]}}

    dictionaries_path = "dictionary_learning/dictionaries"

    model_location = "pythia70m"
    sweep_name = "_test_sae"

    probes_dir = "experiments/trained_bib_probes"

    bib_intervention.run_interventions(
        submodule_trainers,
        model_location,
        sweep_name,
        dictionaries_path,
        probes_dir,
        selection_method,
        probe_train_set_size,
        probe_test_set_size,
        train_set_size,
        test_set_size,
        probe_batch_size,
        llm_batch_size,
        n_eval_batches,
        patching_batch_size,
        T_effects,
        T_max_sideeffect,
        num_classes,
        seed,
    )

    ae_group_paths = utils.get_ae_group_paths(
        dictionaries_path, model_location, sweep_name, submodule_trainers
    )
    ae_paths = utils.get_ae_paths(ae_group_paths)

    output_filename = f"{ae_paths[0]}/class_accuracies.pkl"

    with open(output_filename, "rb") as f:
        class_accuracies = pickle.load(f)

    tolerance = 0.01

    expected_results = {
        -1: {0: 0.7620000243186951, 1: 0.7700000405311584, 2: 0.8100000619888306},
        0: {
            5: {0: 0.7220000624656677, 1: 0.8030000329017639, 2: 0.8110000491142273},
            500: {0: 0.5640000104904175, 1: 0.7080000042915344, 2: 0.7730000615119934},
        },
        1: {
            5: {0: 0.7570000290870667, 1: 0.7520000338554382, 2: 0.8070000410079956},
            500: {0: 0.7590000629425049, 1: 0.5130000114440918, 2: 0.7880000472068787},
        },
        2: {
            5: {0: 0.7470000386238098, 1: 0.7800000309944153, 2: 0.8090000152587891},
            500: {0: 0.6980000138282776, 1: 0.6950000524520874, 2: 0.6080000400543213},
        },
    }

    compare_dicts_within_tolerance(class_accuracies, expected_results, tolerance)
