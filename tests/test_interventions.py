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

    dictionaries_path = "dictionary_learning/dictionaries"
    probes_dir = "experiments/trained_bib_probes"

    ae_sweep_paths = {"pythia70m_test_sae": {"resid_post_layer_3": {"trainer_ids": [0]}}}

    for sweep_name, submodule_trainers in ae_sweep_paths.items():

        bib_intervention.run_interventions(
            submodule_trainers,
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
            include_gender=True,
        )

        ae_group_paths = utils.get_ae_group_paths(dictionaries_path, sweep_name, submodule_trainers)
        ae_paths = utils.get_ae_paths(ae_group_paths)

        output_filename = f"{ae_paths[0]}/class_accuracies.pkl"

        with open(output_filename, "rb") as f:
            class_accuracies = pickle.load(f)
        tolerance = 0.02

        print(class_accuracies)

        expected_results = {
            -1: {0: 0.7860000133514404, 1: 0.8670000433921814, 2: 0.843000054359436},
            0: {
                5: {0: 0.7510000467300415, 1: 0.862000048160553, 2: 0.8300000429153442},
                500: {0: 0.5040000081062317, 1: 0.8170000314712524, 2: 0.8040000200271606},
            },
            1: {
                5: {0: 0.7780000567436218, 1: 0.8650000691413879, 2: 0.8340000510215759},
                500: {0: 0.7330000400543213, 1: 0.5920000076293945, 2: 0.8170000314712524},
            },
            2: {
                5: {0: 0.7740000486373901, 1: 0.8610000610351562, 2: 0.8190000653266907},
                500: {0: 0.734000027179718, 1: 0.8070000410079956, 2: 0.581000030040741},
            },
        }

        compare_dicts_within_tolerance(class_accuracies, expected_results, tolerance)
