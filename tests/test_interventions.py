import pickle

import experiments.bib_intervention as bib_intervention
import experiments.utils as utils
from experiments.pipeline_config import PipelineConfig


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


def test_run_interventions_spurious_correlation():
    test_config = PipelineConfig()

    test_config.probe_train_set_size = 4000
    test_config.probe_test_set_size = 1000

    # Load datset and probes
    test_config.train_set_size = 500
    test_config.test_set_size = 500

    seed = 42

    test_config.chosen_class_indices = [
        "male / female",
        "professor / nurse",
        "male_professor / female_nurse",
        "biased_male / biased_female",
    ]

    test_config.attrib_t_effects = [20]

    test_config.dictionaries_path = "dictionary_learning/dictionaries"
    test_config.probes_dir = "experiments/trained_bib_probes"

    ae_sweep_paths = {"pythia70m_test_sae": {"resid_post_layer_3": {"trainer_ids": [0]}}}

    for sweep_name, submodule_trainers in ae_sweep_paths.items():
        bib_intervention.run_interventions(
            submodule_trainers,
            test_config,
            sweep_name,
            seed,
            verbose=True,
        )

        ae_group_paths = utils.get_ae_group_paths(
            test_config.dictionaries_path, sweep_name, submodule_trainers
        )
        ae_paths = utils.get_ae_paths(ae_group_paths)

        output_filename = f"{ae_paths[0]}/class_accuracies_attrib.pkl"

        with open(output_filename, "rb") as f:
            class_accuracies = pickle.load(f)
        tolerance = 0.03

        with open("tests/test_data/class_accuracies_attrib_spurious.pkl", "rb") as f:
            expected_results = pickle.load(f)

        compare_dicts_within_tolerance(class_accuracies, expected_results, tolerance)


def test_run_interventions_tpp():
    test_config = PipelineConfig()

    test_config.probe_train_set_size = 4000
    test_config.probe_test_set_size = 1000

    # Load datset and probes
    test_config.train_set_size = 500
    test_config.test_set_size = 500

    seed = 42

    test_config.chosen_class_indices = [0, 1, 2]

    test_config.attrib_t_effects = [20]

    test_config.dictionaries_path = "dictionary_learning/dictionaries"
    test_config.probes_dir = "experiments/trained_bib_probes"

    ae_sweep_paths = {"pythia70m_test_sae": {"resid_post_layer_3": {"trainer_ids": [0]}}}

    for sweep_name, submodule_trainers in ae_sweep_paths.items():
        bib_intervention.run_interventions(
            submodule_trainers,
            test_config,
            sweep_name,
            seed,
            verbose=True,
        )

        ae_group_paths = utils.get_ae_group_paths(
            test_config.dictionaries_path, sweep_name, submodule_trainers
        )
        ae_paths = utils.get_ae_paths(ae_group_paths)

        output_filename = f"{ae_paths[0]}/class_accuracies_attrib.pkl"

        with open(output_filename, "rb") as f:
            class_accuracies = pickle.load(f)
        tolerance = 0.03

        with open("tests/test_data/class_accuracies_attrib_tpp.pkl", "rb") as f:
            expected_results = pickle.load(f)

        compare_dicts_within_tolerance(class_accuracies, expected_results, tolerance)
