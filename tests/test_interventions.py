import pickle
import os

import experiments.bib_intervention as bib_intervention
import experiments.utils as utils
from experiments.pipeline_config import PipelineConfig
import experiments.llm_autointerp.llm_utils as llm_utils

# Very dangerous to set this to True, as it will overwrite the expected results
FORCE_UPDATE_EXPECTED_RESULTS = False

# Set this to False for faster testing, but risk being misled by old probes
FORCE_RECOMPUTE_PROBES = True


def compare_dicts_within_tolerance(actual, expected, tolerance, path="", all_diffs=None):
    """
    Recursively compare two nested dictionaries and assert that all numeric values
    are within the specified tolerance. Print global mean and max difference at root call.

    :param actual: The actual dictionary of results
    :param expected: The expected dictionary of results
    :param tolerance: The allowed tolerance for floating point comparisons
    :param path: The current path in the nested structure (used for error messages)
    :param all_diffs: List to collect all differences (used internally for recursion)
    """
    if all_diffs is None:
        all_diffs = []

    assert type(actual) == type(
        expected
    ), f"Type mismatch at {path}: {type(actual)} != {type(expected)}"

    if isinstance(actual, dict):
        assert set(actual.keys()) == set(
            expected.keys()
        ), f"Key mismatch at {path}: {set(actual.keys())} != {set(expected.keys())}"
        for key in actual:
            new_path = f"{path}.{key}" if path else str(key)

            if key == "acc_0" or key == "acc_1" or key == "loss":
                continue

            compare_dicts_within_tolerance(
                actual[key], expected[key], tolerance, new_path, all_diffs
            )
    elif isinstance(actual, (int, float)):
        diff = abs(actual - expected)
        all_diffs.append(diff)
    else:
        assert actual == expected, f"Value mismatch at {path}: {actual} != {expected}"

    # Print global mean and max difference only at the root call
    if path == "":
        if all_diffs:
            mean_diff = sum(all_diffs) / len(all_diffs)
            max_diff = max(all_diffs)

            print(f"Global mean difference: {mean_diff}")
            print(f"Global max difference: {max_diff}")

            assert (
                max_diff <= tolerance
            ), f"Value mismatch at {path}: {actual} not within {tolerance} of {expected}"

        else:
            print("No numeric differences found.")


def test_run_interventions_spurious_correlation_same_layer():
    test_config = PipelineConfig()

    test_config.use_autointerp = False
    test_config.force_node_effects_recompute = True
    test_config.force_ablations_recompute = True
    test_config.force_probe_recompute = FORCE_RECOMPUTE_PROBES
    test_config.probe_layer = "sae_layer"

    test_config.spurious_corr = True

    test_config.probe_train_set_size = 4000
    test_config.probe_test_set_size = 1000

    # Load datset and probes
    test_config.train_set_size = 4000
    test_config.test_set_size = 1000

    seed = 42

    test_config.chosen_class_indices = [
        "male / female",
        "professor / nurse",
        "male_professor / female_nurse",
        "biased_male / biased_female",
    ]

    test_config.attrib_t_effects = [20]

    test_config.dictionaries_path = "dictionary_learning/dictionaries"
    test_config.probes_dir = "experiments/test_trained_bib_probes"

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

        if FORCE_UPDATE_EXPECTED_RESULTS:
            with open("tests/test_data/class_accuracies_attrib_spurious_same_layer.pkl", "wb") as f:
                pickle.dump(class_accuracies, f)

        with open("tests/test_data/class_accuracies_attrib_spurious_same_layer.pkl", "rb") as f:
            expected_results = pickle.load(f)

        compare_dicts_within_tolerance(class_accuracies, expected_results, tolerance)


def test_run_interventions_spurious_correlation():
    test_config = PipelineConfig()

    test_config.use_autointerp = False
    test_config.force_node_effects_recompute = True
    test_config.force_ablations_recompute = True
    test_config.force_probe_recompute = FORCE_RECOMPUTE_PROBES

    test_config.spurious_corr = True

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
    test_config.probes_dir = "experiments/test_trained_bib_probes"

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

        if FORCE_UPDATE_EXPECTED_RESULTS:
            with open("tests/test_data/class_accuracies_attrib_spurious.pkl", "wb") as f:
                pickle.dump(class_accuracies, f)

        with open("tests/test_data/class_accuracies_attrib_spurious.pkl", "rb") as f:
            expected_results = pickle.load(f)

        compare_dicts_within_tolerance(class_accuracies, expected_results, tolerance)


def test_run_interventions_tpp():
    test_config = PipelineConfig()

    test_config.use_autointerp = False
    test_config.force_node_effects_recompute = True
    test_config.force_ablations_recompute = True
    test_config.force_probe_recompute = FORCE_RECOMPUTE_PROBES

    test_config.spurious_corr = False

    test_config.probe_train_set_size = 4000
    test_config.probe_test_set_size = 1000

    # Load datset and probes
    test_config.train_set_size = 500
    test_config.test_set_size = 500

    seed = 42

    test_config.chosen_class_indices = [0, 1, 2]

    test_config.attrib_t_effects = [20]

    test_config.dictionaries_path = "dictionary_learning/dictionaries"
    test_config.probes_dir = "experiments/test_trained_bib_probes"

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

        if FORCE_UPDATE_EXPECTED_RESULTS:
            with open("tests/test_data/class_accuracies_attrib_tpp.pkl", "wb") as f:
                pickle.dump(class_accuracies, f)

        with open("tests/test_data/class_accuracies_attrib_tpp.pkl", "rb") as f:
            expected_results = pickle.load(f)

        compare_dicts_within_tolerance(class_accuracies, expected_results, tolerance)


def test_run_interventions_spurious_correlation_multiple_groupings():
    test_config = PipelineConfig()

    test_config.use_autointerp = False
    test_config.force_node_effects_recompute = True
    test_config.force_ablations_recompute = True
    test_config.force_probe_recompute = FORCE_RECOMPUTE_PROBES

    test_config.spurious_corr = True

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
    test_config.probes_dir = "experiments/test_trained_bib_probes"

    ae_sweep_paths = {"pythia70m_test_sae": {"resid_post_layer_3": {"trainer_ids": [0]}}}

    column1_vals_list = [("professor", "nurse"), ("filmmaker", "dentist")]
    column2_vals = ("male", "female")

    for i, column1_vals in enumerate(column1_vals_list):
        print(f"Running with column1_vals: {column1_vals}")

        test_config.column1_vals = column1_vals
        test_config.column2_vals = column2_vals

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

            if FORCE_UPDATE_EXPECTED_RESULTS:
                with open(
                    f"tests/test_data/class_accuracies_attrib_spurious_{column1_vals[0]}_{column1_vals[1]}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(class_accuracies, f)

            with open(
                f"tests/test_data/class_accuracies_attrib_spurious_{column1_vals[0]}_{column1_vals[1]}.pkl",
                "rb",
            ) as f:
                expected_results = pickle.load(f)

            compare_dicts_within_tolerance(class_accuracies, expected_results, tolerance)

            other_professions = column1_vals_list[1 - i]

            with open(
                f"tests/test_data/class_accuracies_attrib_spurious_{other_professions[0]}_{other_professions[1]}.pkl",
                "rb",
            ) as f:
                other_profession_expected_results = pickle.load(f)

            print(
                f"\nComparing results for {column1_vals} with expected results for {other_professions}:"
            )

            try:
                compare_dicts_within_tolerance(
                    class_accuracies, other_profession_expected_results, tolerance
                )
                print(
                    f"Test failed: Results for {column1_vals} are unexpectedly similar to expected results for {other_professions}."
                )
                raise AssertionError(
                    f"Results for {column1_vals} are unexpectedly similar to expected results for {other_professions}."
                )
            except AssertionError:
                print(
                    f"Test passed: Results for {column1_vals} are different from expected results for {other_professions}, as expected."
                )


# NOTE: This will use ~5k API tokens.
def test_run_interventions_spurious_correlation_autointerp():
    test_config = PipelineConfig()

    llm_utils.set_api_key(test_config.api_llm, "")

    test_config.use_autointerp = True
    test_config.force_node_effects_recompute = True
    test_config.force_ablations_recompute = True
    test_config.force_probe_recompute = FORCE_RECOMPUTE_PROBES

    test_config.prompt_dir = "experiments/llm_autointerp/"
    test_config.force_autointerp_recompute = True

    test_config.num_top_features_per_class = 5

    test_config.spurious_corr = True

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

    test_config.autointerp_t_effects = [5]

    test_config.attrib_t_effects = []

    test_config.dictionaries_path = "dictionary_learning/dictionaries"
    test_config.probes_dir = "experiments/test_trained_bib_probes"

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
        tolerance = 0.03

        filenames = [
            "class_accuracies_auto_interp_spurious.pkl",
            "class_accuracies_bias_shift_dir1_spurious.pkl",
            "class_accuracies_bias_shift_dir2_spurious.pkl",
        ]

        for filename in filenames:
            output_filename = filename.replace("_spurious", "")
            output_filename = f"{ae_paths[0]}/{output_filename}"

            with open(output_filename, "rb") as f:
                class_accuracies = pickle.load(f)

            if FORCE_UPDATE_EXPECTED_RESULTS:
                with open(f"tests/test_data/{filename}", "wb") as f:
                    pickle.dump(class_accuracies, f)

            with open(f"tests/test_data/{filename}", "rb") as f:
                expected_results = pickle.load(f)

            compare_dicts_within_tolerance(class_accuracies, expected_results, tolerance)
