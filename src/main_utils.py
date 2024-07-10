import argparse
import os


def get_slurm_vars(config):
    """
    This function is used to update the args that were provided by the SLURM script. Thus, in the SLURM scipt,
    we provide arguments that determine the current analysis that is run on the cluster (i.e., a specific analysis /
    study / esm sample/ etc) combination. This parameters has to be passed to the python script that runs the
    machine learning analysis. This is done via this function using an ArgumentParser object.

    Args:
        config: Dict, containing the yaml config for default arguments for the parameters

    Returns:
        args: argparse.Namespace, contains the SLURM arguments passed to the script
    """
    # Dictionary of arguments
    args_dict = {
        "--analysis": {
            "default": config["general"]["analysis"],
            "help": "main or suppl",
        },
        "--suppl_type": {
            "default": config["general"]["suppl_type"],
            "help": "e.g., sep_ftf_cmc",
        },
        "--suppl_var": {"default": config["general"]["suppl_var"], "help": "e.g., ftf"},
        "--study": {"default": config["general"]["study"], "help": "ssc or mse"},
        "--esm_sample": {
            "default": config["general"]["esm_sample"],
            "help": "e.g. coco_int",
        },
        "--feature_inclusion_strategy": {
            "default": config["general"]["feature_inclusion_strategy"],
            "help": "e.g., single_items",
        },
        "--model": {
            "default": config["general"]["model"],
            "help": "e.g. linear_baseline_model",
        },
        "--social_interaction_variable": {
            "default": config["general"]["social_interaction_variable"],
            "help": "e.g., social_interaction",
        },
        "--output": {"default": "test_results", "help": "output file path."},
    }

    parser = argparse.ArgumentParser(
        description="ML analysis psychological reactivities OH"
    )

    # Loop through the dictionary and add each argument
    for arg, params in args_dict.items():
        parser.add_argument(
            arg, type=str, default=params["default"], help=params["help"]
        )

    # In the get_slurm_vars function, when adding arguments, specify the type as str2bool for boolean variables
    parser.add_argument(
        "--calc_ia_values",
        type=str2bool,
        default=config["analysis"]["calc_ia_values"],
        help="if for rfr ia_values are calculated, Bool",
    )
    parser.add_argument(
        "--parallelize_reps",
        type=str2bool,
        default=config["analysis"]["parallelize"]["parallelize_reps"],
        help="if we parallelize the analysis across repetitions, Bool",
    )
    parser.add_argument(
        "--parallelize_inner_cv",
        type=str2bool,
        default=config["analysis"]["parallelize"]["parallelize_inner_cv"],
        help="if we parallelize the inner cv of the analysis, Bool",
    )
    parser.add_argument(
        "--parallelize_shap_ia_values",
        type=str2bool,
        default=config["analysis"]["parallelize"]["parallelize_shap_ia_values"],
        help="if we parallelize the shap ia value calculations, Bool",
    )
    parser.add_argument(
        "--parallelize_shap",
        type=str2bool,
        default=config["analysis"]["parallelize"]["parallelize_shap"],
        help="if we parallelize the shap calculations, Bool",
    )
    args = parser.parse_args()
    return args


def str2bool(v):
    """
    Convert a string representation of truth to true (1) or false (0).
    Accepts 'yes', 'true', 't', 'y', '1' as true and 'no', 'false', 'f', 'n', '0' as false.
    Raises ValueError if 'v' is anything else.

    Args:
        v: [bool, str, num], a certain value provied that should be converted to the boolean equivalent

    Returns:
        [v, True, False]: bool, the boolean expression for the given input
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


def update_cfg_with_slurm_vars(cfg, args):
    """
    This function updates the currenct config with the SLURM vars provided so that the machine learning analysis
    can still grab the parameters from the config, but with the updated parameters of the current analysis.
    It is important though that this updated config is used in the main script, not the old config.

    Args:
        cfg: Dict, containg the old YAML config before parameter update

    Returns:
        cfg_updated: Dict, cpontaining the new YAML config with the parameters defined in the SLURM script
    """
    args_dict = vars(args)
    # Loop over the arguments and their values
    for arg_name, arg_value in args_dict.items():
        print(f"Argument {arg_name}: {arg_value}")
        if arg_name in cfg["general"] or arg_name == "output":
            cfg["general"][arg_name] = arg_value
        else:
            if arg_name in cfg["analysis"]:
                cfg["analysis"][arg_name] = arg_value
            elif arg_name in cfg["analysis"]["parallelize"]:
                cfg["analysis"]["parallelize"][arg_name] = arg_value
            else:
                raise ValueError("No matching config entry found")
    return cfg


def allocate_cores_and_update_config(config, total_cores):
    """
    This function allocates a given number of CPUs on the cluster (as defined in the SLURM script) to the different
    task that are computed during the machine learning analysis. This allows different levels of parallelism.
    I tried this to check what yields the highest computation efficiency. How to parallelize is determined
    by the given config.
    Note: Because nested parallelism does not really work with JobLib, currently allocating all CORES to the task
    that run sequentially (i.e., Inner_CV, SHAP value calculations, SHAP IA value calculations), this is rather
    unnecessary and was not used by the final computations.

    Args:
        config: Dict, containing the YAML config
        total_cores: int, number of cores avaiable in the current analysis on the supercomputer cluster

    Returns:
        config, Dict, containg the YAML config where the cores per tasks are included
    """
    print("total_cores in func:", total_cores)

    # One approach to split the cores: Use one per repetition, if enough cores are provided
    if config["analysis"]["parallelize"]["parallelize_reps"]:
        if total_cores >= 10:
            n_jobs_reps = config["analysis"]["cross_validation"]["repetitions"]
        else:
            n_jobs_reps = total_cores
        config["analysis"]["parallelize"]["reps_n_jobs"] = n_jobs_reps
        # Allocate the other cores to what is chosen in the config (shap, rfecv, innercv)
        n_jobs_rest = 3
        if n_jobs_rest < 1:
            n_jobs_rest = None
    else:
        n_jobs_rest = total_cores

    # Distribute remaining cpus equally to given operations
    if config["analysis"]["parallelize"]["parallelize_shap"]:
        print("parallelize shap values")
        config["analysis"]["parallelize"]["shap_n_jobs"] = n_jobs_rest
        print("n_jobs:", config["analysis"]["parallelize"]["shap_n_jobs"])

    if config["analysis"]["parallelize"]["parallelize_shap_outer_cv"]:
        config["analysis"]["parallelize"]["shap_outer_cv_n_jobs"] = n_jobs_rest
    if config["analysis"]["parallelize"]["parallelize_rfe"]:
        config["analysis"]["parallelize"]["rfe_n_jobs"] = n_jobs_rest
    if config["analysis"]["parallelize"]["parallelize_inner_cv"]:
        config["analysis"]["parallelize"]["inner_cv_n_jobs"] = n_jobs_rest
    if config["analysis"]["parallelize"]["parallelize_shap_ia_values"]:
        print("parallelize shap ia values")
        config["analysis"]["parallelize"]["shap_ia_values_n_jobs"] = n_jobs_rest
        print("n_jobs:", config["analysis"]["parallelize"]["shap_ia_values_n_jobs"])

    print("n_jobs rest per repetition:", n_jobs_rest)
    print("reps n_jobs:", config["analysis"]["parallelize"]["reps_n_jobs"])
    print("shap_n_jobs:", config["analysis"]["parallelize"]["shap_n_jobs"])
    print("inner_cv n_jobs:", config["analysis"]["parallelize"]["inner_cv_n_jobs"])
    print(
        "shap ia values n_jobs:",
        config["analysis"]["parallelize"]["shap_ia_values_n_jobs"],
    )
    return config


def sanity_checks_cfg_cluster(config):
    """
    This function sets certain variables automatically when using the cluster which I probably change
    locally during testing. For example, for testing certain analysis settings, I might run the CV
    procedure with num_cv=3, because this does not take too much time locally. If I forget to re-adjust
    this parameter in the config again before runnin the analysis on the cluster, this is done automatically
    in this function.

    Args:
        config: Dict, containing the YAML config

    Returns:
        config: Dict, containing the updated YAML config
    """
    if "store_analysis_results" not in config["analysis"]["machine_learning_methods"]:
        config["analysis"]["machine_learning_methods"].append("store_analysis_results")
    if "store_coefficients" not in config["analysis"]["machine_learning_methods"]:
        config["analysis"]["machine_learning_methods"].append("store_coefficients")
    # for safety, adjust number of reps and outer cvs
    config["analysis"]["cross_validation"]["num_cv"] = 10
    config["analysis"]["cross_validation"]["repetitions"] = 10

    # for safety, adjust the method -> only machine learning is done on the cluster
    config["general"]["preprocessing"] = False
    config["general"]["calc_mlms"] = False
    config["general"]["ml_analysis"] = True
    config["general"]["result_analysis"] = False
    config["general"]["prelim_results_analysis"] = False
    config["general"]["cv_results_plots"] = False
    config["general"]["significance_tests"] = False
    config["general"]["shap_value_analysis"] = False
    return config


def construct_output_path(config):
    """
    This function constructs the path were the results for the current analysis are stored. I used this only
    when I ran analyses locally, otherwise the SLURM script creates the result directory.

    Args:
        config: Dict, containing the YAML config

    Returns:
        output_dir: str, the Path were the results for the current ML analysis are stored
    """
    base_path = config["analysis"]["local_result_dir"]
    analysis = config["general"]["analysis"]
    study = config["general"]["study"]
    esm_sample = config["general"]["esm_sample"]
    feature_inclusion_strategy = config["general"]["feature_inclusion_strategy"]
    model = config["general"]["model"]
    soc_int_var = config["general"]["social_interaction_variable"]

    if analysis == "main":
        paths = ["main", study, esm_sample, feature_inclusion_strategy, model]
        if study == "ssc":
            paths.append(soc_int_var)

    elif analysis == "suppl":
        suppl_analysis = config["general"]["suppl_type"]
        suppl_var = (
            None
            if suppl_analysis == "add_wb_change"
            else config["general"].get("suppl_var", "")
        )
        paths = (
            [suppl_analysis]
            + ([suppl_var] if suppl_var else [])
            + [study, esm_sample, feature_inclusion_strategy, model]
        )

        if study == "ssc" and suppl_analysis not in ["add_wb_change"]:
            paths.append(soc_int_var)

        if suppl_analysis not in [
            "sep_ftf_cmc",
            "sep_pa_na",
            "weighting_by_rel",
            "add_wb_change",
            "mse_no_day_agg",
        ]:
            raise ValueError("Supplementary analysis not implemented")

    else:
        raise ValueError("Study not implemented")

    # Construct the directory path
    output_dir = os.path.join(base_path, *paths)
    return output_dir
