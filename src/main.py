"""
This is the main script that runs all analyses in this repository.
To determine which specific analysis are being performed, please adjust lines 8-16 in "config_refactored-yaml".
Analyses to execute are:
  preprocessing
  calc_mlms
  ml_analysis
  result_analysis
  prelim_results_analysis
  significance_tests
  cv_results_plots
  shap_value_analysis
  osf_suppl_analysis
Analyses that are set to "True" will be performed when this script is executed. Order matters (e.g., before one can
run the machin learning analysis, data needed to be preprocessed and the criterion (random effects) needed to be
computed). For a description of the specific analyses, inspect the Class docstrings.
In principle, one could run multiple analyses with just one call of the main function. Though, I do not recommend this,
because the abstraction level of different Classes differ for computational reasons.
For example, the preprocessing is done for all ESM-samples and feature inclusion strategies together (if specified),
whereas the machine learning-based analysis is only done for one specific ESM-sample / fis / model combination
for computational reasons.
Therefore, I recommend to first preprocess the data for all analysis settings, than run extract the random effects
for all analysis settings, than run the machine learning analysis, etc ...
"""

import os
import sys

# Some sanity checks that guarantee that the code run from different configurations
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
src_root = os.path.abspath(os.path.join(project_root, 'src'))
sys.path.insert(0, project_root)
sys.path.insert(0, src_root)

import yaml

from src.analysis.machine_learning.LassoAnalyzer import LassoAnalyzer
from src.analysis.machine_learning.LinearBaselineAnalyzer import LinearBaselineAnalyzer
from src.analysis.machine_learning.RFRAnalyzer import RFRAnalyzer
from src.analysis.machine_learning.SVRAnalyzer import SVRAnalyzer
from src.analysis.multilevel_modeling.MultilevelModeling import MultilevelModeling
from src.analysis.result_analysis.CVResultPlotter import CVResultPlotter
from src.analysis.result_analysis.PrelimResultAnalyzer import PrelimResultAnalyzer
from src.analysis.result_analysis.ResultAnalyzer import ResultAnalyzer
from src.analysis.result_analysis.ShapValueAnalyzer import ShapValueAnalyzer
from src.analysis.result_analysis.SignificanceTesting import SignificanceTesting
from src.analysis.result_analysis.result_utils import supplementary_result_structurer
from src.main_utils import (
    get_slurm_vars,
    allocate_cores_and_update_config,
    update_cfg_with_slurm_vars,
    construct_output_path,
    sanity_checks_cfg_cluster,
)
from src.preprocessing.PreprocessorSSC import PreprocessorSSC
from src.preprocessing.helper_functions import load_data, preprocess_country_data

if __name__ == "__main__":
    # Some sanity checks that guarantee that the code run from different configurations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    src_root = os.path.abspath(os.path.join(project_root, 'src'))
    sys.path.insert(0, project_root)
    sys.path.insert(0, src_root)

    config_path = "../configs/config_refactor.yaml"
    feature_mapping_path = "../configs/feature_name_mapping.yaml"
    country_data_config_path = "../configs/config_country_data.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config["general"]["preprocessing"]:
        df_dct = load_data(config_path=config_path, nrows=None)
        preprocess_country_data(config_path=country_data_config_path)
        if config["general"]["study"] == "ssc":
            preprocessor_ssc = PreprocessorSSC(
                config_path=config_path, sample_dct=df_dct
            )
            preprocessor_ssc.apply_preprocessing_methods()

    if config["general"]["calc_mlms"]:
        mlm_class = MultilevelModeling(config_path=config_path)
        mlm_class.apply_preprocessing_methods()

    if config["general"]["ml_analysis"]:
        args = get_slurm_vars(config)
        updated_config = update_cfg_with_slurm_vars(cfg=config, args=args)

        if os.getenv("SLURM_JOB_ID"):
            updated_config = sanity_checks_cfg_cluster(updated_config)
            total_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
            print("total_cores in main:", total_cores)
            updated_config = allocate_cores_and_update_config(
                updated_config, total_cores
            )
            output_dir = args.output
        else:
            output_dir = construct_output_path(updated_config)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        if updated_config["general"]["model"] == "linear_baseline_model":
            linear_baseline_analyzer = LinearBaselineAnalyzer(
                config=updated_config, output_dir=output_dir
            )
            linear_baseline_analyzer.apply_methods()
        elif updated_config["general"]["model"] == "lasso":
            lasso_analyzer = LassoAnalyzer(config=updated_config, output_dir=output_dir)
            lasso_analyzer.apply_methods()
        elif updated_config["general"]["model"] == "rfr":
            rfr_analyzer = RFRAnalyzer(config=updated_config, output_dir=output_dir)
            rfr_analyzer.apply_methods()
        elif updated_config["general"]["model"] == "svr":
            svr_analyzer = SVRAnalyzer(config=updated_config, output_dir=output_dir)
            svr_analyzer.apply_methods()
        else:
            raise ValueError("Model not implemented")

    if config["general"]["result_analysis"]:
        for sample in ["coco_int"]: # config["general"]["samples_for_analysis"]:
            result_analyzer = ResultAnalyzer(config_path=config_path, esm_sample=sample)
            result_analyzer.apply_methods()

    if config["general"]["prelim_results_analysis"]:
        for sample in config["general"]["samples_for_analysis"]:
            prelim_result_analyzer = PrelimResultAnalyzer(
                config_path=config_path, esm_sample=sample
            )
            prelim_result_analyzer.apply_methods()

    if config["general"]["significance_tests"]:
        cv_result_plotter = SignificanceTesting(config_path=config_path)
        cv_result_plotter.apply_methods()

    if config["general"]["cv_results_plots"]:
        cv_result_plotter = CVResultPlotter(config_path=config_path)
        cv_result_plotter.apply_methods()

    if config["general"]["shap_value_analysis"]:
        shap_value_analyzer = ShapValueAnalyzer(
            config_path=config_path,
            feature_mapping_path=feature_mapping_path,
        )
        shap_value_analyzer.apply_methods()

    if config["general"]["osf_suppl_analysis"]:
        supplementary_result_structurer(
            root_dir="../results/ml_results_processed",
            filename="lin_model_coefficients.json",
            output_dir="../results/osf_suppl_results/lasso_coefs",
        )
