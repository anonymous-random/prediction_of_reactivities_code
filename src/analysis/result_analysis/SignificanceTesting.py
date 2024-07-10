import json
import os
from itertools import combinations
from math import sqrt
from statistics import stdev

import numpy as np
import pandas as pd
import yaml
from scipy.stats import t
from statsmodels.stats.multitest import fdrcorrection


class SignificanceTesting:
    """
    This class computes test of significance to compare the prediction results for different models
    across different analysis settings. Results for different feature selection strategies were pooled.
        Thus, in Study 1 (ssc / main analysis), 6 comparisons (pairwise comparisons of models) are computed
        for each ESM sample - soc_int_var combination, resulting in 42 statistical tests.
        In Study 2, 6 comparisons are computed for each event, resulting in 18 statistical tests.
    Due to multiple testing, tests of significance are False-Discovery-Rate corrected.
    Results are stored as a table for the supplementary results and as a JSON that is used by the CVResultPlotter
    to include the results of the significance tests as annotations in the CV result plots.

    Attributes:
        config: YAML config determining certain specifications of the analysis.
        result_dct: Dict, the predictions results are loaded from its folders and stored in this Dict.
        fis_aggregated_results: Dict,
        significance_results: Dict,
    """

    def __init__(
        self,
        config_path,
    ):
        """
        Constructor method of the SignificanceTesting Class.

        Args:
            config_path: Path to the .YAML config file.
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.result_dct = None
        self.fis_aggregated_results = None
        self.significance_results = None

    @property
    def sig_test_config(self):
        """This is used to extract the relevant part of the config."""
        return self.config["analysis"]["significance_tests"]

    @property
    def result_base_path(self):
        """Base Path where all CV Results of all analyses are stored"""
        return os.path.normpath(self.sig_test_config["result_base_path"])

    @property
    def analysis(self):
        """Type of analysis, can either be "main" or "suppl"."""
        return self.config["general"]["analysis"]

    @property
    def suppl_type(self):
        """Type of supplementary analysis, only defined if self.analysis == 'suppl', e.g. 'sep_ftf_cmc'."""
        return (
            self.config["general"]["suppl_type"] if self.analysis == "suppl" else None
        )

    @property
    def analysis_level_path(self):
        """
        Used for loading the requested data efficiently. This is not equivalent to 'analysis', because
        for the supplementary analysis 'weighting_by_rel', we use the same data as an in main analysis up to
        the Multilevel Modeling and 'main' is on the same hierarchical level as the suppl_types.
        """
        return (
            None
            if self.suppl_type
            in ["sep_ftf_cmc", "sep_pa_na", "add_wb_change", "weighting_by_rel"]
            else "main"
        )

    @property
    def suppl_var(self):
        """Var of supplementary analysis, only defined if self.suppl_type exists, e.g. 'ftf'."""
        return (
            self.config["general"]["suppl_var"]
            if self.suppl_type and self.suppl_type != "add_wb_change"
            else None
        )

    @property
    def study(self):
        """Study, either "ssc" or "mse"."""
        return self.config["general"]["study"]

    @property
    def esm_samples(self):
        """List of esm-samples."""
        return self.config["general"]["samples_for_analysis"]

    @property
    def metric(self):
        """CV Result Metric used for the significance tests."""
        return self.sig_test_config["metric"]

    @property
    def soc_int_vars(self):
        """Get the soc_int_vars for the specified sample if the study is 'ssc'."""
        if self.study == "ssc":
            dct = dict()
            for esm_sample in self.esm_samples:
                dct[esm_sample] = [
                    soc_int_var
                    for soc_int_var, values in self.config["state_data"]["ssc"][
                        "social_interaction_vars"
                    ].items()
                    if esm_sample in values["samples"]
                ]
            return dct
        else:
            return None

    @property
    def data_path_study(self):
        """
        Creates the data_path for the significance test calculations according to the study, analysis and ESM-sample
        up to the degree of the correct study. An example would be results/ml_results_processed/sep_ftf_cmc/ftf/ssc.
        path components that are None will be filtered out, so that the correct path depending on the specific
        analysis is returned.
        """
        path_components = [
            self.result_base_path,
            self.analysis_level_path,
            self.suppl_type,
            self.suppl_var,
            self.study,
        ]
        filtered_path_components = [comp for comp in path_components if comp]
        return os.path.normpath(os.path.join(*filtered_path_components))

    @property
    def store_base_path(self):
        """Base Path were all significance tests of all analyses are stored."""
        return self.sig_test_config["store_base_path"]

    @property
    def store_path(self):
        """Specific store_path for the current configuration of analysis_type, suppl_type, etc."""
        path_components = [
            self.store_base_path,
            self.analysis_level_path,
            self.suppl_type,
            self.suppl_var,
            self.study,
        ]
        filtered_path_components = [comp for comp in path_components if comp]
        return os.path.normpath(os.path.join(*filtered_path_components))

    def apply_methods(self):
        """This function applies the preprocessing methods specified in the config."""
        for method in self.sig_test_config["methods"]:
            if method not in dir(SignificanceTesting):
                raise ValueError(f"Method '{method}' is not implemented yet.")
            getattr(self, method)()

    def get_result_data(self):
        """
        This method loads the predictions results of the specified analysis and study for all esm samples.
        It does so by using recursing the given root directory, extracting the relevant files from the
        terminal directories and mirroring the folder structure in the Result Dict.
        """
        result_dct = dict()
        for root, dirs, files in os.walk(self.data_path_study):
            if "cv_results.json" in files:
                # Get the relative path from data_path_study
                relative_path = os.path.relpath(root, self.data_path_study)
                path_components = relative_path.split(os.sep)
                current_level = result_dct
                for component in path_components:
                    if component not in current_level:
                        current_level[component] = {}
                    current_level = current_level[component]
                file_path = os.path.join(root, "cv_results.json")
                current_level[f"cv_results_{self.metric}"] = self.load_cv_results(
                    file_path=file_path, metric=self.metric
                )
        setattr(self, "result_dct", result_dct)

    def load_cv_results(self, file_path, metric):
        """
        Loads the cv results for a specified metric and checks if the given metric exists.

        Args:
            file_path: str, path to the file containing the prediction results (up to "/.../cv_results.json")
            metric: str, name of a certain metric, must have been specified in the config for the ML-analysis

        Returns:
            {rep: results[metric] for rep, results in data.items()}: Dict, containing the number of repetitions
                of the 10x10x10 CV as keys, and the values obtained of the given metric in each outer fold of
                one rep as values.
                So if rep=3 and num_outer_cv = 3, the Dict structure would look like this:
                {0: [a,b,c], 1: [d,e,f], 2: [g,h,i]}
        """
        assert (
            metric in self.config["analysis"]["scoring_metric"]["outer_cv_loop"]
        ), f"metric {metric} not calculated"
        with open(file_path, "r") as file:
            data = json.load(file)
            return {rep: results[metric] for rep, results in data.items()}

    def summarize_results_across_fis(self):
        """
        This method is a wrapper for aggregating the raw prediction results across feature selection strategies.
        We do this, because to limit the number of statistical tests, we compare only models, no feature
        selection strategies. This is valid, because the number of samples is identical across feature
        selection strategies."""
        aggregated_results = dict()
        results = self.result_dct.copy()

        for esm_sample in results:
            aggregated_results[esm_sample] = {}
            for fis in results[esm_sample]:
                for model in results[esm_sample][fis]:
                    if model not in aggregated_results[esm_sample]:
                        aggregated_results[esm_sample][model] = {}

                    # This sets a variable "default" in mse to mirror the ssc dict hierarchy
                    if self.study == "ssc":
                        for soc_int_var, data in results[esm_sample][fis][
                            model
                        ].items():
                            self.process_cv_results(
                                data, aggregated_results[esm_sample][model], soc_int_var
                            )
                    elif self.study == "mse":
                        data = results[esm_sample][fis][model]
                        self.process_cv_results(
                            data, aggregated_results[esm_sample][model]
                        )

            self.calculate_means(aggregated_results[esm_sample])

        setattr(self, "fis_aggregated_results", aggregated_results)

    def process_cv_results(self, data, target_dict, soc_int_var=None):
        """
        This function collects the results obtained for different feature inclusion strategies for
        the metric the significance tests are based on.

        Args:
            data: Dict, contains the prediction results for a given analysis setting.
            target_dict: Dict, will be filled with the results
            soc_int_var: [None, str], only defined if study==ssc, another step in the folder hierarchy
        """
        key = soc_int_var if soc_int_var else "default"
        if key not in target_dict:
            target_dict[key] = {
                rep: {"sum": [], "count": 0}
                for rep in data[f"cv_results_{self.metric}"]
            }

        for rep, values in data[f"cv_results_{self.metric}"].items():
            if not target_dict[key][rep][
                "sum"
            ]:  # If the list is empty, initialize it with zeros
                target_dict[key][rep]["sum"] = [0] * len(values)
            # Sum values for each index in the list
            for i, val in enumerate(values):
                target_dict[key][rep]["sum"][i] += val
            target_dict[key][rep]["count"] += 1

    @staticmethod
    def calculate_means(aggregated_results):
        """Compute the mean of the values of a metric for each repetition in a given dictionary."""
        for model in aggregated_results:
            for soc_int_var in aggregated_results[model]:
                for rep in aggregated_results[model][soc_int_var]:
                    sum_values = aggregated_results[model][soc_int_var][rep]["sum"]
                    count = aggregated_results[model][soc_int_var][rep]["count"]
                    aggregated_results[model][soc_int_var][rep] = [
                        val / count for val in sum_values
                    ]

    def apply_significance_tests(self):
        """
        This is a wrapper method for the calculation of the significance test results. Specifically, it
            Iterates through all comparisons
            Computes paired-dependent t-tests for each comparison
            Applies False Discovery Correction to the obtained p-values
            Creates a tabular-like df containing all p and t values
            Creates a Dict with the p values for usage in the CVResultPlotter class.
        """
        test_training_ratio = 1 / (
            self.config["analysis"]["cross_validation"]["num_cv"] - 1
        )
        significance_results = {}
        model_pairs = list(
            combinations(["linear_baseline_model", "lasso", "rfr", "svr"], 2)
        )
        # Iterate through Dict hierarchy and conduct comparisons (always model_x vs model_y)
        for esm_sample in self.fis_aggregated_results:
            significance_results[esm_sample] = {}
            for soc_int_var in self.fis_aggregated_results[esm_sample]["lasso"]:
                significance_results[esm_sample][soc_int_var] = {}
                for model1, model2 in model_pairs:
                    data1 = [
                        val
                        for sublist in self.fis_aggregated_results[esm_sample][model1][
                            soc_int_var
                        ].values()
                        for val in sublist
                    ]
                    data2 = [
                        val
                        for sublist in self.fis_aggregated_results[esm_sample][model2][
                            soc_int_var
                        ].values()
                        for val in sublist
                    ]
                    t_stat, p = self.corrected_dependent_ttest(
                        data1, data2, test_training_ratio
                    )
                    significance_results[esm_sample][soc_int_var][
                        (model1, model2)
                    ] = dict()
                    significance_results[esm_sample][soc_int_var][(model1, model2)][
                        "p"
                    ] = p
                    significance_results[esm_sample][soc_int_var][(model1, model2)][
                        "t"
                    ] = t_stat

        # FDR correction, create result df, store result_df
        corrected_p_values = self.fdr_correct_p_values(significance_results)
        result_df = self.create_p_value_df(corrected_p_values, significance_results)
        if self.sig_test_config["store_df"]:
            os.makedirs(self.store_path, exist_ok=True)
            file_path = os.path.join(self.store_path, "significance_results.xlsx")
            result_df.to_excel(file_path)

        # Updating the p-values in the significance_results dictionary
        p_val_dct = significance_results.copy()
        for key, new_p in corrected_p_values.items():
            dataset, setting, model_pair = key
            if dataset in p_val_dct and setting in p_val_dct[dataset]:
                if model_pair in p_val_dct[dataset][setting]:
                    p_val_dct[dataset][setting][model_pair] = new_p

        # Store result_json, change model_pair key to strings for unambiguous storing
        if self.sig_test_config["store_json"]:
            p_val_dct = {
                dataset: {
                    setting: {
                        str(model_pair): value
                        for model_pair, value in setting_dict.items()
                    }
                    for setting, setting_dict in dataset_dict.items()
                }
                for dataset, dataset_dict in p_val_dct.items()
            }
            os.makedirs(self.store_path, exist_ok=True)
            file_path = os.path.join(self.store_path, "significance_results.json")
            with open(file_path, "w") as file:
                json.dump(p_val_dct, file, indent=4)

    def fdr_correct_p_values(self, result_dict):
        """
        Correct p-values using False Discovery Rate (FDR) as described by Benjamini & Hochberg (1995)

        Args:
            result_dict: Dict, containing all results for a certain analysis setting (e.g., main/ssc)

        Returns:
              corrected_p_values_dct: Dict, same structure as result_dict, but with corrected p_values
        """
        p_values = []
        labels = []
        for esm_sample, data in result_dict.items():
            for soc_int_var, comparisons in data.items():
                for model_pair, stats in comparisons.items():
                    p_value = stats["p"]
                    p_values.append(p_value)
                    labels.append((esm_sample, soc_int_var, model_pair))
        adjusted_p_values = fdrcorrection(p_values)[1]
        # format the p_values for the table accordingly
        formatted_p_values = self.format_p_values(adjusted_p_values)
        corrected_p_values_dct = {
            label: formatted_p for label, formatted_p in zip(labels, formatted_p_values)
        }
        return corrected_p_values_dct

    def create_p_value_df(self, corrected_p_values, result_dct):
        """
        Create a DataFrame from corrected p-values. This dataframe has a multiindex where
            Row Indices are: soc_int_var, model1
            Column Indices are: esm_sample, model2, state (p and t)

        Args:
            corrected_p_values: Dict, containing the FDR corrected p_values
            result_dct: Dict, containing the raw results of the significance tests (to get the t values)

        Returns:
            df_pivot: df, contains all FDR corrected significance results for a given analysis.
        """
        df_data = []
        for (
            esm_sample,
            soc_int_var,
            model_pair,
        ), adjusted_p in corrected_p_values.items():
            model1, model2 = model_pair
            t_value = result_dct[esm_sample][soc_int_var][model_pair]["t"]
            df_data.append(
                {
                    "esm_sample": esm_sample,
                    "soc_int_var": soc_int_var,
                    "model1": model1,
                    "model2": model2,
                    "stat": "t",
                    "value": t_value,
                }
            )
            df_data.append(
                {
                    "esm_sample": esm_sample,
                    "soc_int_var": soc_int_var,
                    "model1": model1,
                    "model2": model2,
                    "stat": "p",
                    "value": adjusted_p,
                }
            )
        df = pd.DataFrame(df_data)

        soc_int_var_ordering = self.sig_test_config["df_ordering"]["soc_int_vars"][
            self.study
        ]

        df_pivot = (
            df.pivot_table(
                index=["soc_int_var", "model1"],
                columns=["esm_sample", "model2", "stat"],
                values="value",
                aggfunc="first",
            )
            .reindex(soc_int_var_ordering, level="soc_int_var")
            .reindex(self.sig_test_config["df_ordering"]["models"], level="model1")
            .reindex(
                self.sig_test_config["df_ordering"]["models"], level="model2", axis=1
            )
            .reindex(
                self.sig_test_config["df_ordering"]["esm_samples"],
                level="esm_sample",
                axis=1,
            )
            .reindex(self.sig_test_config["df_ordering"]["stats"], level="stat", axis=1)
        )
        return df_pivot

    @staticmethod
    def corrected_dependent_ttest(data1, data2, test_training_ratio=1 / 9):
        """
        Python implementation for the corrected paired t-test as described by Nadeau & Bengio (2003).

        Args:
            data1: list, containing the prediction results for a certain setting (up to a specific model)
            data2: list, containing the prediction results for a another setting (up to a specific model)
            test_training_ratio: float, depends on the number of folds in the outer_cv (i.e., 10 in this setting)

        Returns:
            t_stat: float, t statistic of the comparison of data1 and data2
            p: float, p-value for the comparison of data1 and data2
        """
        n = len(data1)
        differences = [(data1[i] - data2[i]) for i in range(n)]
        sd = stdev(differences)
        divisor = 1 / n * sum(differences)
        denominator = sqrt(1 / n + test_training_ratio) * sd
        t_stat = np.round(divisor / denominator, 2)
        df = n - 1  # degrees of freedom
        p = np.round((1.0 - t.cdf(abs(t_stat), df)) * 2.0, 4)  # p value
        return t_stat, p

    @staticmethod
    def format_p_values(lst_of_p_vals):
        """
        This function formats the p_values according to APA standards (3 decimals, <.001 otherwise)

        Args:
            lst_of_p_vals: list, containing the p_values for a given analysis setting

        Returns:
            formatted_p_vals: list, contains p_values formatted according to APA standards
        """
        formatted_p_vals = []
        for p_val in lst_of_p_vals:
            if p_val < 0.001:
                formatted_p_vals.append("<.001")
            else:
                formatted = "{:.3f}".format(p_val).lstrip("0")
                formatted_p_vals.append(formatted)
        return formatted_p_vals
