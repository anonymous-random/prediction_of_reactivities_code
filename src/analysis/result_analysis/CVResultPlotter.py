import ast
import io
import json
import os

import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from PIL import Image
from matplotlib import pyplot as plt


class CVResultPlotter:
    """
    The objective of this class is twofold:
    1) It creates a complete table with the prediction results for a given analysis that is stored as an xlsx file.
    2) It plots specific prediction results for a given analysis on a multiplot.
        Which analysis settings are included in a plot are specified in the config. It is possible to
            compare all settings in one analysis (e.g., plot all sample-soc_int_var combinations in main/ssc)
            compare different suppl_vars (e.g., plot certain sample-soc_int_var combinations comparing PA/NA)
            compare different analysis (this is only implemented for the comparison main/mse and add_wb_change)

    Attributes:
        config: YAML config determining certain specifications of the analysis.
        cv_result_dct: Dict, contains the results of the machine learning analysis for a specific analysis setting.
        current_sig_path: str, path were the results of the significance test for the current analysis are stored.
    """

    def __init__(
        self,
        config_path,
    ):
        """
        Constructor method of the BaseMLAnalyzer Class.

        Args:
            config_path: Path to the .YAML config file.
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            self.cv_result_dct = None
        self.current_sig_path = None

    @property
    def sum_results_base_path(self):
        """Data path were the processed results of the machine learning analysis are stored."""
        return self.config["analysis"]["cv_results_plots_tables"]["base_path"]

    @property
    def cvr_cfg(self):
        """Specific config for this the CVResultPlotter class."""
        return self.config["analysis"]["cv_results_plots_tables"]

    @property
    def samples_for_analysis(self):
        """List of ESM-samples used for the current analysis."""
        return self.config["general"]["samples_for_analysis"]

    @property
    def study(self):
        """Study of the analysis, specified in config, either ssc or mse."""
        return self.config["general"]["study"]

    @property
    def analysis(self):
        """Type of analysis, can either be "main" or "suppl"."""
        return self.config["general"]["analysis"]

    @property
    def analysis_type(self):
        """
        This property also differentiates between the main analysis (main) and the supplementary analyses (suppl).
        We need this in addition to self.analysis because of the logic I used to store the results on the cluster.
        "main" was on one hierarchy together with the suppl_type vars.
        """
        return "main" if self.config["general"]["analysis"] == "main" else None

    @property
    def suppl_type(self):
        """Type of supplementary analysis, only defined if self.analysis == 'suppl', e.g. 'sep_ftf_cmc'."""
        if self.analysis == "main":
            return None
        else:
            return self.config["general"]["suppl_type"]

    @property
    def suppl_var(self):
        """
        Var of supplementary analysis, only defined if self.suppl_type exists, e.g. 'ftf'.
        In this class, this property can be a list of two variables, e.g. ['ftf', 'cmc'].
        If so, these two variables will be compared in the plots generated.
        """
        return (
            None
            if self.analysis_type == "main"
            or self.suppl_type == "add_wb_change"
            or isinstance(self.analysis, list)
            else self.config["general"]["suppl_var"]
        )

    @property
    def sum_results_path(self):
        """
        This property is either the specific data_path up to the suppl_type in the folder hierarchy (if one
        analyzes only a certain analysis setting or compares two suppl_var of the same suppl_type) or it
        is just the base path (if one compares two separate analysis settings).
        """
        if isinstance(self.analysis, str):
            path_components = [
                self.sum_results_base_path,
                self.analysis_type,
                self.suppl_type,
            ]
        elif isinstance(self.analysis, list):
            path_components = [
                self.sum_results_base_path,
            ]
            # Filter out empty or None values
        filtered_path_components = [comp for comp in path_components if comp]
        return os.path.normpath(os.path.join(*filtered_path_components))

    @property
    def plot_base_path(self):
        """Base Path were all plots are stored, defined in the config."""
        return self.config["analysis"]["cv_results_plots_tables"]["plot_path"]

    @property
    def plot_path(self):
        """
        Specific plot path for a certain analysis setting. Is as fine-grained as the analysis settings
        that are compared. Two examples:
            If suppl_type is sep_ftf_cmc and ftf and cmc are compared in the plot, this plot
            will be stored in the folder sep_ftf_cmc
            If main and add_wb_change will be compared in the plot, this plot will be stored in
            the first provided variable (the "main" folder, in this case).
        """
        if isinstance(self.analysis, str):
            path_components = [
                self.plot_base_path,
                self.analysis_type,
                self.suppl_type,
            ]
        elif isinstance(self.analysis, list):
            path_components = [
                self.plot_base_path,
                self.suppl_type[0],
            ]
        # Filter out empty or None values
        filtered_path_components = [comp for comp in path_components if comp]
        return os.path.normpath(os.path.join(*filtered_path_components))

    @property
    def sig_results_path(self):
        """Specific path were the significance results for a specific analysis are stored."""
        if isinstance(self.analysis, str):
            path_components = [
                self.cvr_cfg["sig_results_path"],
                self.analysis_type,
                self.suppl_type,
            ]
        elif isinstance(self.analysis, list):
            path_components = [
                self.cvr_cfg["sig_results_path"],
            ]
        # Filter out empty or None values
        filtered_path_components = [comp for comp in path_components if comp]
        return os.path.normpath(os.path.join(*filtered_path_components))

    @property
    def order_dct(self):
        """The order in which the ESM-samples are plotted, defined in the config."""
        return self.cvr_cfg["orderings"]

    @property
    def soc_int_vars(self):
        """
        Set a Dict with esm_samples and corresponding soc_int_vars as attribute if self.study == ssc.
        In comparison to other classes, suppl_var is not defined here, so that social situaion specifics
        depending on the suppl_Var (i.e., the addition of interaction quantity in CoCo UT if ftf) are
        handled differently
        """
        soc_int_var_dct = dict()
        for esm_sample in self.samples_for_analysis:
            soc_int_var_dct[esm_sample] = [
                soc_int_var
                for soc_int_var, vals in self.config["state_data"]["ssc"][
                    "social_interaction_vars"
                ].items()
                if esm_sample in vals["samples"]
            ]
        return soc_int_var_dct

    def apply_methods(self):
        """This function applies the preprocessing methods specified in the config."""
        for method in self.config["analysis"]["cv_results_plots_tables"]["methods"]:
            if method not in dir(CVResultPlotter):
                raise ValueError(f"Method '{method}' is not implemented yet.")
            getattr(self, method)()

    def create_cv_result_dct_wrapper(self):
        """
        A wrapper function for create_cv_result_dct. Depending on the analysis settings, it sets
        the correct path for the function. Its usage enables more flexibility for e.g. creating
        plots that compare different suppl_vars. At the end, it sets the dict containing the
        machine learning-based prediction results as a class attribute.
        """
        dct = {}
        if isinstance(self.analysis, list):
            for suppl_type in self.suppl_type:
                adjusted_path = os.path.join(self.sum_results_path, suppl_type)
                results = self.create_cv_result_dct(result_dict=dct, path=adjusted_path)
        else:
            if isinstance(self.suppl_var, list):  # e.g. ['pa', 'na']
                results = {}
                for suppl_var in self.suppl_var:
                    adjusted_path = os.path.join(self.sum_results_path, suppl_var)
                    results = self.create_cv_result_dct(
                        result_dict=dct, path=adjusted_path
                    )
            elif isinstance(self.suppl_var, str):
                adjusted_path = os.path.join(self.sum_results_path, self.suppl_var)
                results = self.create_cv_result_dct(result_dict=dct, path=adjusted_path)
            else:
                results = self.create_cv_result_dct(
                    result_dict=dct, path=self.sum_results_path
                )
        setattr(self, "cv_result_dct", results)

    def compare_feature_inclusion_strategies(self) -> dict:
        """
        Summarizes metrics (r2, rmse, spearman) for each feature inclusion strategy across
        samples, models, and interaction variables.

        This method:
          - Filters out 'svr' model entirely.
          - Includes 'linearb_baseline_model' only under 'scale_means' strategy.
          - Aggregates metric values from all combinations of sample, model, and interaction variable.
          - Computes mean ('m') and standard deviation ('sd') for each metric per strategy.

        Returns:
            Dict[str, Dict[str, Dict[str, float]]]:
                Top-level keys are feature inclusion strategies,
                next level keys are metrics,
                innermost dict has keys 'm' (mean) and 'sd' (standard deviation).
        """
        # Initialize storage
        results = {}
        for strat in ("feature_selection", "scale_means", "single_items"):
            results[strat] = {metric: [] for metric in ("r2", "rmse", "spearman")}

        # Iterate through the stored cv results
        for sample_dct in self.cv_result_dct["ssc"].values():
            for strat, models_dct in sample_dct.items():
                if strat not in results:
                    continue
                for model_name, interactions_dct in models_dct.items():
                    # Exclude svr entirely, otherwise conclusions may be biased
                    if model_name in ["svr", "linear_baseline_model"]:
                        continue

                    for interaction_dct in interactions_dct.values():
                        # Each interaction_dct has a 'metrics' dict
                        metrics = interaction_dct.get("metrics", {})
                        for metric_name in ("r2", "rmse", "spearman"):
                            value = metrics.get(metric_name, {}).get("mean")
                            if value is None:
                                continue
                            results[strat][metric_name].append(value)

        # Build summary dict
        summary = {}
        for strat, metrics_dct in results.items():
            summary[strat] = {}
            for metric_name, values in metrics_dct.items():
                if values:
                    mean_val = np.mean(values)
                    sd_val = np.std(values) if len(values) > 1 else 0.0
                else:
                    mean_val = "NaN"
                    sd_val = "NaN"
                summary[strat][metric_name] = {"m": mean_val, "sd": sd_val}

        return summary

    def create_cv_result_dct(self, result_dict, path):
        """
        This function walks through all subdirectories of a given root directory and extracts all machine learning
        results from these directories. It puts them in a dictionary that mirrors the directory structure for
        unambiguous assignment.
        Create a nested dict that contains all machine learning results in a given root_dir
        based on the directory structure.

        Args:
            result_dict: Dict, stores the prediction results, mirrors the subdirectory structure
            path: str, root directory from where to start traversing the subdirectories

        Returns:
            current_dct: Dict, contains the prediction results, mirrors the subdirectory structure
        """
        # Traversing through all subdirectories of "path"
        for root, dirs, files in os.walk(path):
            normalized_path = os.path.normpath(root)
            path_parts = normalized_path.split(os.sep)

            # Create nested dictionaries based on the subdirectory structure
            current_level = result_dict
            for part in path_parts:
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]

            # Check for the presence of the 'cv_results.json' file and process it
            if "cv_results.json" in files:
                with open(os.path.join(root, "cv_results.json"), "r") as f:
                    data = json.load(f)
                    metrics_summary = {}
                    for metric, values in data.items():
                        mean_key = "mean_" + metric + "_across_outer_cvs"
                        std_key = "std_" + metric + "_across_outer_cvs"
                        if mean_key in values and std_key in values:
                            metrics_summary[metric] = {
                                "mean": values[mean_key],
                                "std": values[std_key],
                            }
                    current_level["metrics"] = metrics_summary

        # for keeping the rest of the code simpler, we set the same dct structure, independent of the analysis_type
        key_path = self.sum_results_path.split("\\")
        current_dct = result_dict
        for key in key_path:
            if key in current_dct:
                current_dct = current_dct[key]
            else:
                raise KeyError(f"Key '{key}' not found in the dictionary")
        return current_dct

    def create_cv_result_table_wrapper(self):
        """
        This function is a wrapper for the function create_cv_result_table that controls for the
        different dict structures depending on the e.g. the value of suppl_var. This wrapper makes the creation
        of the result tables more flexible.
        """
        if isinstance(self.suppl_var, list):  # e.g. ['pa', 'na']
            for suppl_var in self.suppl_var:
                self.create_cv_result_table(
                    data=self.cv_result_dct[suppl_var].copy(), suppl_var=suppl_var
                )
        elif isinstance(self.suppl_var, str):
            self.create_cv_result_table(
                data=self.cv_result_dct[self.suppl_var].copy(), suppl_var=self.suppl_var
            )
        else:
            self.create_cv_result_table(data=self.cv_result_dct.copy(), suppl_var=None)

    def create_cv_result_table(self, data, suppl_var):
        """
        This function creates a df that contains all numerical cv results for one specific analysis setting.
        More specifically, this table contains the results for all ESM-sample / soc_int_var (if study == ssc) /
        feature inclusion strategy / model combinations. We integrated these variables in the MultiIndex structure.
            Row Indices: SSC: soc_int_vars, fis, metric / MSE: fis, metric
            Col Indices: ESM-sample, model, statistic

        Args:
            data: Dict, containing the prediction results for the analysis specified.
            suppl_var: str, suppl_var of a specific analysis, so that separate tables are created.
        """
        data_for_df = self.transform_dct_data(data)
        for study, study_data in data_for_df.items():
            # Creating the MultiIndex DataFrame and filling it with the data
            row_index = pd.MultiIndex.from_tuples(
                study_data.keys(),
                names=self.cvr_cfg["table"]["row_idx_names"][study],
            )
            col_index = pd.MultiIndex.from_tuples(
                set(key for dct in data_for_df[study].values() for key in dct),
                names=self.cvr_cfg["table"]["col_idx_names"],
            )
            df_final = pd.DataFrame(index=row_index, columns=col_index)
            for row_key, col_data in data_for_df[study].items():
                for col_key, value in col_data.items():
                    df_final.loc[row_key, col_key] = value

            # Reordering the MultiIndex for rows
            if study == "ssc":
                df_final = df_final.reindex(
                    pd.MultiIndex.from_product(
                        [
                            self.order_dct["soc_int_var"],
                            self.order_dct["fis"],
                            self.cvr_cfg["table"]["metrics_to_include"],
                        ],
                        names=df_final.index.names,
                    )
                )
            elif study == "mse":
                df_final = df_final.reindex(
                    pd.MultiIndex.from_product(
                        [
                            self.order_dct["fis"],
                            self.cvr_cfg["table"]["metrics_to_include"],
                        ],
                        names=df_final.index.names,
                    )
                )
            else:
                raise ValueError(f"Study {study} not known")

            # Reordering the MultiIndex for columns
            df_final = df_final.reindex(
                columns=pd.MultiIndex.from_product(
                    [
                        self.order_dct["sample"],
                        self.cvr_cfg["table"]["models_to_include"],
                        self.order_dct["stat"],
                    ],
                    names=df_final.columns.names,
                )
            )
            # Applying the custom format to each cell in the DataFrame
            for metric in df_final.index.get_level_values("Metric").unique():
                if study == "ssc":
                    df_final.loc[(slice(None), slice(None), metric), :] = df_final.loc[
                        (slice(None), slice(None), metric), :
                    ].applymap(lambda x: self.custom_df_formatter(x, metric))
                elif study == "mse":
                    df_final.loc[(slice(None), metric), :] = df_final.loc[
                        (slice(None), metric), :
                    ].applymap(lambda x: self.custom_df_formatter(x, metric))
            df_final.replace("nan", np.nan, inplace=True)

            # Store results
            if self.cvr_cfg["store_table"]:
                if suppl_var:
                    current_plot_path = os.path.join(self.plot_path, suppl_var)
                else:
                    current_plot_path = self.plot_path
                if not os.path.exists(current_plot_path):
                    os.makedirs(current_plot_path)
                filename = os.path.join(current_plot_path, f"{study}_cv_results.xlsx")
                print("store table as ", filename)
                df_final.to_excel(filename, na_rep="", engine="openpyxl")

    @staticmethod
    def transform_dct_data(data):
        """
        This transforms the data from the dictionary to a format that enables creating a MultiIndex DF. Specifically,
        it changes the nesting hierarchy of the dict so that it matches the aimed MultiIndex structure.

        Args:
            data: Dict, containing the prediction results, old nesting hierarchy

        Returns:
            transformed: Dict, containing the prediction results, new nesting hierarchy
        """
        transformed = {"ssc": {}, "mse": {}}
        for study, study_data in data.items():
            for esm_sample, esm_sample_data in study_data.items():
                for (
                    feature_inclusion_strategy,
                    feature_strategy_data,
                ) in esm_sample_data.items():
                    for model, model_data in feature_strategy_data.items():
                        if study == "ssc":
                            for (
                                social_variable,
                                social_variable_data,
                            ) in model_data.items():
                                for metric, metrics_data in social_variable_data[
                                    "metrics"
                                ].items():
                                    for statistic, value in metrics_data.items():
                                        # Tuple for row index
                                        row_key = (
                                            social_variable,
                                            feature_inclusion_strategy,
                                            metric,
                                        )
                                        # Tuple for column index
                                        col_key = (esm_sample, model, statistic.upper())
                                        # Populate the transformed dictionary
                                        transformed["ssc"].setdefault(row_key, {})[
                                            col_key
                                        ] = value
                        elif study == "mse":
                            for metric, metrics_data in model_data["metrics"].items():
                                for statistic, value in metrics_data.items():
                                    # Tuple for row index
                                    row_key = (
                                        feature_inclusion_strategy,
                                        metric,
                                    )
                                    # Tuple for column index
                                    col_key = (esm_sample, model, statistic.upper())
                                    # Populate the transformed dictionary
                                    transformed["mse"].setdefault(row_key, {})[
                                        col_key
                                    ] = value
                        else:
                            raise ValueError("Unknown study")
        return transformed

    @staticmethod
    def custom_df_formatter(x, metric):
        """
        This formats the cells of the dataframe based on the metric according to APA standards.
        More specifically, it removes the leading zero and rounds to 3 decimals.

        Args:
            x: numeric, prediction results for a specific analysis setting (e.g., R2 = 0.0521)
            metric: str, metric of this specific results, in ["r2", "spearman", "rmse"]

        Returns:
            "{:.3f}".format(x).replace("0.", ".").replace("-0.", "-."): formatted result
        """
        if metric in ["r2", "spearman", "rmse"]:
            return "{:.3f}".format(x).replace("0.", ".").replace("-0.", "-.")
        else:
            return x

    def plot_cv_results_wrapper(self):
        """
        This is a wrapper function for creating the CV_Result plots. It sets certain global parameters (e.g.,
        plot styles, colors of axes), and exerts the more specific functions "plot_ssc" and "plot_mse"
        depending on the values of certain class attributes. These possibilities exist
            1) isinstance(self.suppl_var, str): Supplementary analysis and suppl_var is defined. Then we plot
            all settings for this suppl_var (e.g., one plot for all settings of sep_ftf_cmc/ftf)
            2) isinstance(self.suppl_var, list): Supplementary analysis and suppl_var contains two variables
            of the same supplementary analysis. Then we plot one or multiple plots were the different suppl_vars
            are compared. Which settings to include in which plot is defined in the config (e.g., present the
            predictability differences for ftf and cmc for certain soc_int_var-esm sample combinations)
            3) isinstance(self.suppl_type, list): Compare results for different suppl_types or analysis. Only usecase
            at the moment is ['main'; 'add_wb_change']. Then the results for these two are compared in one plot.
            4) everything else: Create one plot for a certain analysis setting where suppl_var is not defined,
            (e.g., plot all esm_sample-soc_int_var combinations for main)
        """
        # Global plot settings
        if self.cvr_cfg["plot"]["seaborn_style"]:
            sns.set_style(self.cvr_cfg["plot"]["seaborn_style"])
        if self.cvr_cfg["plot"]["ggplot_style"]:
            plt.style.use("ggplot")
        plt.rcParams["axes.facecolor"] = self.cvr_cfg["plot"]["axes_facecolor"]
        data = self.cv_result_dct.copy()

        if isinstance(self.suppl_var, str):  # e.g. 'pa''
            for suppl_var, suppl_var_vals in data.items():
                for study, study_vals in suppl_var_vals.items():
                    if self.study != study:
                        continue
                    # Specific path for significance results
                    setattr(
                        self,
                        "current_sig_path",
                        os.path.join(self.sig_results_path, suppl_var, study),
                    )

                    if (
                        self.suppl_type
                        not in self.cvr_cfg["table"]["study_mapping"][study]
                    ):
                        continue
                    for metric in self.cvr_cfg["plot"]["metrics_to_plot"]:
                        if study == "ssc":
                            self.plot_ssc(
                                study_vals=study_vals,
                                metric=metric,
                                suppl_var=suppl_var,
                            )
                        elif study == "mse":
                            self.plot_mse(
                                study_vals=study_vals,
                                metric=metric,
                                suppl_var=suppl_var,
                            )

        if isinstance(self.suppl_var, list):  # e.g. ['pa', 'na']
            for metric in self.cvr_cfg["plot"]["metrics_to_plot"]:
                for study in list(
                    set.intersection(*(set(vals.keys()) for vals in data.values()))
                ):
                    if self.study != study:
                        continue
                    if study == "ssc":
                        self.contrast_suppl_vars_ssc(data, metric)
                    elif study == "mse":
                        self.contrast_suppl_vars_mse(data, metric)

        if isinstance(
            self.suppl_type, list
        ):  # e.g. ['main', 'add_wb_change'], not spotless yet
            for metric in self.cvr_cfg["plot"]["metrics_to_plot"]:
                for study in list(
                    set.intersection(*(set(vals.keys()) for vals in data.values()))
                ):  # only implemented for the comparison of main and add_wb_change, thus, only for MSE
                    if study == "mse":
                        # Use same function that contrasts suppl_vars for contrasting analyses, ok for now
                        self.contrast_suppl_vars_mse(data, metric)

        else:  # e.g. if suppl_var is None, for example in main
            for study, study_vals in data.items():
                if self.study != study:
                    continue
                if hasattr(self, "suppl_var") and self.suppl_var is not None:
                    updated_path = os.path.join(
                        self.sig_results_path, self.suppl_var, study
                    )
                else:
                    updated_path = os.path.join(self.sig_results_path, study)
                setattr(self, "current_sig_path", updated_path)

                if (
                    self.suppl_type is not None
                    and self.suppl_type
                    not in self.cvr_cfg["table"]["study_mapping"][study]
                ):
                    continue
                for metric in self.cvr_cfg["plot"]["metrics_to_plot"]:
                    if study == "ssc":
                        self.plot_ssc(
                            study_vals=study_vals, metric=metric, suppl_var=None
                        )
                    elif study == "mse":
                        self.plot_mse(
                            study_vals=study_vals, metric=metric, suppl_var=None
                        )

    def plot_ssc(self, study_vals, metric, suppl_var=None):
        """
        This function creates the main plot for the ssc plots. It always contains 7 or 8 smaller
        subplots, ESM-sample names as headings, and a common legend for the feature inclusion strategy.
        It exerts the function "prepare_and_plot" that plots the results in the subplots.
        Further, it adjusts the positions of the subplots and the legend when all subplots are filled,
        so that the whole plot looks appealing.

        Args:
            study_vals: Dict, containing the prediction results for a specific analysis setting and study
                (e.g., hierarchy could be coco_int/scale_means/social_interaction/lasso/...)
            metric: str, current metric to plot, list of metrics is defined in the config
            suppl_var: [str,None], suppl_var if defined in the current analysis setting
        """
        # Create and configure main plot, subplots, and headings
        fig = plt.figure(figsize=(22, 12))
        gs = gridspec.GridSpec(
            self.cvr_cfg["plot"]["num_rows"]["ssc"] + 1,
            self.cvr_cfg["plot"]["num_cols"]["ssc"],
            height_ratios=[0.001, 0.999, 0.999],
        )
        headings = {
            "CoCo International": slice(0, 2),
            "Emotions": 2,
            "CoCo UT": 3,
        }
        for title, cols in headings.items():
            heading_ax = fig.add_subplot(gs[0, cols])
            heading_ax.set_title(title, fontsize=18, weight="bold")
            heading_ax.axis("off")

        current_soc_int_vars = getattr(self, "soc_int_vars")
        if suppl_var in ["ftf", "ftf_pa"]:
            current_soc_int_vars["coco_ut"].append("interaction_quantity")

        # Create ordered copy of self.soc_int_vars
        soc_int_vars_ordered = {
            sample: current_soc_int_vars[sample]
            for sample in self.cvr_cfg["orderings"]["sample"]
            if sample in current_soc_int_vars
        }
        axes = {}
        for esm_sample, esm_sample_vars in soc_int_vars_ordered.items():
            for soc_int_var in esm_sample_vars:
                # Get the right grid position for the current esm_sample-soc_int_var combination
                grid_position = (
                    self.cvr_cfg["plot"]["grid_positions"]["ssc"]
                    .get(esm_sample, {})
                    .get(soc_int_var)
                )
                if grid_position:
                    ax = fig.add_subplot(gs[tuple(grid_position)])
                    axes[(esm_sample, soc_int_var)] = ax
                else:
                    raise ValueError("grid position not available on Figure")
                esm_sample_vals = study_vals[esm_sample]
                # Plot data in the subplots
                self.prepare_and_plot(
                    study="ssc",
                    metric=metric,
                    esm_sample_vals=esm_sample_vals,
                    ax=ax,
                    esm_sample=esm_sample,
                    soc_int_var=soc_int_var,
                )
                # Set the title for each row
                title_soc_int_var = soc_int_var.replace("_", " ").title()
                ax.set_title(
                    title_soc_int_var,
                    fontsize=13,
                    fontweight="bold",
                    loc="center",
                    pad=15,
                )
        # Set subplot adjustments
        plt.subplots_adjust(hspace=0.5, wspace=0.4)
        legend_grid_position = self.cvr_cfg["plot"]["legend"]["ax"]["ssc"]
        ax_legend = fig.add_subplot(
            gs[legend_grid_position[0], legend_grid_position[1]]
        )
        if self.suppl_var not in ["ftf", "ftf_pa"]:
            self.set_legend(ax_legend=ax_legend, study="ssc")
        else:
            ax_legend.axis("off")

        # Store or display plots
        if self.config["analysis"]["cv_results_plots_tables"]["store_plot"]:
            self.store_plot(study="ssc", metric=metric, suppl_var=suppl_var)
            plt.close(fig)
        else:
            plt.show()
            self.check_plot_grayscale(fig=fig, filename_raw=None, show_plot=True)

    def contrast_suppl_vars_ssc(self, data, metric):
        """
        This function creates the main plots for the ssc plots if different suppl_vars are compared in the plots.
        It always contains 8 smaller subplots (so that 4 esm_sample-soc_int_var combinations can be compared
        for both suppl_vars, e.g. pa and na), ESM-sample names as headings, and a common legend
        for the feature inclusion strategy. For SSC, there will be more than one main plot.
        It exerts the function "prepare_and_plot" that fills the subplots.
        Further, it adjusts the positions of the subplots and the legend when all subplots are filled,
        so that the whole plot looks appealing.

        Args:
            data: Dict, containing the prediction results for a specific analysis setting and study
                (e.g., hierarchy could be coco_int/scale_means/social_interaction/lasso/...)
            metric: str, current metric to plot, list of metrics is defined in the config
        """
        soc_int_vars_ordered = {
            sample: self.soc_int_vars[sample]
            for sample in self.cvr_cfg["orderings"]["sample"]
            if sample in self.soc_int_vars
        }
        plot_mapping = {
            0: ["coco_int"],
            1: ["emotions", "coco_ut"],
            2: ["coco_int", "emotions"],
        }

        # Create and configure main plot, subplots, and headings for all main plots
        for num_plot, current_samples in plot_mapping.items():
            fig = plt.figure(figsize=(22, 12))
            gs = gridspec.GridSpec(
                self.cvr_cfg["plot"]["num_rows"]["ssc"] + 1,
                self.cvr_cfg["plot"]["num_cols"]["ssc"],
                height_ratios=[0.001, 0.999, 0.999],
            )
            current_heading_mapping = self.cvr_cfg["contrast_suppl_vars"][
                "heading_mapping"
            ]["ssc"][self.suppl_type][num_plot]
            headings = {
                item["sample"]: slice(*item["grid_slice"])
                for item in current_heading_mapping
            }
            for title, cols in headings.items():
                heading_ax = fig.add_subplot(gs[0, cols])
                heading_ax.set_title(title, fontsize=18, weight="bold")
                heading_ax.axis("off")

            # Define which analysis settings are compared in which main plot
            current_vars = {
                sample: vars
                for sample, vars in soc_int_vars_ordered.items()
                if sample in current_samples
            }
            if num_plot == 2:
                current_vars = {
                    sample: vals[:2] for sample, vals in current_vars.items()
                }

            axes = {}
            for esm_sample, vals in current_vars.items():
                for num, suppl_var in enumerate(data.keys()):
                    # Set specific path for significance test results
                    setattr(
                        self,
                        "current_sig_path",
                        os.path.join(self.sig_results_path, suppl_var, "ssc"),
                    )
                    # Add Interaction Quantity for FTF in CoCo UT, if it is not already there
                    if esm_sample == "coco_ut" and suppl_var == "ftf":
                        vals.append("interaction_quantity")
                    elif esm_sample == "coco_ut" and suppl_var == "cmc":
                        try:
                            vals.remove("interaction_quantity")
                        except:
                            continue

                    for soc_int_var in vals:
                        # Get the right grid positions for the current subplot data
                        grid_position = None
                        if num_plot in [0, 1]:
                            grid_position = (
                                self.cvr_cfg["contrast_suppl_vars"]["grid_positions"][
                                    "ssc"
                                ]
                                .get(esm_sample, {})
                                .get(soc_int_var, {})[num]
                            )
                        elif num_plot == 2:
                            grid_position = (
                                self.cvr_cfg["contrast_suppl_vars"]["grid_positions"][
                                    "ssc_coco_int_emotions"
                                ]
                                .get(esm_sample, {})
                                .get(soc_int_var, {})[num]
                            )
                        if grid_position:
                            ax = fig.add_subplot(gs[tuple(grid_position)])
                            axes[(esm_sample, soc_int_var)] = ax
                        else:
                            raise ValueError("grid position not available on Figure")
                        esm_sample_vals = data[suppl_var]["ssc"][esm_sample]
                        # Plot data in the subplots
                        self.prepare_and_plot(
                            study="ssc",
                            metric=metric,
                            esm_sample_vals=esm_sample_vals,
                            ax=ax,
                            esm_sample=esm_sample,
                            soc_int_var=soc_int_var,
                        )
                        # Display cmc as cm in the plots
                        if suppl_var == 'cmc':
                            suppl_var_plot = 'cm'
                        else:
                            suppl_var_plot = suppl_var

                        # Set the title for each row
                        title_soc_int_var = (
                            soc_int_var.replace("_", " ").title()
                            + " - "
                            + suppl_var_plot.upper()
                        )
                        ax.set_title(
                            title_soc_int_var,
                            fontsize=13,
                            fontweight="bold",
                            loc="center",
                            pad=15,
                        )
            # Set subplot adjustments
            plt.subplots_adjust(hspace=0.5, wspace=0.4)
            legend_grid_position = self.cvr_cfg["contrast_suppl_vars"][
                "legend_position"
            ]["ssc"][num_plot]
            ax_legend = fig.add_subplot(
                gs[legend_grid_position[0], legend_grid_position[1]]
            )
            ax_legend = self.adjust_legend_position(
                ax_legend=ax_legend,
                study="ssc",
                num_plot=num_plot,
            )
            self.set_legend(ax_legend=ax_legend, study="ssc")

            # Store or present plots
            if self.config["analysis"]["cv_results_plots_tables"]["store_plot"]:
                suppl_var_1, suppl_var_2 = list(data.keys())[:2]
                self.store_plot(
                    study="ssc",
                    metric=metric,
                    suppl_var=(suppl_var_1, suppl_var_2),
                    num_plot=num_plot,
                )
                plt.close(fig)
            else:
                plt.show()
                self.check_plot_grayscale(fig=fig, filename_raw=None, show_plot=True)

    def set_legend(self, ax_legend, study):
        """
        This function places a common legend in the outer figure of the cv result plot.

        Args:
            ax_legend: matplotlib.axes.Axes.legend, Legend object that is changed through this function.
            study: str, either mse or ssc, influences the position of the legend in the plot.
        """
        legend_lst = []
        for fis in self.order_dct["fis"]:
            legend = plt.Line2D(
                [0],
                [0],
                linestyle="none",
                marker=self.cvr_cfg["plot"][fis]["fmt"],
                color=self.cvr_cfg["plot"][fis]["color"],
                label=self.cvr_cfg["plot"][fis]["label"],
            )
            legend_lst.append(legend)
        ax_legend.axis("off")
        ax_legend.legend(
            handles=legend_lst,
            bbox_to_anchor=self.cvr_cfg["plot"]["legend"]["position"][study],
            fontsize=12,
            frameon=True,
            fancybox=True,
            handlelength=2,
            handleheight=2,
        )
        legend_frame = ax_legend.get_legend().get_frame()
        legend_frame.set_edgecolor("black")
        legend_frame.set_linewidth(1)
        legend_frame.set_boxstyle("square, pad=0.5")

    def adjust_legend_position(
        self, ax_legend, study, num_plot=None, compare_analyses=False
    ):
        """
        This function adjusts the position of the legend based on the config specifications and some manual processing.

        Args:
            ax_legend: matplotlib.axes.Axes.legend, Legend object that is changed through this function.
            study: str, either mse or ssc, influences the needed position adjustments.
            num_plot: [int, None], needed for custom position adjustments, e.g., if the number of subplots differ
                between num_plots
            compare_analyses: bool, if analysis are compared (e.g., pa/na), or not (e.g., only pa)

        Returns: ax_legend: Specific position of the legend
        """
        # Custom adjustments if certain analyses are compared (e.g., suppl_vars)
        if compare_analyses:
            height_adjustment = self.cvr_cfg["contrast_suppl_vars"][
                "legend_adjustments"
            ]["mse_add_wb_change"]["height"]
            width_adjustment = self.cvr_cfg["contrast_suppl_vars"][
                "legend_adjustments"
            ]["mse_add_wb_change"]["width"]
            pos = ax_legend.get_position()
            new_pos = [
                pos.x0 + width_adjustment,
                pos.y0 + height_adjustment,
                pos.width,
                pos.height,
            ]
            ax_legend.set_position(new_pos)
        # Custom adjustments if num_plot is given
        if num_plot in [0, 2]:
            height_adjustment = self.cvr_cfg["contrast_suppl_vars"][
                "legend_adjustments"
            ]["ssc_plots_0_2"]
            pos = ax_legend.get_position()
            new_pos = [pos.x0, pos.y0 + height_adjustment, pos.width, pos.height]
            ax_legend.set_position(new_pos)
        # Custom adjustments for the setting specified
        if study == "mse" or self.suppl_type == "sep_pa_na" and num_plot != 0:
            width_adjustment = self.cvr_cfg["contrast_suppl_vars"][
                "legend_adjustments"
            ]["other"]
            pos = ax_legend.get_position()
            new_pos = [pos.x0 + width_adjustment, pos.y0, pos.width, pos.height]
            ax_legend.set_position(new_pos)
        return ax_legend

    def plot_mse(self, study_vals, metric, suppl_var=None):
        """
        This function creates the main plot for the mse plots. It always contains 3 smaller subplots,
        ESM-sample / event names as headings, and a common legend for the feature inclusion strategy.
        It exerts the function "prepare_and_plot" that plots the results in the subplots.
        Further, it adjusts the positions of the subplots and the legend when all subplots are filled,
        so that the whole plot looks appealing.

        Args:
            study_vals: Dict, containing the prediction results for a specific analysis setting and study
                (e.g., hierarchy could be coco_int/scale_means/social_interaction/lasso/...)
            metric: str, current metric to plot, list of metrics is defined in the config
            suppl_var: [str,None], suppl_var if defined in the current analysis setting
        """
        # Create and configure main plot, subplots, and headings for all main plots
        fig = plt.figure(figsize=(16, 4))
        gs = gridspec.GridSpec(
            self.cvr_cfg["plot"]["num_rows"]["mse"] + 1,
            self.cvr_cfg["plot"]["num_cols"]["mse"],
            height_ratios=[0.001, 0.999],
        )
        headings = {
            "CoCo International": 0,
            "Emotions": 1,
            "CoCo UT": 2,
        }
        for title, cols in headings.items():
            heading_ax = fig.add_subplot(gs[0, cols])
            heading_ax.set_title(title, fontsize=15, weight="bold")
            heading_ax.axis("off")

        axes = {}
        for col, esm_sample in enumerate(self.cvr_cfg["orderings"]["sample"]):
            ax = fig.add_subplot(gs[1, col])
            axes[esm_sample] = ax
            esm_sample_vals = study_vals[esm_sample]
            # Plot data in the subplots
            self.prepare_and_plot(
                study="mse",
                metric=metric,
                esm_sample_vals=esm_sample_vals,
                ax=ax,
                esm_sample=esm_sample,
            )
            # Set the title for each row
            title = self.cvr_cfg["plot"]["title_mapping"]["mse"][esm_sample]
            ax.set_title(
                title,
                fontsize=11,
                fontweight="bold",
                loc="center",
                pad=15,
            )
        # Set subplot adjustments
        plt.subplots_adjust(hspace=0.65, wspace=0.4)
        legend_grid_position = self.cvr_cfg["plot"]["legend"]["ax"]["mse"]
        ax_legend = fig.add_subplot(
            gs[legend_grid_position[0], legend_grid_position[1]]
        )
        self.set_legend(ax_legend=ax_legend, study="mse")

        # Store or present plots
        if self.config["analysis"]["cv_results_plots_tables"]["store_plot"]:
            self.store_plot(study="mse", metric=metric, suppl_var=suppl_var)
            plt.close(fig)
        else:
            plt.show()
            self.check_plot_grayscale(fig=fig, filename_raw=None, show_plot=True)

    def contrast_suppl_vars_mse(self, data, metric):
        """
        This function creates the main plot for contrasting mse plots. It always contains 6 smaller subplots
        (so that the three events can be compared on different suppl_vars or for different analysis),
        ESM-sample / event names as headings, and a common legend for the feature inclusion strategy.
        It exerts the function "prepare_and_plot" that plots the results in the subplots.
        Further, it adjusts the positions of the subplots and the legend when all subplots are filled,
        so that the whole plot looks appealing.

        Args:
            data: Dict, containing the prediction results for a specific analysis setting and study
                (e.g., hierarchy could be coco_int/scale_means/social_interaction/lasso/...)
            metric: str, current metric to plot, list of metrics is defined in the config
        """
        # Create and configure main plot, subplots, and headings for all main plots
        fig = plt.figure(figsize=(22, 12))  # Adjust the figure size as needed
        gs = gridspec.GridSpec(
            self.cvr_cfg["plot"]["num_rows"]["mse"] + 2,
            self.cvr_cfg["plot"]["num_cols"]["mse"],
            height_ratios=[0.001, 0.999, 0.999],
        )
        headings = self.cvr_cfg["contrast_suppl_vars"]["heading_mapping"]["mse"]
        for title, cols in headings.items():
            heading_ax = fig.add_subplot(gs[0, cols])
            heading_ax.set_title(title, fontsize=18, weight="bold")
            heading_ax.axis("off")

        axes = {}
        for col, esm_sample in enumerate(self.cvr_cfg["orderings"]["sample"], 1):
            for num, suppl_var in enumerate(data.keys()):
                # adjusted significance path
                setattr(
                    self,
                    "current_sig_path",
                    os.path.join(self.sig_results_path, suppl_var, "mse"),
                )
                ax = fig.add_subplot(gs[num + 1, col])
                axes[esm_sample] = ax
                esm_sample_vals = data[suppl_var]["mse"][esm_sample]
                # Plot data in the subplots
                self.prepare_and_plot(
                    study="mse",
                    metric=metric,
                    esm_sample_vals=esm_sample_vals,
                    ax=ax,
                    esm_sample=esm_sample,
                )
                # Set the title for each row
                maj_soc_event = self.cvr_cfg["plot"]["title_mapping"]["mse"][esm_sample]
                if isinstance(self.suppl_type, list):
                    title = maj_soc_event
                else:
                    title = maj_soc_event + " - " + suppl_var.upper()
                ax.set_title(
                    title,
                    fontsize=13,
                    fontweight="bold",
                    loc="center",
                    pad=15,
                )
        # Set subplot adjustments
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        legend_grid_position = self.cvr_cfg["contrast_suppl_vars"]["legend_position"][
            "mse"
        ]
        ax_legend = fig.add_subplot(
            gs[legend_grid_position[0], legend_grid_position[1]]
        )
        if isinstance(self.suppl_type, list):
            compare_analyses = True
        else:
            compare_analyses = False
        ax_legend = self.adjust_legend_position(
            ax_legend=ax_legend,
            study="mse",
            compare_analyses=compare_analyses,
        )
        self.set_legend(ax_legend=ax_legend, study="mse")
        # Add a row description in the left column for comparing main and add_wb_change
        ax_upper = fig.add_subplot(gs[1, 0])
        ax_upper.text(
            0.7,
            0.8,
            "Main analysis",
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=20,
        )
        ax_upper.axis("off")
        ax_lower = fig.add_subplot(gs[2, 0])
        ax_lower.text(
            0.7,
            0.8,
            "Supplementary \nanalysis: Add the \ninitial change in \nwell-being",
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=20,
        )
        ax_lower.axis("off")

        # Store or present results
        if self.config["analysis"]["cv_results_plots_tables"]["store_plot"]:
            suppl_var_1, suppl_var_2 = list(data.keys())[:2]
            self.store_plot(
                study="mse",
                metric=metric,
                suppl_var=(suppl_var_1, suppl_var_2),
                num_plot=None,
            )
            plt.close(fig)
        else:
            plt.show()
            self.check_plot_grayscale(fig=fig, filename_raw=None, show_plot=True)

    def prepare_and_plot(
        self, study, metric, esm_sample_vals, ax, esm_sample, soc_int_var=None
    ):
        """
        This function is used to plot the prediction results for a specific analysis setting in a specific
        subplot (given by the ax parameter). It is exerted everytime if a subplot is filled in a main plot.
        It gets the specific plot data (this step is study-specific) and then runs "plot_cv_graphs"

        Args:
            study: str, either ssc or mse
            metric: str, in ['r2', 'rmse', 'spearman']
            esm_sample_vals: Dict, containing the prediction results for a specific analysis setting, study, and
                esm sample (e.g., hierarchy could be scale_means/social_interaction/lasso/...)
            ax: matplotlib.ax Object, specific position on the main plot where to plot the current data
            esm_sample: str, specific esm-sample for the current subplot
            soc_int_var: [str, None]: specific soc_int_var for the current subplot (if ssc)
        """
        sorted_models = self.cvr_cfg["orderings"]["model"]
        sorted_models_axis_names = self.cvr_cfg["orderings"]["model_for_axis"]
        if soc_int_var:  # ssc
            result_dct = self.get_cv_plot_data(
                sorted_models, metric, esm_sample_vals, soc_int_var
            )
        else:  # mse
            result_dct = self.get_cv_plot_data(sorted_models, metric, esm_sample_vals)
        # This actually plots the data
        self.plot_cv_graphs(
            result_dct=result_dct,
            study=study,
            metric=metric,
            sorted_models=sorted_models,
            sorted_models_axis_names=sorted_models_axis_names,
            ax=ax,
            esm_sample=esm_sample,
            soc_int_var=soc_int_var,
        )

    def get_cv_plot_data(self, sorted_models, metric, sample_vals, soc_int_var=None):
        """
        This function reorders the results for a specific analysis settings so that plotting them is easy.

        Args:
            sorted_models: lst of strings, ML-models in the order that will be displayed in the plots
            metric: str, in ['r2', 'rmse', 'spearman']
            sample_vals: Dict, containing the prediction results for a specific analysis setting, study, and
                esm sample (e.g., hierarchy could be scale_means/social_interaction/lasso/...)
            soc_int_var: [str, None]: specific soc_int_var for the current subplot (if ssc)

        Returns:
            results: Dict, containing the results optimal for plotting as indicated in the structure below
        """
        results = {
            "mean_single_items": [],
            "std_single_items": [],
            "mean_scale_means": [],
            "std_scale_means": [],
            "mean_feature_selection": [],
            "std_feature_selection": [],
        }
        for model in sorted_models:
            if (
                self.suppl_type == "weighting_by_rel"
                and "feature_selection" in self.cvr_cfg["orderings"]["fis"]
            ):
                self.cvr_cfg["orderings"]["fis"].remove("feature_selection")
            for key in self.cvr_cfg["orderings"]["fis"]:
                item = sample_vals[key].get(model, {})
                if soc_int_var:
                    item = item.get(soc_int_var, {})
                metrics = item.get("metrics", {}).get(metric, {})
                results[f"mean_{key}"].append(metrics.get("mean"))
                results[f"std_{key}"].append(metrics.get("std"))
        return results

    def plot_cv_graphs(
        self,
        result_dct,
        study,
        metric,
        sorted_models,
        sorted_models_axis_names,
        ax,
        esm_sample,
        soc_int_var=None,
    ):
        """
        This function actually plots the prediction results for a specific analysis setting (esm_sample, metric,
        soc_int_var [if ssc]) in the given ax object.

        Args:
            result_dct: Dict, containing the prediction results for the current analysis setting.
            study: str, either ssc or mse
            metric: str, in ['r2', 'rmse', 'spearman']
            sorted_models: lst of str, sorted models for the plot, as they are represented in the code (e.g., lasso)
            sorted_models_axis_names: lst of str, str representations of sorted_models for the paper (e.g., LASSO)
            ax: matplotlib.ax Object, specific position on the main plot where to plot the current data
            esm_sample: str, specific esm-sample for the current subplot
            soc_int_var: [str, None]: specific soc_int_var for the current subplot (if ssc)
        """
        # Plotting values and error bars for each configuration on the associated X-positions
        x_positions = range(len(sorted_models))
        for x in x_positions:
            for fis in self.cvr_cfg["orderings"]["fis"]:
                if fis != "scale_means" and sorted_models[x] == "linear_baseline_model":
                    continue
                ax.errorbar(
                    x=x + self.cvr_cfg["plot"][fis]["shift"],
                    y=result_dct[f"mean_{fis}"][x],
                    yerr=result_dct[f"std_{fis}"][x],
                    fmt=self.cvr_cfg["plot"][fis]["fmt"],
                    color=self.cvr_cfg["plot"][fis]["color"],
                    elinewidth=1.5,
                    capsize=5,
                )
        # Formatting and labeling
        ax.set_xticks([x for x in x_positions])
        ax.set_xticklabels(
            sorted_models_axis_names, fontsize=12, weight="bold", rotation=0
        )
        ax.set_xlim(self.cvr_cfg["plot"]["x_lim"])
        # Significance tests were only calculated for R²
        if metric == "r2":
            combinations = self.get_significance_results(
                esm_sample=esm_sample, soc_int_var=soc_int_var
            )
            self.show_significance(
                combinations=combinations, ax=ax, study=study, result_dct=result_dct
            )
            plt.rcParams["mathtext.fontset"] = "custom"
            plt.rcParams["mathtext.bf"] = "cm:italic:bold"
            ax.set_ylabel(r"$\mathbf{R}^2$", fontsize=13)
            # Format y-range and y-ticks
            y_range = self.get_r2_ylim(study=study)
            ax.set_ylim(y_range)
            if study == "ssc":
                negative_yticks = np.arange(0, y_range[0], -0.03)[
                    1:
                ]  # Skip the first to avoid duplicating 0
                positive_yticks = np.arange(0, y_range[1], 0.03)
                yticks = np.sort(np.concatenate((negative_yticks, positive_yticks)))
                ax.set_yticks(yticks)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(self.custom_formatter))
        elif metric == "rmse":
            # Currently not used
            if self.cvr_cfg["plot"]["metrics"]["rmse"]["log_scale"]:
                ax.set_yscale("log")
            ax.set_ylabel("RMSE", fontsize=13, weight="bold")
            ax.set_ylim(self.cvr_cfg["plot"]["metrics"]["rmse"]["y_lim"][study])
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        else:  # Spearman's rho
            plt.rcParams["mathtext.fontset"] = "custom"
            plt.rcParams["mathtext.bf"] = "cm:italic:bold"
            ax.set_ylabel(r"$\mathbf{\rho}$", fontsize=13)
            ax.set_ylim(self.cvr_cfg["plot"]["metrics"]["rho"]["y_lim"][study])
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(self.custom_formatter))

        # Only show left and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['top'].set_visible(False)

        # Optional: set their color if needed
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')

    def get_r2_ylim(self, study):
        """
        This function returns the y-lim that fits best for a certain setting based on the config.

        Args:
            study: str, either ssc or mse

        Returns:
            ylim: lst, contains the optimal minimum and maximum y value for the current setting.
        """
        ylim_r2_cfg = self.cvr_cfg["plot"]["metrics"]["r2"]["y_lim"][study]
        if self.analysis == "main":
            ylim = ylim_r2_cfg["main"]
        elif self.analysis == "suppl":
            if self.suppl_type == "add_wb_change":
                ylim = ylim_r2_cfg["suppl"][self.suppl_type]
            else:
                if isinstance(self.suppl_var, list):
                    y_lim_map_var = "_".join(self.suppl_var)
                else:
                    y_lim_map_var = self.suppl_var
                ylim = ylim_r2_cfg["suppl"][self.suppl_type][y_lim_map_var]
        elif "add_wb_change" in self.suppl_type:
            ylim = ylim_r2_cfg["suppl"]["add_wb_change"]
        else:
            raise ValueError("Check analysis/suppl_type/suppl_var combinations")
        return ylim

    def get_significance_results(self, esm_sample, soc_int_var=None):
        """
        This function gets the significance test results for the current analysis setting from the given path and
        returns the significance test results for one subplot (thus, for the 3 model comparisons that are shown
        in the plot) as a nested list. Results for SVR are excluded, because its result are not presented in the plot.

        Args:
            esm_sample: str, specific esm-sample for the current subplot
            soc_int_var: [str, None]: specific soc_int_var for the current subplot (if ssc)

        Returns:
            sig_nested_lst: List containing the results of the pairwise significance tests between the models
                in the following format: [[(0, 2), 0.01], [(0, 1), 0.0011]]. Thus, the first tuple denotes the
                models x-positions (e.g., 0,2) and the associated float (e.g., 0.01) is the FDR-adjusted p-value
                of this comparison. The model-x_position assignment is: 0: LBM, 1: LASSO, 2: RFR
        """
        # Load the significance test results
        with open(
            os.path.join(self.current_sig_path, "significance_results.json"), "r"
        ) as f:
            sig_results_dct = json.load(f)
        if not soc_int_var:
            soc_int_var = "default"
        spec_sig_results = sig_results_dct[esm_sample][soc_int_var]
        spec_sig_results = {
            key: (value if value == "<.001" else float(value))
            for key, value in spec_sig_results.items()
        }
        # Check if all available model comparisons should be plotted
        filtered_sig_results = {
            key: value
            for key, value in spec_sig_results.items()
            if all(
                elem in self.cvr_cfg["plot"]["models_to_plot"]
                for elem in ast.literal_eval(key)
            )
        }
        model_x_axis_mapping = {
            model: num for num, model in enumerate(self.cvr_cfg["orderings"]["model"])
        }
        # Create the list containing all relevant comparisons and its p-value
        sig_nested_lst = [
            [(model_x_axis_mapping[elem], model_x_axis_mapping[elem2]), value]
            for key, value in filtered_sig_results.items()
            for elem, elem2 in [ast.literal_eval(key)]
        ]
        return sig_nested_lst

    def show_significance(self, combinations, study, ax, result_dct):
        """
        This function plots brackets that denote significance in the current subplots. The horizontal position is given
        by the model comparisons, the vertical position is calculated based on the highest object in the plot.

        Args:
            combinations: List containing the results of the pairwise significance tests between the models
                in the following format: [[(0, 2), 0.01], [(0, 1), 0.0011]]. Thus, the first tuple denotes the
                models x-positions (e.g., 0,2) and the associated float (e.g., 0.01) is the FDR-adjusted p-value
                of this comparison. The model-x_position assignment is: 0: LBM, 1: LASSO, 2: RFR
            study: str, either ssc or mse
            ax: matplotlib.ax Object, specifies in which subplots the significance annotations are plotted
            result_dct: Dict, contains the prediction results for the current setting
        """
        # Get max value from the current data
        bottom, top = ax.get_ylim()
        yrange = top - bottom
        highest_point = self.find_max_point(result_dct)
        # introduce some margin
        highest_point += 0.005
        # Show only significant differences
        significant_combinations = [
            val
            for val in combinations
            if (val[1] == "<.001" or (isinstance(val[1], float) and val[1] <= 0.05))
        ]
        if significant_combinations:
            # Plot significance brackets
            for i, significant_combination in enumerate(significant_combinations):
                x1 = significant_combination[0][0]
                x2 = significant_combination[0][1]
                level = len(significant_combinations) - i
                bar_height = highest_point + (
                    level * self.cvr_cfg["plot"]["sign_height_ratio"][study]
                )
                bar_tips = bar_height - (yrange * 0.02)
                plt.plot(
                    [x1, x1, x2, x2],
                    [bar_tips, bar_height, bar_height, bar_tips],
                    lw=1,
                    c="k",
                )
                # Significance levels and associated symbols
                p = significant_combination[1]

                if isinstance(p, float):
                    sig_symbol = f"*$\\it{{p}}$ = {p:.3f}".lstrip("0")
                else:
                    sig_symbol = f"*$\\it{{p}}$ {p}"

                text_height = bar_height + 0.002  # 0.0001
                plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha="center", c="k")

    @staticmethod
    def find_max_point(data_dict):
        """
        Find function identifies the highest point in a certain plot so that the height of the significance brackets
        can be adjusted accordingly. The highest point would be the upper bracket of the error bar of the most positive
        mean-sd combination in the current plot.

        Args:
            data_dict: Dict, containing the prediction results of the current analysis

        Returns:
            max_point: float, the highest point in the current plot as a numeric value
        """
        max_point = 0
        for key in data_dict:
            if "mean" in key:
                std_key = key.replace("mean", "std")
                if std_key in data_dict:  # Check if the std_key exists
                    for mean, std in zip(data_dict[key], data_dict[std_key]):
                        if mean is not None and std is not None:
                            max_point = max(max_point, mean + std)
        return max_point

    @staticmethod
    def custom_formatter(x, pos=None):
        """
        Formats the y ticks according to APA style (no leading zeros for values between -1 and 1) and two decimals
        for the y-ticks of the metrics on the y-axis of the plots.

        Args:
            x: float, a certain numeric value representing the prediction results for a specific metric
            pos: None, dummy value for compatibility

        Returns:
            formatted: str, x adequately formatted and returned as a string
        """
        if -1 < x < 1:
            formatted = f"{x:.2f}".replace("0.", ".").replace("-0.", "-.")
        else:
            formatted = f"{x:.2f}"
        return formatted

    def store_plot(self, study, metric, suppl_var=None, num_plot=None):
        """
        This function is used to store the plots. It creates the specific plot paths were the plots are stored,
        creates the filename and stores the plots in a sufficiently high resolution for clear display in the paper
        (600 DPI). It stores the plot in color and in grayscale (because of the printed version of journals).

        Args:
            study: str, either ssc or mse
            metric: str, in ['r2', 'rmse', 'spearman']
            suppl_var: [None, str, lst], Can be a) None, b) a string representing the current suppl_var of
                the analysis or c) a tuple-like (e.g., list) containing two suppl_vars that are contrasted in a plot
        """
        if isinstance(suppl_var, str):
            current_plot_path = os.path.join(self.plot_path, suppl_var, study)
        else:  # None or tuple
            current_plot_path = os.path.join(self.plot_path, study)
        if not os.path.exists(current_plot_path):
            os.makedirs(current_plot_path)
        if isinstance(suppl_var, tuple):
            filename_raw = os.path.join(
                current_plot_path,
                f"{suppl_var[0]}_{suppl_var[1]}_{metric}_plot_{num_plot}",
            )
        else:  # None or string
            filename_raw = os.path.join(current_plot_path, f"{metric}_plot")
        if self.cvr_cfg["plot"]["filetype"] == "jpeg":
            filename = filename_raw + ".jpg"
            plt.savefig(filename, bbox_inches="tight", dpi=600)
            self.check_plot_grayscale(
                filename_raw=filename_raw, fig=None, show_plot=False, data_ending=".jpg"
            )

    @staticmethod
    def check_plot_grayscale(
        fig=None, filename_raw=None, show_plot=True, data_ending=".jpg"
    ):
        """
        This function checks how a certain plot looks if it would be printed in grayscale. It plots or
        stores the grayscale plot in the same dir where the colored plot is stored.

        Args:
            fig: matlotlib.figure object
            filename_raw: the filename of the plot to be stored without the filetype-specific ending
            show_plot: bool, determines if the plot should be presented or stored
            data_ending: str, determines the filetype for storing, e.g., ".jpg"
        """
        if show_plot:
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            img = Image.open(buf).convert("L")
        else:
            img = Image.open(filename_raw + data_ending).convert("L")
        dpi = 600
        figsize = (img.width / dpi, img.height / dpi)
        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(np.array(img), cmap="gray")
        plt.axis("off")
        # Store or present plot
        if filename_raw:
            new_filename = f"{filename_raw}_grayscale" + data_ending
            plt.savefig(new_filename)
        if show_plot:
            plt.show()
