import io
import itertools
import json
import os
import pickle
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import yaml
from PIL import Image
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler


class ShapValueAnalyzer:
    """
    This class is used to generate insights from the SHAP values and the SHAP interaction values.
    Specifically, it does the following:
        Summarizing the SHAP values on different levels of abstractions
            One-hot encoded features (these are the "raw" SHAP values)
            Single features (e.g., aggregating raw SHAP values from the same categorical variable)
            Scale Means (e.g., aggregating raw SHAP values that build a dimension)
            Broad Categories (i.e., socio-demographics, personality, political and societal, country-level)
        Creates plots using SHAP values
            SHAP summary plots for all specific analysis settings (e.g., one plot for one soc_int_var including
                RFR and LASSO and all feature inclusion strategies)
            SHAP importance plots across broad feature categories (1 per analysis setting)
        Creates plots using SHAP interaction values (only for RFR)
            Scatter plot showing the relationship between main and interactive effects across specific settings
                (1 per analysis setting)
            Heatmap plot comparing specific feature interactions and main effects for a specific analysis (e.g.,
            one plot for one soc_int_var and one feature inclusion strategy)
    Plots are formatted for presentation in the paper and in the supplement using a mapping between the feature
    names in the code and the feature names in the plot (feature_name_mapping.yaml) and stored according to
    Journal standards.

    Attributes:
        config: YAML config determining certain specifications of the analysis.
        feature_mapping: YAML config mapping the feature names in the code to names used for presentation.
        results: Dict, Used for keeping the raw SHAP values in memory.
        shap_abstraction_levels: Dict, here we store the SHAP values summarized across abstraction levels.
        col_order_before_scaling: Dct, containing the column order of the df containing the features before the
            custom scaler was applied, because applying this scaler changed the column order. The columns
            depend on the esm_sample, the feature inclusion strategy, and potentially the soc_int_var, therefore
            these parameters are mirrored in the Dict structure.
        col_order_after_scaling: Dct, containing the column order of the df containing the features before the
            custom scaler was applied, because applying this scaler changed the column order. The columns
            depend on the esm_sample, the feature inclusion strategy, and potentially the soc_int_var, therefore
            these parameters are mirrored in the Dict structure.
        col_order_mapping: Dict, mapping between col_order_before_scaling and col_order_after_scaling on the level
            of individual columns
    """

    def __init__(self, config_path, feature_mapping_path):
        """
        Constructor method of the ShapValueAnalyzer Class.

        Args:
            config_path: Path to the .YAML config file.
            feature_mapping_path: Path to th .YAML feature mapping file.
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        with open(feature_mapping_path, "r") as f:
            self.feature_mapping = yaml.safe_load(f)
        self.results = dict()
        self.shap_abstraction_levels = dict()
        self.col_order_before_scaling = {}
        self.col_order_after_scaling = {}
        self.col_order_mapping = {}

    @property
    def shap_config(self):
        """Part of the config with configurations for the shap summaries / shap plots."""
        return self.config["analysis"]["shap_value_analysis"]

    @property
    def result_base_path(self):
        """Data path of the raw results of the machine learning analysis."""
        return os.path.normpath(self.shap_config["paths"]["result_path"])

    @property
    def feature_base_path(self):
        """Data path for the features of the specified analysis."""
        return os.path.normpath(self.shap_config["paths"]["feature_path"])

    @property
    def plot_base_path(self):
        """Path for storing the shap plots."""
        return os.path.normpath(self.shap_config["paths"]["plot_path"])

    @property
    def analysis_type(self):
        """Type of analysis. Is "main" or "None", because we did not include a "suppl" folder on the cluster."""
        return "main" if self.config["general"]["analysis"] == "main" else "suppl"

    @property
    def analysis_level_path(self):
        """
        Used for loading the requested data efficiently. This is not equivalent to 'analysis', because
        for the supplementary analysis 'weighting_by_rel', we use the same data as an in main analysis up to
        the Multilevel Modeling and 'main' is on the same hierarchical level as the suppl_types.
        """
        return (
            None
            if self.suppl_type in ["sep_ftf_cmc", "sep_pa_na", "add_wb_change"]
            else "main"
        )

    @property
    def suppl_type(self):
        """Type of supplementary analysis, only defined if self.analysis == 'suppl', e.g. 'sep_ftf_cmc'."""
        return (
            None
            if self.analysis_type == "main"
            else self.config["general"]["suppl_type"]
        )

    @property
    def suppl_type_level_path(self):
        """
        Used for loading the requested data efficiently. This is not equivalent to 'analysis', because
        for the supplementary analysis 'weighting_by_rel', we use the same data as an in main analysis up to
        the Multilevel Modeling.
        """
        return self.suppl_type if self.suppl_type != "weighting_by_rel" else None

    @property
    def suppl_var(self):
        """Var of supplementary analysis, only defined if self.suppl_type exists, e.g. 'ftf'."""
        return (
            None
            if self.analysis_type == "main" or self.suppl_type == "add_wb_change"
            else self.config["general"]["suppl_var"]
        )

    @property
    def suppl_var_level_path(self):
        """
        Used for loading the requested data efficiently. This is not equivalent to 'analysis', because
        for the supplementary analysis 'weighting_by_rel', we use the same data as an in main analysis up to
        the Multilevel Modeling.
        """
        return self.suppl_var if self.suppl_type != "weighting_by_rel" else None

    @property
    def study(self):
        """Study, either mse or ssc."""
        return self.config["general"]["study"]

    @property
    def feature_inter_path(self):
        """
        Specific intermediate data path for the features based on current analysis. Thus,
        it connects the analysis_level_path (e.g., main) with further hierarchies, e.g.
        concerning suppl_type or suppl_var, and the study. One example would simply be ("main/ssc").
        For specific processing, e.g. the esm-sample must be added.
        """
        path_components = [
            self.analysis_level_path,
            self.suppl_type_level_path,
            self.suppl_var_level_path,
            self.study,
        ]
        # Filter out empty or None values
        filtered_path_components = [comp for comp in path_components if comp]
        return os.path.normpath(os.path.join(*filtered_path_components))

    @property
    def result_plot_inter_path(self):
        """Path for the results and plots on the level of study."""
        if self.suppl_type == "weighting_by_rel":
            analysis_level_path = None
        else:
            analysis_level_path = self.analysis_level_path
        path_components = [
            analysis_level_path,
            self.suppl_type,
            self.suppl_var,
            self.study,
        ]
        # Filter out empty or None values
        filtered_path_components = [comp for comp in path_components if comp]
        return os.path.normpath(os.path.join(*filtered_path_components))

    @property
    def result_path(self):
        """Returns the path for the results up to the specified study."""
        return os.path.normpath(
            os.path.join(self.result_base_path, self.result_plot_inter_path)
        )

    @property
    def plot_path(self):
        """Returns the path for the results (up to the specified study)."""
        return os.path.normpath(
            os.path.join(self.plot_base_path, self.result_plot_inter_path)
        )

    @property
    def feature_path(self):
        """Returns the path for the features (up to the specified study)."""
        return os.path.normpath(
            os.path.join(self.result_base_path, self.feature_inter_path)
        )

    @property
    def shap_plots(self):
        """List of the different shap plots to create."""
        return self.shap_config["plots"]["plot_types"]

    @property
    def esm_samples(self):
        """List of the esm-samples used for this analysis."""
        return self.config["general"]["samples_for_analysis"]

    def apply_methods(self):
        """This function applies the preprocessing methods specified in the config."""
        for method in self.config["analysis"]["shap_value_analysis"]["methods"]:
            if method not in dir(ShapValueAnalyzer):
                raise ValueError(f"Method '{method}' is not implemented yet.")
            getattr(self, method)()

    def load_data_from_folders(self):
        """
        This function loads the data from all subdirectories of a base directory that is specified in the config.
        Because this class only processes shap values, we exclude e.g. the files containing the cv_results.
        For enabling the comparison of different esm_samples in one plot, we iterate over esm_samples
        and added the esm_sample as another hierarchy level in the Dict Structure.
        It does not return anything but sets the loaded dictionary containing the shap_values as an attribute
        of the class (self.results)
        SHAP values for the SVR are skipped, because some computations failed -> JSON files are corrupted
        """
        for esm_sample in self.esm_samples:
            self.results[esm_sample] = {}
            esm_sample_path = os.path.join(self.result_path, esm_sample)
            for root, dirs, files in os.walk(esm_sample_path):
                if not dirs:
                    # Get the path components
                    relative_path = os.path.relpath(root, esm_sample_path)
                    components = relative_path.split(os.path.sep)
                    if "svr" in components:
                        continue
                    # Navigate and create nested dictionaries
                    current_level = self.results[esm_sample]
                    for comp in components:
                        current_level = current_level.setdefault(comp, {})
                    # Load and scale features
                    # Note: Features here contain the column order after scaling
                    features = self.get_and_scale_features(components, esm_sample)
                    current_level["features"] = features
                    # Load JSON files into the terminal dictionary
                    for file in files:
                        if file in [
                            "shap_values.json",
                            "shap_ia_values.json",
                            "features",
                        ]:
                            print(relative_path)
                            with open(os.path.join(root, file), "r") as f:
                                if file in ["shap_values.json", "shap_ia_values.json"]:
                                    current_level[file[:-5]] = json.load(f)
                                else:
                                    current_level[file] = json.load(f)
        print("loaded data")

    def get_and_scale_features(self, components, esm_sample):
        """
        This function
            loads the features used for the ML analysis.
            selects features (only for the linear_baseline_analyzer)
            scales continuous features (e.g., for correct presentation of SHAP plots)
            handles the column order problem
        Using all features and scaling them is actually not 100% correct, because scaling happens in the pipeline
        (i.e., only for a subset of data) to prevent data leakage. But divergence should be negligible for the purpose
        of creating the SHAP plots.

        Args:
            components: Tuple, Path components resulting from relative_path.split() representing e.g. the fis,
                or the sample (the relative_path represents the folder structure) of the data.
            esm_sample: str, given esm_sample

        Returns:
            features_result: df, containing the scaled features and its values with the correct assignment
        """
        # Construct the base path and filename
        sample = esm_sample
        feature_inclusion_strategy = components[0]
        model = components[1]
        if self.study == "ssc":
            soc_int_var = components[2]
            file_name = f"{sample}_{soc_int_var}_one_hot_encoded_preprocessed.pkl"
        elif self.study == "mse":
            soc_int_var = None
            major_soc_event = self.config["analysis"]["result_analysis"][
                "mse_assignment"
            ][sample]
            file_name = f"{sample}_{major_soc_event}_one_hot_encoded_preprocessed.pkl"
        else:
            raise ValueError("study not implemented")

        feature_inclusion_data = (
            "single_items"
            if feature_inclusion_strategy in ["single_items", "feature_selection"]
            else "scale_means"
        )
        # Assemble the full path
        feature_path = os.path.join(
            self.feature_base_path,
            self.feature_inter_path,
            "traits",
            feature_inclusion_data,
            file_name,
        )

        # Load and scale the features
        with open(feature_path, "rb") as f:
            features = pickle.load(f)
        if model == "linear_baseline_model":
            features = features[
                [
                    col
                    for col in features.columns
                    if any(
                        col.startswith(prefix)
                        for prefix in ["age", "sex", "educational_attainment", "bfi2"]
                    )
                ]
            ]
            features = features.drop("sex_clean_1", axis=1)

        # Set feature order before scaling as attribute
        self.set_column_order_attributes(
            features=features,
            sample=sample,
            fis=feature_inclusion_strategy,
            soc_int_var=soc_int_var,
            time_var="before",
        )
        # Scale only continuous columns
        binary_cols = features.columns[(features.isin([0, 1])).all(axis=0)]
        non_binary_cols = features.columns.difference(
            binary_cols
        )  # This line breaks the column order
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features[non_binary_cols])
        scaled_features_df = pd.DataFrame(
            scaled_features, columns=non_binary_cols, index=features.index
        )
        features_result = pd.concat([scaled_features_df, features[binary_cols]], axis=1)

        # set feature order after scaling as attribute
        self.set_column_order_attributes(
            features=features_result,
            sample=sample,
            fis=feature_inclusion_strategy,
            soc_int_var=soc_int_var,
            time_var="after",
        )
        # create a mapping between the column orders as class attribute
        self.set_column_order_mapping(
            sample=sample,
            fis=feature_inclusion_strategy,
            soc_int_var=soc_int_var,
        )

        return features_result

    def set_column_order_attributes(self, features, sample, fis, soc_int_var, time_var):
        """
        This function is a fix of the feature order issue caused by the custom scaler.
        It sets the column order of the features before scaling and after scaling as class attributes due adequately
        handle the different orders. Because the number of columns and the column order, respectively, may depend on
            the esm-sample (dataset)
            the feature inclusion strategy (fis)
            the soc_int_var (soc_int_var)
        separate orders are stored in the class attributes based on these paramters

        Args:
            features: df, containing the features of a given analysis setting.
            sample: str, esm-sample, e.g., "coco_int"
            fis: str, feature_inclusion_strategy, e.g., "scale_means"
            soc_int_var: str, social situation variavle, e.g., "social_interaction"
            time_var: str, determining if it is the column order before or after scaling, therefore, is must be
                "before" or "after"
        """
        if sample not in getattr(self, f"col_order_{time_var}_scaling"):
            getattr(self, f"col_order_{time_var}_scaling")[sample] = {}
        if self.study == "mse":
            getattr(self, f"col_order_{time_var}_scaling")[sample][
                fis
            ] = features.columns.tolist()
        elif self.study == "ssc":
            if fis not in getattr(self, f"col_order_{time_var}_scaling")[sample]:
                getattr(self, f"col_order_{time_var}_scaling")[sample][fis] = {}
            getattr(self, f"col_order_{time_var}_scaling")[sample][fis][
                soc_int_var
            ] = features.columns.tolist()

    def set_column_order_mapping(self, sample, fis, soc_int_var):
        """
        This sets a mapping between the column order before and after scaling using both class attributes

        Args:
            sample: esm_sample
            fis: feature_inclusion_strategy
            soc_int_var: soc_int_var
        """
        if sample not in getattr(self, f"col_order_mapping"):
            getattr(self, f"col_order_mapping")[sample] = {}
        if self.study == "mse":
            mapping = {
                old_col: new_col
                for old_col, new_col in zip(
                    self.col_order_before_scaling[sample][fis],
                    self.col_order_after_scaling[sample][fis],
                )
            }
            getattr(self, f"col_order_mapping")[sample][fis] = mapping
        elif self.study == "ssc":
            if fis not in getattr(self, f"col_order_mapping")[sample]:
                getattr(self, f"col_order_mapping")[sample][fis] = {}
            mapping = {
                old_col: new_col
                for old_col, new_col in zip(
                    self.col_order_before_scaling[sample][fis][soc_int_var],
                    self.col_order_after_scaling[sample][fis][soc_int_var],
                )
            }
            getattr(self, f"col_order_mapping")[sample][fis][soc_int_var] = mapping

    def apply_shap_analyses(self):
        """
        This function is a wrapper for the analyses that are applied to all shap_values. Specifically, it
            creates the scatter plot comparing main and interactive effects
            Loops through the analysis settings and creates
                summary plots showing the feature importance and direction (based on SHAP values)
                heatmap plots comparing specific main and interactive effects (based on SHAP IA values)
                dependence plots visualizing specific interactions (not used and not tested ATM)
            summarizes the SHAP values on different levels of abstraction
            creates the global importance plot based on broad feature categories
        """
        print("Currently using:", self.result_base_path, "for loading the shap values")
        dct = self.results.copy()
        plot_data = self.structure_shap_values_for_plots(dct, self.study)
        lst_of_plots = self.shap_config["plots"]["plot_types"]
        shap_agg_dict = {}

        # Global Plot Style
        if self.shap_config["plots"]["ggplot_style"]:
            plt.style.use("ggplot")
        plt.rcParams["axes.facecolor"] = self.shap_config["plots"]["axes_facecolor"]

        # IA Scatter Plot -> Allow flexibility to draw different samples / models in one plot
        if "ia_scatter_plot" in lst_of_plots and plot_data:
            self.shap_ia_scatter_plot_wrapper(
                data=plot_data,
            )

        # Having applied "structure_shap_values_for_plots", we can use the same loop for mse and ssc
        for esm_sample, esm_sample_vals in plot_data.items():
            shap_agg_dict[esm_sample] = {}
            for (
                soc_int_var,
                soc_int_var_vals,
            ) in esm_sample_vals.items():  # if mse: ['dummy']
                shap_agg_dict[esm_sample][soc_int_var] = {}
                # summary plots for multiple fis and models
                print(esm_sample, soc_int_var)
                if "summary_plot" in lst_of_plots:
                    self.summary_plot_wrapper(
                        data_dct=soc_int_var_vals,
                        esm_sample=esm_sample,
                        soc_int_var=soc_int_var,
                    )

                # Create IA plots if specified in config and if SHAP IA values exist for the current setting
                if self.find_shap_ia_values(soc_int_var_vals):
                    print(esm_sample, soc_int_var, "xxx")
                    if self.shap_config["storing_shap_ia_values"]["store"]:
                        self.store_ia_values(
                            ia_values=soc_int_var_vals,
                            esm_sample=esm_sample,
                            soc_int_var=soc_int_var,
                        )
                    if "ia_heatmap" in lst_of_plots:
                        self.ia_heatmap_wrapper(
                            ia_values=soc_int_var_vals,
                            esm_sample=esm_sample,
                            soc_int_var=soc_int_var,
                        )
                    if "dependence_plot" in lst_of_plots:
                        ia_pair_dct = self.get_ia_pairs(
                            ia_values=soc_int_var_vals,
                            esm_sample=esm_sample,
                            soc_int_var=soc_int_var,
                        )
                        self.dependence_plot_wrapper(
                            data_dct=soc_int_var_vals,
                            esm_sample=esm_sample,
                            soc_int_var=soc_int_var,
                            ia_pairs_dct=ia_pair_dct,
                        )
                else:
                    print(
                        f"no SHAP interaction values found for {self.analysis_type}_{self.suppl_type}_"
                        f"{self.suppl_var}_{self.study}_{esm_sample}"
                    )

                # Aggregate SHAP values across abstraction levels
                if "importance_plot" in self.shap_config["plots"]["plot_types"]:
                    for fis, fis_vals in soc_int_var_vals.items():
                        shap_agg_dict[esm_sample][soc_int_var][fis] = {}
                        for model, model_vals in fis_vals.items():
                            if (
                                model == "svr"
                            ):  # skip svr -> NaN, unreliable SHAP values
                                continue
                            self.calc_abs_shap_values(
                                plot_data[esm_sample][soc_int_var][fis][model][
                                    "shap_values"
                                ]
                            )
                            print("summarize abstraction levels")
                            print(esm_sample, soc_int_var, fis, model)
                            shap_agg_dict[esm_sample][soc_int_var][fis][
                                model
                            ] = self.summarize_abstraction_levels(
                                shap_val_dct=plot_data[esm_sample][soc_int_var][fis][
                                    model
                                ],
                                fis=fis,
                                esm_sample=esm_sample,
                            )

        # Store the SHAP values for different abstraction levels
        if self.shap_config["storing_shap_values"]["store_abstraction_levels"]:
            spec_output_dir = os.path.join(
                self.shap_config["storing_shap_values"]["output_dir"],
                self.result_plot_inter_path,
            )
            self.create_dir_structure(
                shap_dct=shap_agg_dict, output_dir=spec_output_dir
            )

        # Create global importance plot for broad categories
        if "importance_plot" in self.shap_config["plots"]["plot_types"]:
            self.importance_plot_wrapper(plot_data)

    def create_dir_structure(self, shap_dct, output_dir):
        """
        Create a nested directory structure based on a given dictionary and save SHAP values as JSON files.
        This method recursively creates directories and saves SHAP values based on the structure of the
        input dictionary `shap_dct`. The method checks if each key in the dictionary corresponds to
        a model specified in the configuration (`shap_config`). If it does, the value (assumed to be SHAP values)
        is saved as a JSON file named `shap_values.json` in the current directory. If the key does not correspond
        to a model, a new subdirectory is created and the function continues recursively.

        Args:
            shap_dct: Dict, The dictionary containing SHAP values and nested keys representing the
                desired directory structure.
            output_dir: str, The base directory where the nested structure will be created.
        """

        def create_structure_recursive(sub_dict, current_path):
            """
            Recursively create directories and save SHAP values as JSON files. This inner function navigates
            through the nested dictionary `sub_dict`, creates directories for each key that is not a model,
            and saves SHAP values as JSON files for model keys.

            Args:
                sub_dict: Dict, The current level of the dictionary to process.
                current_path: str, The current directory path where the subdirectories or files will be created.
            """
            # Iterate through each key in the sub-dictionary
            for key, value in sub_dict.items():
                if key in self.shap_config["storing_shap_values"]["models"]:
                    json_path = os.path.join(current_path, "shap_values.json")
                    with open(json_path, "w") as json_file:
                        json.dump(value, json_file, indent=4)
                    continue
                # Create a new subdirectory for each key that is not a model key
                new_path = os.path.join(current_path, key)
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                create_structure_recursive(value, new_path)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Start the recursive directory creation
        create_structure_recursive(shap_dct, output_dir)

    def store_ia_values(self, ia_values, esm_sample, soc_int_var=None):
        """
        This function stores the corrected shap ia values as dictionaries. These were only obtained for
        certain analysis settings using the RFR. The rescaling (/1000000) is due to an upscaling before
        storing them on the cluster

        Args:
            ia_values: Dict, containing the SHAP interaction values
            esm_sample: current esm sample
            soc_int_var: current soc int var
        """
        for fis, fis_vals in ia_values.items():
            if fis == "feature_selection":  # no ia_values for fs
                continue
            # correct feature assignments
            corrected_ia_values = self.correct_feature_assignment(
                shap_ia_dct=fis_vals["rfr"]["shap_ia_values"].copy(),
                esm_sample=esm_sample,
                fis=fis,
                soc_int_var=soc_int_var,
            )
            # Applies division only to numeric columns
            processed_dict = {
                dataset: {
                    ia_df_key: (
                        ia_df.apply(
                            lambda col: col / 1000000 if col.dtype.kind in "fi" else col
                        ).to_dict()
                    )
                    for ia_df_key, ia_df in dataset_dct.items()
                }
                for dataset, dataset_dct in corrected_ia_values.items()
            }
            if soc_int_var is None:
                soc_int_var = "dummy"  # for MSE
            store_path = os.path.join(
                self.shap_config["storing_shap_ia_values"]["output_dir"],
                self.result_plot_inter_path,
                esm_sample,
                soc_int_var,
                fis,
                "rfr",
            )
            os.makedirs(store_path, exist_ok=True)
            filename = os.path.join(store_path, "shap_ia_values.json")
            with open(filename, "w") as json_file:
                json.dump(processed_dict, json_file, indent=4)

    def find_shap_ia_values(self, dct):
        """
        Recursively searches for 'shap_ia_values' key in nested dictionary and checks if it's not empty.

        Args:
            dct: Dict, current dict hierarchy that is searched through
        """
        if isinstance(dct, dict):
            for key, value in dct.items():
                if (
                    key == "shap_ia_values" and value
                ):  # Check if key is 'shap_ia_values' and dict is not empty
                    return True
                elif isinstance(
                    value, dict
                ):  # If the value is another dict, search recursively
                    if self.find_shap_ia_values(value):
                        return True
        return False

    @staticmethod
    def calc_abs_shap_values(shap_val_dct):
        """
        This function calculates the absolute value of the shap values and summarize it across persons.
        It modifies the given Dict inplace.

        Args:
            shap_val_dct: Nested dict containing the shap values (train/test -> specific shap values)
        """
        for dataset in ["train", "test"]:
            shap_val_dct[dataset]["abs_avg_across_reps_samples"] = np.mean(
                np.abs(shap_val_dct[dataset]["avg_across_reps"]), axis=0
            ).tolist()

    def summarize_abstraction_levels(self, shap_val_dct, fis, esm_sample):
        """
        This function summarizes the raw shap values (calculated for every single one-hot-encoded feature)
        on different abstraction levels (single features, scale means, broad categories).
        Therefore, it takes the absolute shap values, because otherwise important features could cancel
        each other out and results would be meaningless.
        It changes the given dictionary (shap_val_dct) in-place.

        Args:
            shap_val_dct: Dict, containing the SHAP values, is modified in-place
            fis: str, feature inclusion strategy
            esm_sample: str, given ESM-sample
        """
        print("xxx", fis)
        shap_vals = shap_val_dct["shap_values"]
        feature_names = shap_val_dct["features"].columns.tolist()
        for dataset in ["train", "test"]:
            # set the ordered abs shap values as dct
            shap_vals[dataset]["abs_shap_values_dct"] = {
                feature_name: shap_val
                for feature_name, shap_val in sorted(
                    zip(
                        feature_names, shap_vals[dataset]["abs_avg_across_reps_samples"]
                    ),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )
            }
            # set a dict with the features names and the value for each person
            shap_vals[dataset]["avg_shap_dct"] = {
                feature_names[i]: [
                    person[i] for person in shap_vals[dataset]["avg_across_reps"]
                ]
                for i in range(len(feature_names))
            }
            shap_data = shap_vals[dataset]["abs_shap_values_dct"]

            # Aggregate acros single features
            shap_vals[dataset]["shap_abstr_categoricals"] = self.summarize_categoricals(
                shap_data
            )

            # Aggregate across scale means
            if fis != "scale_means":
                shap_vals[dataset][
                    "shap_abstr_scale_means"
                ] = self.summarize_scale_means(
                    shap_vals[dataset]["shap_abstr_categoricals"]
                )

            # Aggregate across broad feature categories
            shap_vals[dataset][
                "shap_abstr_broad_categories"
            ] = self.summarize_broad_categories(shap_data)
        return shap_vals

    @staticmethod
    def summarize_categoricals(abs_shap_dct):
        """
        This function summarizes the absolute shap values of one-hot-encoded categoricals to assign one importance
        value to one categorical. More specifically, it calculates a sum importance and a mean importance, so that
        one can differentiate between the effect of the whole category and the average effect ob one category
        of this cateogiral variable.

        Args:
            abs_shap_dct: Dict, containing the individual shap values for all variables

        Returns:
            processed_data: Dict containing the summarized shap values (mean and sum) for categorical variables
                                 and individual shap values for other variables as values and the feature names as
                                 keys
        """
        grouped_data = defaultdict(list)
        # Grouping the data by categorical variable
        processed_data = {}
        for feature, value in abs_shap_dct.items():
            category = re.match(r"(\w+)_clean_(\d+)", feature)
            if category:
                grouped_data[category.group(1)].append(value)
            else:
                processed_data[feature] = [value]
        # Calculating sum and mean
        for category, values in grouped_data.items():
            sum_val = sum(values)
            mean_val = sum_val / len(values)
            processed_data[category] = [sum_val, mean_val]
        return processed_data

    def summarize_scale_means(self, abs_shap_dct):
        """
        This function summarizes the absolute shap values obtained in the fis "single_items" or "feature_selection"
        according to the scale means. Therefore, it takes the output of the method summarize_categoricals, so that the
        shap value for former one-hot encoded variables are already summarized. Again, it computes a sum and a mean
        to differ between the full effect and the average effect of one OHE variable belong to a scale.

        Args:
            abs_shap_dct: Dict containing the individual shap values for continuous variables and the already
                summarized mean and sum of the categorical variables.

        Returns:
            results_scale_means: Dict containing the summarized shap values (mean and sum) for categorical variables
                and continuous variables that can be assigned to a scale
        """
        trait_config = self.config["trait_data"]
        # only can form scale means for personality or political and societal attitudes
        config_pl_ques_vars = (
            trait_config["personality"] + trait_config["polit_soc_attitudes"]
        )
        ques_names = {
            re.match(r"(.*?)_\d+_clean", key).group(1)
            if re.match(r"(.*?)_\d+_clean", key)
            else key
            for key in abs_shap_dct.keys()
        }
        # For corona_aff_binaries -> make another regex
        ques_names = {
            re.match(r"(.*?)_\d", key).group(1) if re.match(r"(.*?)_\d", key) else key
            for key in ques_names
        }
        # apply this also to the abs_shap_dct
        abs_shap_dct = {
            re.match(r"(.*?)_\d", key).group(1)
            if re.match(r"(.*?)_\d", key)
            else key: value
            for key, value in abs_shap_dct.items()
        }
        results_scale_means = {}
        for var_name in ques_names:
            # Check if the extracted ques is in the personality or pol_soc_att in the config
            ques_config = next(
                (item for item in config_pl_ques_vars if item["name"] == var_name), None
            )
            if not ques_config:
                results_scale_means[var_name] = abs_shap_dct[var_name]
                continue
            if "dimension_mapping" in ques_config:
                dimension_mapping = ques_config["dimension_mapping"]
            else:  # only 1 dimension, no dimension mapping specified in the config, create one
                item_count = sum(
                    key.startswith(var_name) for key in abs_shap_dct.keys()
                )
                dimension_mapping = {"": [range(1, item_count + 1)]}

            # structure data
            organized_data = defaultdict(list)
            for dimension, items in dimension_mapping.items():
                for item in items:
                    key = f"{var_name}_{item}_clean"
                    if key in abs_shap_dct:
                        organized_data[f"{var_name}_{dimension}"].append(
                            abs_shap_dct[key]
                        )
            # Calculating sum and mean
            for ques_dim, values in organized_data.items():
                sum_val = sum(value for sublist in values for value in sublist)
                num_elements = sum(len(sublist) for sublist in values)
                mean_val = sum_val / num_elements if num_elements else 0
                results_scale_means[f"{ques_dim}"] = [sum_val, mean_val]
        return results_scale_means

    def summarize_broad_categories(self, abs_shap_dct):
        """
        This function summarizes the absolute shap values across broad categories (socio-demoagraphics,
        personality, political and societal attitudes, country-level variables [if available]).
        Therefore, it again takes all raw shap values as the input (otherwise calculations of sums and means
        would be flawed) and assigns it to one of the categories. Again, an average and a sum is computed and stored.

        Args:
            abs_shap_dct: Dict containing the individual shap values for all variables

        Returns:
            results_scale_means: Dict containing the summarized shap values (1st value: sum, 2nd value: mean)
                across categories
        """
        # extract config names of the dct keys
        cfg_names = self.extract_name_generic(abs_shap_dct)
        category_mapping = self.create_cfg_category_mapping()
        # Organizing data based on category mapping
        organized_data = defaultdict(list)
        for data_var, value in abs_shap_dct.items():
            extracted_name = cfg_names[data_var]
            for category, variables in category_mapping.items():
                if extracted_name in variables:
                    organized_data[category].append(value)
        # Calculating sum and mean
        result = {}
        for category, values in organized_data.items():
            sum_val = sum(values)
            mean_val = sum_val / len(values)
            result[category] = [sum_val, mean_val]
        return result

    def create_cfg_category_mapping(self):
        """
        This function creates a mapping between single variables and the broad categories defined in the config.

        Returns:
            category_mapping, dict with the broad categories as keys and a list containing the associated variables

        """
        trait_config = self.config["trait_data"]
        category_mapping = defaultdict(list)
        for category, items in trait_config.items():
            if category in self.shap_config["categories"]:
                for item in items:
                    if category == "country_vars":
                        for var_name in item.get("var_names", []):
                            category_mapping[category].append(var_name)
                    else:
                        category_mapping[category].append(item["name"])
        if self.suppl_type == "add_wb_change":
            category_mapping["wb_change"] = ["wb_change_pre_post_event"]
        return category_mapping

    @staticmethod
    def extract_name_generic(shap_dct):
        """
        This function creates a mapping between the names of features as they are currently used in the code
        (e.g., with suffix, or with one-hot encoding numbers) and maps it to the corresponding entry in the cfg.

        Args:
            shap_dct: Could be a list or a dict containing the feature names as they are in the code

        Returns:
            extracted_names: Dict, containing a key:value mapping between the current name of the var in the code
                and the var in the config
        """
        extracted_names = {}
        for key in shap_dct:
            if "_clean" not in key:  # country_vars
                name = key
            else:
                if re.match(r".*_clean_\d+", key):
                    if "_aff_" in key:  # edgecase
                        name = re.match(r"(.*?)_\d+_clean_\d+", key).group(1)
                    else:
                        name = re.match(r"(.*?)_clean_\d+", key).group(1)
                elif re.match(r".*_\d+_clean", key):
                    name = re.match(r"(.*?)_\d+_clean", key).group(1)
                elif re.match(r".*_clean", key):
                    name = re.match(r"(.*?)_clean", key).group(1)
                else:
                    raise ValueError(f"Structure {key} cannot be matched")
            extracted_names[key] = name
        return extracted_names

    def ia_heatmap_wrapper(self, ia_values, esm_sample, soc_int_var=None):
        """
        This function is used as a wrapper to plot insights from the shap interaction values
        Shap Values were already summarized on the cluster, this includes
            1) The SHAP interaction values summarized across persons
            2) The most influential interactions per person, thus a) the variables and
                b) the pairs that interacts most in two dictionaries
            3) The interaction values summarized across features and samples, so that for each feature there
                is a score reflecting a "global" interaction value

        The feature assignment correction for the ia_corr_matrix is done in this method.

        Args:
            ia_values: the different summaries of the shap interaction values for train and test set
            esm_sample: ESM-sample
            soc_int_var: [str, None], Current soc_int_var (if SSC) or dummy/None (if MSE)
        """
        for fis, fis_vals in ia_values.items():
            if fis == "feature_selection":  # no ia_values for fs
                continue
            # correct feature assignments
            corrected_ia_values = self.correct_feature_assignment(
                shap_ia_dct=fis_vals["rfr"]["shap_ia_values"].copy(),
                esm_sample=esm_sample,
                fis=fis,
                soc_int_var=soc_int_var,
            )
            for dataset in ["train", "test"]:
                self.shap_ia_corr_matrix(
                    data=corrected_ia_values[dataset],
                    fis=fis,
                    esm_sample=esm_sample,
                    dataset=dataset,
                    soc_int_var=soc_int_var,
                )

    def get_ia_pairs(self, ia_values, esm_sample, soc_int_var=None):
        """
        This function is used to extract the variable pairs that will be visualized in the dependence
        plot, thus, the most interacting variables in a certain analyses.
        There are multiple possibilities to summarize the shap interaction values.
        We chose the most common method, thus take the mean absolut shap ia values across persons
        and then choose the variable pairs with the highest values
        ##NOTE: Currently deprecated##

        Args:
            ia_values: Dict, contaning the IA values
            esm_sample: str, ESM sample
            soc_int_var: [str, None], Current soc_int_var (if SSC) or dummy/None (if MSE)

        Returns:
            best_pair_dct: Dict containing the most interacting pairs of variables
        """
        best_pair_dct = {}
        for fis, fis_vals in ia_values.items():
            if fis == "feature_selection":  # no ia_values for fs
                continue
            best_pair_dct[fis] = {}
            corrected_ia_values = self.correct_feature_assignment(
                fis_vals["rfr"]["shap_ia_values"].copy(), esm_sample, fis, soc_int_var
            )
            for dataset in ["train", "test"]:
                best_pair_dct[fis][dataset] = {}
                current_df = corrected_ia_values[dataset][
                    "abs_agg_ia_persons"
                ].copy()  # this is a df
                main_effect_sizes = current_df.values.diagonal().copy()
                np.fill_diagonal(current_df.values, np.nan)

                # Get the top N interaction values sorted by their interaction size
                top_n = (
                    self.shap_config["plots"]["dependence_plot"]["num_ia_pairs"] * 2
                )  # a bit hacky
                top_n_interactions = current_df.stack().nlargest(top_n)
                # Sort the feature pairs based on the main effect size
                sorted_top_n_interactions = {}
                for pair in top_n_interactions.index:
                    sorted_pair = sorted(
                        pair,
                        key=lambda x: main_effect_sizes[current_df.columns.get_loc(x)],
                        reverse=True,
                    )
                    sorted_top_n_interactions[tuple(sorted_pair)] = top_n_interactions[
                        pair
                    ]
                best_pair_dct[fis][dataset] = sorted_top_n_interactions
        return best_pair_dct

    def shap_ia_scatter_plot_wrapper(self, data, soc_int_var=None):
        """
        This is a wrapper for the shap_ia_scatter_plot method. It creates the root plot on which the subplots
        for a more specific setting are plotted and apply some formatting. Currently, the subplots are filled
        with data from different ESM smaples and different feature inclusion strategies that are definde in the
        config (e.g., coco_int+Emotions x single_items+scale_means, as shown in the paper).

        Args:
            data: Dict, containing the SHAP ia_values
            soc_int_var: [str, None], Current soc_int_var (if SSC) or dummy/None (if MSE)
        """
        fis_to_plot = ["single_items", "scale_means"]
        combos_to_plot = self.shap_config["plots"]["scatter_plot"]["sample_var_combos"][
            self.study
        ]
        dataset_to_plot = self.shap_config["plots"]["scatter_plot"]["dataset_to_plot"]
        combos_data = {}
        esm_samples = []
        name_suffix = ""
        # Extract relevant data from full dict
        for combo in combos_to_plot:
            for esm_sample, esm_sample_vals in data.items():
                for soc_int_var, soc_int_var_vals in esm_sample_vals.items():
                    if (
                        combo["esm_sample"] == esm_sample
                        and combo["soc_int_var"] == soc_int_var
                    ):
                        combos_data[f"{esm_sample}_{soc_int_var}"] = soc_int_var_vals
                        esm_samples.append(esm_sample)
                        name_suffix += f"{esm_sample}_{soc_int_var}_"
                        soc_int_var_to_plot = soc_int_var
        name_suffix += dataset_to_plot

        fig, axes = plt.subplots(
            len(fis_to_plot), len(combos_to_plot), figsize=(25, 16)
        )
        for i, (combo, data) in enumerate(combos_data.items()):
            for j, fis in enumerate(fis_to_plot):
                ax = axes[j, i]
                # correct feature assignment
                data_correct = self.correct_feature_assignment(
                    shap_ia_dct=data[fis]["rfr"]["shap_ia_values"].copy(),
                    esm_sample=esm_samples[i],
                    fis=fis,
                    soc_int_var=soc_int_var,
                )
                # Plot the current esm sample - fis combo (i,j) in the given subplot
                self.shap_ia_scatter_plot(
                    data=data_correct[dataset_to_plot],
                    combo=combo,
                    fis=fis,
                    ax=ax,
                    esm_sample=esm_samples[i],
                    soc_int_var=soc_int_var_to_plot,
                )
                # Apply scientific notation conditionally if values are too small
                ax.ticklabel_format(
                    style="sci", axis="y", scilimits=(-4, 4), useMathText=True
                )
                ax.yaxis.get_offset_text().set_fontsize(14)

        # Adjust layout and display
        fig.tight_layout(pad=3.5)
        if self.shap_config["plots"]["store_plots"]:
            self.store_plot(plot_type="ia_scatter_plot", name_suffix=name_suffix)
        else:
            plt.show()
            self.check_plot_grayscale(fig=fig, filename_raw=None, show_plot=True)

    def correct_feature_assignment(
        self, shap_ia_dct, esm_sample, fis, soc_int_var=None
    ):
        """
        This function customly processes the SHAP IA values
            It corrects the feature assignment error caused by alphabetic sorting of the custom scaler.
            It creates dataframes for the 3 types of IA value summaries
                1) Most influential IAs -> n_features x n_features
                2) Most influential IA -> Single Features and Pairs
                3) Most influential Features across persons and interactions, n_features x 1
            It stores the df in a Dict, separately for train and test set

        Args:
            shap_ia_dct: Dict containing the different shap_ia_value summaries for train and test set
            esm_sample: Given ESM sample
            fis: given feature inclusion strategy
            soc_int_var: [str, None], Current soc_int_var (if SSC) or dummy/None (if MSE)

        Returns:
            corrected_ia_dct: Dict, like shap_ia_dct, but with correctly assigned feature names and DataFrames
                instead of dicts.
        """
        new_dct = {}
        if self.study == "ssc":
            feature_mapping = self.col_order_mapping[esm_sample][fis][soc_int_var]
        elif self.study == "mse":
            feature_mapping = self.col_order_mapping[esm_sample][fis]
        else:
            raise ValueError("Unknown study")

        for dataset in ["train", "test"]:
            new_dct[dataset] = {}

            # 1) n_features x n_features (nested dict)
            df_agg_ia_persons = pd.DataFrame.from_dict(
                shap_ia_dct[dataset]["agg_ia_persons"], orient="index"
            )
            df_agg_ia_persons = df_agg_ia_persons.rename(
                columns=feature_mapping, index=feature_mapping
            )
            new_dct[dataset]["agg_ia_persons"] = df_agg_ia_persons
            df_abs_agg_ia_persons = pd.DataFrame.from_dict(
                shap_ia_dct[dataset]["abs_agg_ia_persons"], orient="index"
            )
            df_abs_agg_ia_persons = df_abs_agg_ia_persons.rename(
                columns=feature_mapping, index=feature_mapping
            )
            new_dct[dataset]["abs_agg_ia_persons"] = df_abs_agg_ia_persons

            # 2) Most infuential IA -> Single Features and Pairs
            feature_counts = pd.Series(
                shap_ia_dct[dataset]["most_influential_ia_features"]
            )
            feature_counts_df = feature_counts.reset_index()
            feature_counts_df.columns = ["Feature", "Count"]
            feature_counts_df = feature_counts_df.rename(
                columns=feature_mapping, index=feature_mapping
            )
            feature_counts_df = feature_counts_df.applymap(
                lambda x: feature_mapping.get(x, x)
            )
            new_dct[dataset]["most_influential_ia_features"] = feature_counts_df
            pair_counts_series = pd.Series(
                shap_ia_dct[dataset]["most_influential_ia_pairs"]
            )
            pair_counts_series.index = pair_counts_series.index.map(eval)
            pair_counts_df = pair_counts_series.reset_index()
            pair_counts_df.columns = ["feature_1", "feature_2", "Count"]
            pair_counts_df = pair_counts_df.rename(
                columns=feature_mapping, index=feature_mapping
            )
            pair_counts_df = pair_counts_df.applymap(
                lambda x: feature_mapping.get(x, x)
            )
            new_dct[dataset]["most_influential_ia_pairs"] = pair_counts_df

            # 3) Most influential Features across persons and interactions, n_features x 1
            df_agg_ia_persons_ias = pd.DataFrame.from_dict(
                shap_ia_dct[dataset]["agg_ia_persons_ias"],
                orient="index",
                columns=["agg_ia_val"],
            )
            df_agg_ia_persons_ias = df_agg_ia_persons_ias.rename(
                columns=feature_mapping, index=feature_mapping
            )
            new_dct[dataset]["agg_ia_persons_ias"] = df_agg_ia_persons_ias
            df_abs_agg_ia_persons_ias = pd.DataFrame.from_dict(
                shap_ia_dct[dataset]["abs_agg_ia_persons_ias"],
                orient="index",
                columns=["agg_ia_val"],
            )
            df_abs_agg_ia_persons_ias = df_abs_agg_ia_persons_ias.rename(
                columns=feature_mapping, index=feature_mapping
            )
            new_dct[dataset]["abs_agg_ia_persons_ias"] = df_abs_agg_ia_persons_ias
        return new_dct

    def shap_ia_scatter_plot(self, data, fis, combo, ax, esm_sample, soc_int_var=None):
        """
        This function creates a scatter plot for the shap interaction values. It uses the (absolute)
        shap value matrix (n_features x n_features) summarized across persons to compare the size
        of the main effect with the size of the interaction effect for each feature.
        On the x-axis, we plot the main effect size, on the y-axis, we plot the strength of the interaction
        effect. Further, it highlights certain dots and add the feature name, it calculates and presents
        the rank correlation for each feature category and all features, and it formats parameters of the plot.
        The plot created here is plotted on ax, so it is only displayed together with the scatter plots for
        other combos.

        Args:
            data: Dict, contains the SHAP ia values for a given model (i.e., RFR) and a given feature inclusion
                strategy.
            fis: str, feature inclusion strategy
            combo: str, indicate the current ESM sample - fis combination to be plotted in this ax
            ax: ax: matplotlib.ax Object, specific position on the main plot where to plot the current data
            esm_sample: str, a given ESM sample
            soc_int_var: str, current soc_int_var
        """
        df_abs_ia_persons = data["abs_agg_ia_persons"].copy()
        df_abs_ia_persons_ias = data["abs_agg_ia_persons_ias"].copy()
        feature_names = list(df_abs_ia_persons.index)
        feature_names_pretty = self.adjust_feature_names(
            features=feature_names, esm_sample=esm_sample, fis=fis, plot="scatter_plot"
        )
        mapping = self.feature_mapping.copy()

        # Extracting main effects and interactivity per feature and rescale to original scale
        main_effects = df_abs_ia_persons.values.diagonal() / 1000000
        interactivity_measures = df_abs_ia_persons_ias["agg_ia_val"] / 1000000

        # Ensure that the feature names (indices) match between the two DataFrames
        if list(df_abs_ia_persons.index) != list(df_abs_ia_persons_ias.index):
            df_abs_ia_persons_ias = df_abs_ia_persons_ias.T
        assert list(df_abs_ia_persons.index) == list(
            df_abs_ia_persons_ias.index
        ), "Feature Assignment is wrong"

        # Assign feature to broad feature categories and calc rank correlation
        added_legend = set()
        for category_raw, color in self.shap_config["plots"][
            "cat_color_mapping"
        ].items():
            marker = self.shap_config["plots"]["cat_point_style_mapping"][category_raw]
            category_pretty = self.shap_config["plots"]["category_mapping"][
                category_raw
            ]

            category_indices = [
                i
                for i, feature in enumerate(feature_names)
                if self.get_feature_category(feature=feature, mapping=mapping, fis=fis)
                == category_raw
            ]
            cat_feature_names = [
                feature_name
                for feature_name in feature_names
                if self.get_feature_category(
                    feature=feature_name, mapping=mapping, fis=fis
                )
                == category_raw
            ]
            cat_corr, cat_p_val = self.get_corr(
                df_abs_ia_persons=df_abs_ia_persons[cat_feature_names],
                df_abs_ia_persons_ias=df_abs_ia_persons_ias.loc[cat_feature_names],
                corr_type=self.shap_config["plots"]["scatter_plot"]["corr_type"],
            )
            cat_corr_txt = self.corr_formatter(
                corr=cat_corr,
                p_val=cat_p_val,
                corr_type=self.shap_config["plots"]["scatter_plot"]["corr_type"],
            )
            # Add category as legend
            if category_indices:
                for idx in category_indices:
                    # Only add a label once to avoid duplicates in the legend
                    if idx == category_indices[0]:
                        if category_raw not in added_legend:
                            ax.scatter(
                                [],
                                [],
                                color=color,
                                marker=marker,
                                s=50,
                                label=f"{category_pretty} ({cat_corr_txt})",
                                alpha=0.5,
                            )
                            added_legend.add(category_raw)

                    # Note: Highlighting of dots is only defined for the paper plots, not the supplementary plots
                    try:
                        features_to_highlight = self.shap_config["plots"][
                            "scatter_plot"
                        ]["features_to_label"][self.analysis_type][self.suppl_type][
                            self.suppl_var
                        ][
                            self.study
                        ][
                            soc_int_var
                        ][
                            esm_sample
                        ][
                            fis
                        ]
                    except KeyError:
                        features_to_highlight = []
                    # Plot non-highlighted dots
                    if feature_names[idx] not in features_to_highlight:
                        ax.scatter(
                            x=main_effects[idx],
                            y=interactivity_measures[idx],
                            color=color,
                            marker=marker,
                            s=50,
                            alpha=0.5,
                            edgecolors=None,
                            linewidths=0.5,
                        )
                # Plot highlighted dots
                for idx in category_indices:
                    if feature_names[idx] in features_to_highlight:
                        ax.scatter(
                            x=main_effects[idx],
                            y=interactivity_measures[idx],
                            color=color,
                            marker=marker,
                            s=50,
                            alpha=1,
                            edgecolors="black",
                            linewidths=0.5,
                        )
                        # Add label with manual offset to highlighted points
                        x_new = (
                            main_effects[idx]
                            + features_to_highlight[feature_names[idx]][0]
                        )
                        y_new = (
                            interactivity_measures[idx]
                            + features_to_highlight[feature_names[idx]][1]
                        )
                        ax.text(
                            x=x_new,
                            y=y_new,
                            s=feature_names_pretty[idx],
                            color="black",
                            verticalalignment="center",
                            horizontalalignment="left",
                            fontsize=11,
                        )

        # Plot correlation across all samples and associated p values
        corr, p_val = self.get_corr(
            df_abs_ia_persons=df_abs_ia_persons,
            df_abs_ia_persons_ias=df_abs_ia_persons_ias,
            corr_type=self.shap_config["plots"]["scatter_plot"]["corr_type"],
        )
        corr_txt = self.corr_formatter(
            corr=corr,
            p_val=p_val,
            corr_type=self.shap_config["plots"]["scatter_plot"]["corr_type"],
        )
        ax.text(
            x=self.shap_config["plots"]["scatter_plot"]["txt_x"],
            y=self.shap_config["plots"]["scatter_plot"]["txt_y"],
            s=corr_txt,
            transform=ax.transAxes,
            fontsize=16,
            verticalalignment="center",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Format axes, labels, and titles
        ax.ticklabel_format(useOffset=True)
        fis_pretty = self.shap_config["plots"]["str_mapping"]["fis"][fis]
        esm_sample, soc_int_var = self.split_string_based_on_list(
            combo, self.shap_config["plots"]["esm_sample_order"]
        )
        esm_sample_pretty = self.shap_config["plots"]["esm_sample_mapping"][esm_sample]
        if self.study == "ssc":
            soc_int_var_pretty = self.shap_config["plots"]["soc_int_var_mapping"][
                soc_int_var
            ]
            ax.set_title(
                f"Main and Interaction Effects for {soc_int_var_pretty} in "
                f"{esm_sample_pretty} for {fis_pretty}",
                fontsize=17,
                fontweight="bold",
                pad=20,
            )
        elif self.study == "mse":
            event_pretty = self.shap_config["plots"]["event_mapping"][esm_sample]
            ax.set_title(
                f"Main and Interaction Effects for the {event_pretty} in "
                f"{esm_sample_pretty} for {fis_pretty}",
                fontsize=17,
                fontweight="bold",
                pad=20,
            )
        ax.set_xlabel(
            "Sum of the Absolute Main Effects of a Feature Across Persons",
            fontsize=16,
            labelpad=10,
        )
        ax.set_ylabel(
            "Sum of the Absolute Pairwise Interaction Effects \n"
            "of a Feature Across Persons and Interactions",
            fontsize=16,
            labelpad=10,
        )
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(loc="lower right", fontsize=15)

    @staticmethod
    def split_string_based_on_list(input_string, prefix_list):
        """
        Splits the input string into two parts based on the first matching prefix in the list. This is used
        e.g. to extract the ESM sample name from a string that contains the ESM sample and the soc_int_var.

        Args:
            input_string: str, words are seperated by underscores
            prefix_list: list, contains the esn_samples

        Returns:
            (input_string[:prefix_length], input_string[prefix_length + 1:]): A tuple containing the esm_sample
                name (first tuple entry) and the soc_int_var name (second tuple entry)
        """
        for prefix in prefix_list:
            if input_string.startswith(prefix):
                # Calculate the length of the prefix to split the string correctly
                prefix_length = len(prefix)
                return (
                    input_string[:prefix_length],
                    input_string[prefix_length + 1 :],
                )  # +1 to remove the underscore

    def get_corr(self, df_abs_ia_persons, df_abs_ia_persons_ias, corr_type):
        """
        This function calculates a correlation between the features with the strongest main effects and the
        features with the strongest interactivity. Currently implemented are Spearmans rank correlation (rho)
        and the classical Pearson correlation (r). In the paper, we use rho.

        Args:
            df_abs_ia_persons: pd.DataFrame containing the main and interaction shap values summarized across persons
            df_abs_ia_persons_ias: pd.DataFrame containing an interactivity measure per feature
            corr_type: Either "spearmanr" for Spearmans rank correlation of "pearsonr" for Pearson correlation

        Returns:
            correlation: float, obtained correlation of "corr_type"
            p_value: float, p_value of that correlation
        """
        main_effects = df_abs_ia_persons.to_numpy().diagonal()
        main_effects_df = pd.DataFrame(
            main_effects, index=df_abs_ia_persons_ias.index, columns=["MainEffect"]
        )
        interactivity_df = df_abs_ia_persons_ias.rename(
            columns={"agg_ia_val": "Interactivity"}
        ).reset_index(names="Feature")
        # Sort and Align DataFrames Based on Values
        aligned_main_effects = main_effects_df.sort_values(by="MainEffect").reindex(
            index=interactivity_df.sort_values(by="Interactivity")["Feature"]
        )["MainEffect"]
        aligned_interactivity = interactivity_df.set_index("Feature").sort_values(
            by="Interactivity"
        )["Interactivity"]
        # Compute correlation
        if len(df_abs_ia_persons_ias) > 1:
            if corr_type == "spearmanr":
                correlation, p_value = spearmanr(
                    aligned_main_effects, aligned_interactivity
                )
            elif corr_type == "pearsonr":
                correlation, p_value = pearsonr(
                    aligned_main_effects, aligned_interactivity
                )
            else:
                raise NotImplementedError(f"Corr {corr_type} not implemented")
        else:  # only 1 or 0 features for that feature category -> no correlation can ce computed
            correlation, p_value = None, None
        return correlation, p_value

    @staticmethod
    def corr_formatter(corr, p_val, corr_type):
        """
        This function formats the correlation for a pretty visualization on the plot according to APA standards.

        Args:
            corr: float, the correlation obtained
            p_val: float, the p value obtained for corr
            corr_type: Either "spearmanr" for Spearmans rank correlation of "pearsonr" for Pearson correlation

        Returns:
            formatted_txt: str, formatted correlation with asterisks denoting significance for presentation
        """
        if corr_type == "pearsonr":
            corr_symbol = r"$\mathit{r}$"
        elif corr_type == "spearmanr":
            corr_symbol = r"$\rho$"
        else:
            raise NotImplementedError(f"corr type {corr_type} not implemented")
        if corr:
            formatted_corr = f"{corr:.2f}".replace("0.", ".")
            if p_val < 0.001:
                sig_symbol = "***"
            elif p_val < 0.01:
                sig_symbol = "**"
            elif p_val < 0.05:
                sig_symbol = "*"
            else:
                sig_symbol = ""
            formatted_txt = corr_symbol + f": {formatted_corr}{sig_symbol}"
        else:  # if corr is None
            formatted_txt = corr_symbol + f": - "
        return formatted_txt

    def shap_ia_corr_matrix(self, data, fis, esm_sample, dataset, soc_int_var=None):
        """
        This function creates a correlation matrix for the features a) with the strongest main effects and b) with the
        strongest interaction effects. See https://www.kaggle.com/code/wti200/analysing-interactions-with-shap
        Therefore, it summarizes the absolute main effects and interaction indices across persons. Specifically, it
            extracts a given number of features with the strongest main effects (defined in config, e.g. 15)
            extracts the SHAP IA values for these features
            sorts the features by main effect size and feature category
            creates three separate heatmap in one plot
                In the upper triangle, the absolute IA values are displayed
                In the diagonal, the main effects are displayed
                In the lower triangle, the raw IA values are displayed

        Args:
            data: Dict, containing the different summaries of the shap interaction values
            fis: Feature Inclusion Strategy
            esm_sample: Current esm_sample
            dataset: Current ML dataset, "train" or "test"
            soc_int_var: [str, None], Current soc_int_var (if SSC) or dummy/None (if MSE)
        """
        category_mapping = self.create_cfg_category_mapping()
        df_abs_corr = data["abs_agg_ia_persons"]
        df_corr = data["agg_ia_persons"]
        # Note: Here, currently values are 1/1000 of the original scale

        # Extract n features with the highest absolute main effects and get its indices
        num_features = self.shap_config["plots"]["ia_heatmap"]["num_features"]
        top_features_abs_indices = (
            df_abs_corr.to_numpy().diagonal().argsort()[-num_features:][::-1]
        )
        top_features_abs = df_abs_corr.columns[top_features_abs_indices]

        # Fill the new df with corresponding IA values and sort
        top_features_df = df_corr.loc[top_features_abs, top_features_abs].copy()
        top_features_abs = df_abs_corr.loc[top_features_abs, top_features_abs]
        sorted_df = self.sort_features_by_cat_and_effect(
            top_features_df, category_mapping, fis
        )
        level_1_index = sorted_df.index.get_level_values(1)
        top_features_abs = top_features_abs.reindex(
            index=level_1_index, columns=level_1_index
        )
        mask = np.triu(np.ones(sorted_df.shape), k=1).astype(bool)
        sorted_df.values[mask] = top_features_abs.values[mask]

        # Make feature names pretty
        feature_names = sorted_df.index.get_level_values(1).tolist()
        category_names = sorted_df.index.get_level_values(0).tolist()
        category_names_pretty = [
            self.shap_config["plots"]["category_mapping"][cat] for cat in category_names
        ]
        feature_names_pretty = self.adjust_feature_names(
            feature_names, fis, esm_sample=esm_sample, plot="ia_heatmap"
        )
        new_multiindex = pd.MultiIndex.from_arrays(
            [category_names_pretty, feature_names_pretty], names=sorted_df.index.names
        )
        # Feature names as indices for rows and columns
        sorted_df.index = new_multiindex
        sorted_df.columns = new_multiindex
        sorted_df = sorted_df / 10
        # Note: Values are now 1/100 of the original scale -> this is added in Figure description

        # Create figure and masks for separate heatmaps
        fig = plt.figure(figsize=(45, 25), facecolor="lightgrey", edgecolor="r")
        ax = fig.add_subplot()
        ax.grid(False)
        mask_diagonal = np.eye(len(sorted_df), dtype=bool)
        mask_upper = np.triu(np.ones_like(sorted_df, dtype=bool), k=1)
        mask_lower = np.tril(np.ones_like(sorted_df, dtype=bool), k=-1)

        # Make the diverging heatmaps (diagonal, lower triangular) symmetric around zero
        values_diagnoal = sorted_df.values[~mask_diagonal]
        norm_diagonal = TwoSlopeNorm(
            vmin=np.nanmin(values_diagnoal), vcenter=0, vmax=np.nanmax(values_diagnoal)
        )
        values_lower = sorted_df.values[~mask_lower]
        norm_lower = TwoSlopeNorm(
            vmin=np.nanmin(values_lower) / 10,
            vcenter=0,
            vmax=np.nanmax(values_lower) / 10,
        )

        # Plot heatmap for lower triangle
        sns.heatmap(
            data=sorted_df.round(decimals=3),
            cmap=self.shap_config["plots"]["ia_heatmap"]["cmap_raw_ia_effects"],
            annot=True,
            annot_kws={"size": 18},
            fmt=self.shap_config["plots"]["ia_heatmap"]["fmt"],
            cbar=False,
            mask=~mask_lower,
            norm=norm_lower,
            ax=ax,
        )

        # Plot heatmap for upper triangle
        sns.heatmap(
            data=sorted_df.round(decimals=3),
            cmap=self.shap_config["plots"]["ia_heatmap"]["cmap_abs_ia_effects"],
            annot=True,
            annot_kws={"size": 18},
            fmt=self.shap_config["plots"]["ia_heatmap"]["fmt"],
            cbar=False,
            mask=~mask_upper,
            ax=ax,
        )

        # Plot heatmap for diagonal
        sns.heatmap(
            data=sorted_df.round(decimals=3),
            cmap=self.shap_config["plots"]["ia_heatmap"]["cmap_main_effects"],
            annot=True,
            annot_kws={"size": 18},
            fmt=self.shap_config["plots"]["ia_heatmap"]["fmt"],
            cbar=False,
            mask=~mask_diagonal,
            norm=norm_diagonal,
            ax=ax,
        )

        # Set bottom x-axis labels (features - level 2)
        labels_lvl2 = [lvl2 for _, lvl2 in sorted_df.columns]

        # Remove the old string splits and make custom string splits for optimal plot display
        labels_lvl2_raw = [strng.replace("\n", " ") for strng in labels_lvl2]
        max_length = self.shap_config["plots"]["ia_heatmap"]["strng_length_x"]
        x_labels_lvl2 = [
            self.split_long_string(feature, max_length=max_length)
            for feature in labels_lvl2_raw
        ]
        ax.set_xticklabels(x_labels_lvl2, rotation=0, ha="center", fontsize=20)

        # Calculate positions for category labels (level 1 of the MultiIndex)
        category_positions = {}
        for i, (lvl1, lvl2) in enumerate(sorted_df.columns):
            if lvl1 not in category_positions:
                category_positions[lvl1] = []
            category_positions[lvl1].append(i)

        # Add top x-axis labels (categories - level 1) as text annotations, centered
        for category, positions in category_positions.items():
            center_pos = sum(positions) / len(positions)
            ax.text(
                center_pos,
                -0.7,
                category,
                rotation=0,
                ha="center",
                va="top",
                fontsize=23,
                color="black",
                fontweight="bold",
            )

        # Some plot formatting
        ax.tick_params(axis="x", colors="black", bottom=True, top=False)
        ax.tick_params(axis="y", colors="black")
        ax.set_yticklabels(labels_lvl2, fontsize=20)
        ax.set_title(
            "SHAP Mean Absolute and Original Main and Interactive Effects Separated by Feature Category",
            color="black",
            fontsize=45,
            y=0.97,
            fontweight="bold",
            pad=150,
        )
        plt.tight_layout()
        if self.shap_config["plots"]["store_plots"]:
            self.store_plot(
                plot_type="ia_corr_heatmap",
                dataset=dataset,
                esm_sample=esm_sample,
                soc_int_var=soc_int_var,
                fis=fis,
            )
        else:
            plt.show()
            # self.check_plot_grayscale(fig=fig, filename_raw=None, show_plot=True)
            print()

    @staticmethod
    def get_feature_category(feature, mapping, fis):
        """
        This function attempts to match a given feature to its broad feature category based on a given mapping.
        It searches for the top-level category of the feature in the nested dictionary by exploring all branches
        and considering esm_sample. The top-level category is the key of the outermost dict where the match is found.
        Top-level category would e.g. be "socio_demographics" for the feature "age_1_clean", because it is the
        outermost dict key in "mapping" in the branch were "age" is located.

        Args:
            feature: str, a given feature name as it is in the code we search a matching category for
            mapping: Dict, basically a copy of the feature_mapping_yaml
            fis: str, feature inclusion strategy, used to skip irrelevant branches of the mapping.

        Returns:
            [None, top_level_key]: Either the correct match is returned, or nothing
        """
        feature_pretty = feature.replace("_clean", "")
        stack = [
            (None, mapping, iter(mapping.items()))
        ]  # Keep track of top-level parent key
        top_level_key = None  # This will hold the key of the top-level dictionary
        while stack:
            parent_key, current_mapping, iterator = stack[-1]
            try:
                key, value = next(iterator)
                if key == "hofstede":
                    print()
                # Update the top-level key when we're at the top level of the dictionary
                if parent_key is None:
                    top_level_key = key
                if (
                    key != fis
                    and isinstance(current_mapping, dict)
                    and fis in current_mapping
                ):
                    continue
                if key == feature_pretty:
                    return top_level_key
                else:
                    if isinstance(value, dict):
                        stack.append((key, value, iter(value.items())))
            except StopIteration:
                # No more items in the current level, go back up
                stack.pop()
        return None

    def sort_features_by_cat_and_effect(self, df, category_mapping, fis):
        """
        This function sorts the column and row indices 1) by category and 2) by main effect strength
        across and inside categories.

        Args:
            df: df, unordered df containing a given number of features and its SHAP values
            category_mapping: Dict, mapping between feature categories and features
            fis: str, feature inclusion strategy

        Returns:
            extended_df: df, where the rows and columns are ordered by main category and main effect strength for
                nice visualization
        """
        # Sort features by category
        mapping = self.feature_mapping.copy()
        column_tuples = [
            (self.get_feature_category(feature=col, mapping=mapping, fis=fis), col)
            for col in df.columns
        ]
        row_tuples = [
            (self.get_feature_category(feature=idx, mapping=mapping, fis=fis), idx)
            for idx in df.index
        ]
        multiindex_columns = pd.MultiIndex.from_tuples(
            column_tuples, names=["Category", "Feature"]
        )
        multiindex_rows = pd.MultiIndex.from_tuples(
            row_tuples, names=["Category", "Feature"]
        )
        multiindex_df = pd.DataFrame(
            df.values, index=multiindex_rows, columns=multiindex_columns
        )

        # Compute the highest diagonal value for each category
        diag_values = {
            feature: value
            for feature, value in zip(df.columns, df.to_numpy().diagonal())
        }
        category_max_diag = {}
        for category, features in category_mapping.items():
            category_features = [
                feature
                for feature in diag_values
                if any(cat_feature in feature for cat_feature in features)
            ]
            if (
                category_features
            ):  # Check if the category has any features in diag_values
                category_max_diag[category] = max(
                    (abs(diag_values[feature]) for feature in category_features),
                    key=abs,
                )
            else:
                category_max_diag[category] = -float(
                    "inf"
                )  # Default value if no features are present

        sorted_features = {}
        for category in category_max_diag:
            features_in_category = [
                feat
                for feat in diag_values
                if self.get_feature_category(feature=feat, mapping=mapping, fis=fis)
                == category
            ]
            # Sort features within each category based on the absolute diagonal value, in descending order
            sorted_features[category] = sorted(
                features_in_category, key=lambda x: abs(diag_values[x]), reverse=True
            )

        # Sorting categories based on their highest diagonal values
        sorted_categories = sorted(
            category_max_diag, key=lambda x: np.abs(category_max_diag[x]), reverse=True
        )

        # Reconstructing the sorted MultiIndex with separators
        sorted_multiindex = []
        for category in sorted_categories:
            if category in multiindex_columns.get_level_values(0).unique():
                sorted_multiindex.extend(
                    [
                        (category, feature)
                        for feature in sorted_features[category]
                        if feature in df.columns
                    ]
                )
                sorted_multiindex.append((category, ""))  # Adding separator

        # Creating a DataFrame with NaNs for separators
        n = len(sorted_multiindex)
        nan_array = np.full((n, n), np.nan)
        extended_df = pd.DataFrame(
            nan_array,
            index=pd.MultiIndex.from_tuples(sorted_multiindex),
            columns=pd.MultiIndex.from_tuples(sorted_multiindex),
        )

        # Filling the original values
        for (cat1, feat1), (cat2, feat2) in itertools.product(
            multiindex_df.index, multiindex_df.columns
        ):
            if (
                "sep" not in [feat1, feat2]
                and feat1 in df.columns
                and feat2 in df.columns
            ):
                extended_df.loc[(cat1, feat1), (cat2, feat2)] = multiindex_df.loc[
                    (cat1, feat1), (cat2, feat2)
                ]

        return extended_df

    def summary_plot_wrapper(self, data_dct, esm_sample, soc_int_var):
        """
        This function is a wrapper for create_summary_plot. It sets up a root plot where all the summary_plots
        are placed in. Currently, we create 6 summary plots on the big plot (2 models x 3 fis), separately
        for the training and the test set. Each modelxfis combo correspond to an ax Object, on which
        the specific subplot is plotted on.

        Args:
            data_dct: Dict, containing the SHAP values
            esm_sample: str, current esm sample
            soc_int_var: str, current soc_int_var
        """
        models_to_plot = self.shap_config["plots"]["summary_plot"]["models_to_plot"]
        fis_to_plot = self.shap_config["plots"]["summary_plot"]["fis_to_plot"]
        if self.suppl_type == "weighting_by_rel":  # no feature selection available
            if "feature_selection" in fis_to_plot:
                fis_to_plot.remove("feature_selection")

        for dataset in ["train", "test"]:
            fig, axes = plt.subplots(
                len(models_to_plot), len(fis_to_plot), figsize=(40, 16)
            )
            # Loop over model / fis combinations
            for j, fis in enumerate(fis_to_plot):
                for i, model in enumerate(models_to_plot):
                    ax = axes[i, j]
                    plt.sca(ax)
                    features_pretty = self.adjust_feature_names(
                        features=data_dct[fis][model]["features"],
                        fis=fis,
                        esm_sample=esm_sample,
                        plot="summary_plot",
                    )
                    self.create_summary_plot(
                        data=data_dct[fis][model],
                        features=features_pretty,
                        dataset=dataset,
                        model=model,
                        fis=fis,
                    )
            plt.tight_layout()
            if self.shap_config["plots"]["store_plots"]:
                self.store_plot(
                    plot_type="summary_plot",
                    dataset=dataset,
                    esm_sample=esm_sample,
                    soc_int_var=soc_int_var,
                )
            else:
                plt.show()
                self.check_plot_grayscale(fig=fig, filename_raw=None, show_plot=True)

    def dependence_plot_wrapper(self, data_dct, esm_sample, soc_int_var, ia_pairs_dct):
        """
        This function is a wrapper for "create_dependence_plot". It creates a root plot where a
        certain number of dependence plots are placed into.
        We only plot dependece plots for the RFR, because we have calculated interaction values.
        Note: This is not used ATM and not tested

        Args:
            data_dct: Dict, containing the SHAP values
            ia_pairs_dct: Series with Multiindex (feature names) and IA effect strength as value
            esm_sample: str, given ESM sample
            soc_int_var: str, given soc_int_var
            ia_pairs_dct: Dict, containing a certain number of feature pairs with the highest pairwise interactivity
        """
        fis_to_plot = self.shap_config["plots"]["dependence_plot"]["fis_to_plot"]
        num_pairs_to_plot = self.shap_config["plots"]["dependence_plot"]["num_ia_pairs"]
        model = self.shap_config["plots"]["dependence_plot"]["model"]  # rfr
        for dataset in ["train", "test"]:
            # Currently summary plot for 1 model, 3 ia_pairs and 2 fis
            fig, axes = plt.subplots(
                len(fis_to_plot), num_pairs_to_plot, figsize=(24, 12)
            )
            for i, fis in enumerate(fis_to_plot):
                for j, ia_pair in enumerate(ia_pairs_dct[fis][dataset].keys()):
                    ax = axes[i, j]
                    # get feature mapping
                    features_pretty = self.adjust_feature_names(
                        features=data_dct[fis][model]["features"],
                        fis=fis,
                        esm_sample=esm_sample,
                        plot="dependence_plot",
                    )
                    ia_pair_pretty = self.adjust_feature_names(
                        features=list(ia_pair),
                        fis=fis,
                        esm_sample=esm_sample,
                        plot="dependence_plot",
                    )
                    self.create_dependence_plot(
                        data=data_dct[fis][model],
                        features=features_pretty,
                        dataset=dataset,
                        ia_pair=ia_pair_pretty,
                        num_feature=j,
                        ax=ax,
                    )
            # Adjust layout and display
            plt.tight_layout(pad=3)
            if self.shap_config["plots"]["store_plots"]:
                self.store_plot(
                    plot_type="dependence_plot",
                    dataset=dataset,
                    esm_sample=esm_sample,
                    soc_int_var=soc_int_var,
                )
            else:
                plt.show()
                self.check_plot_grayscale(fig=fig, filename_raw=None, show_plot=True)

    def importance_plot_wrapper(self, data_dct):
        """
        This function creates and stores several shap importance plots on a root plot.
        The specifications how many importance plots are created and which are defined in the config.
        Currently, we plot the mean and the sum of the shap values summarized across categories per
        soc_int_var and sample (or only per sample for MSE), so that it results in one plot for one analysis
        (e.g., one plot for main/ssc with all ESM-samples and soc_int_vars)

        Args:
            data_dct: Dict, containing the raw SHAP values
        """
        for dataset in ["train", "test"]:
            data_agg_across_categories = self.summarize_abstraction_shap_for_plot(
                data_dct=data_dct, sum_across_models=True
            )
            fig = plt.figure(figsize=(24, 13))
            gs = gridspec.GridSpec(
                nrows=self.shap_config["plots"]["importance_plot"]["grid_spec"][
                    self.study
                ]["n_rows"],
                ncols=self.shap_config["plots"]["importance_plot"]["grid_spec"][
                    self.study
                ]["n_cols"],
            )
            axes = {}
            # create an importance plot for each ESM sample soc_int_var combination
            for esm_sample, esm_sample_vals in data_agg_across_categories.items():
                if self.study == "ssc":
                    ordered_soc_int_vars = [
                        i
                        for i in self.shap_config["plots"]["soc_int_var_order"]
                        if i in esm_sample_vals.keys()
                    ]
                elif self.study == "mse":
                    ordered_soc_int_vars = ["dummy"]
                else:
                    raise ValueError("Study must be ssc or mse")

                for soc_int_var in ordered_soc_int_vars:
                    for summary_stat in ["sum", "mean"]:
                        # Adjusted to access the nested dictionary structure
                        grid_position = (
                            self.shap_config["plots"]["importance_plot"][
                                "grid_positions"
                            ][self.study]
                            .get(esm_sample)
                            .get(soc_int_var)
                            .get(summary_stat)
                        )
                        print(grid_position)
                        if grid_position:
                            ax = fig.add_subplot(
                                gs[tuple(grid_position)]
                            )  # Convert list to tuple
                            axes[(esm_sample, soc_int_var, summary_stat)] = ax
                        else:
                            raise ValueError("grid position not available on Figure")
                        current_data = esm_sample_vals[soc_int_var]["shap_values"][
                            dataset
                        ]["abs_avg_across_cat"]
                        self.create_importance_plot(
                            data=current_data,
                            ax=ax,
                            shap_summary_type=summary_stat,
                            esm_sample=esm_sample,
                            soc_int_var=soc_int_var,
                        )

            # Apply some formatting
            plt.tight_layout(pad=2.5)
            plt.subplots_adjust(
                hspace=self.shap_config["plots"]["importance_plot"]["subplot_adjust"][
                    "hspace"
                ][self.study]
            )
            if self.study == "ssc":
                row_1_axes = [
                    ax
                    for (esm_sample, soc_int_var, metric), ax in axes.items()
                    if 0.25 < ax.get_position().y0 < 0.35
                ]
                row_2_axes = [
                    ax
                    for (esm_sample, soc_int_var, metric), ax in axes.items()
                    if 0.45 < ax.get_position().y0 < 0.55
                ]
                for ax in row_1_axes:
                    pos = ax.get_position()
                    ax.set_position([pos.x0, pos.y0 - 0.018, pos.width, pos.height])
                for ax in row_2_axes:
                    pos = ax.get_position()
                    ax.set_position([pos.x0, pos.y0 + 0.018, pos.width, pos.height])
            if self.shap_config["plots"]["store_plots"]:
                self.store_plot(
                    plot_type="importance_plot",
                    dataset=dataset,
                )
            else:
                plt.show()
                self.check_plot_grayscale(fig=fig, filename_raw=None, show_plot=True)

    def summarize_abstraction_shap_for_plot(self, data_dct, sum_across_models):
        """
        This function summarizes the SHAP values specifically for the importance plot. This results in one
        importance score for a broad feature category in one esm_sample - soc_int_var combination. Thus,
        importance scores for models and fis are pooled for this "global" importance.
        SHAP values for SVR are excluded, beucase the results were corrupted.

        Args:
            data_dct: Dict, containing the raw SHAP values
            sum_across_models: bool, determines if separate scores are calculated for separated models (if False)
                or it is pooled across models (if True)

        Returns:
            final_result: Dict, contains the structure that is plotted in the importance plot
        """
        # Processing the dictionary
        results = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
            )
        )

        for esm_sample, esm_sample_vals in data_dct.items():
            for soc_int_var, soc_int_var_vals in esm_sample_vals.items():
                for dataset in ["train", "test"]:
                    category_aggregates = defaultdict(list)

                    if sum_across_models:
                        for fis, fis_vals in soc_int_var_vals.items():
                            for model, model_vals in fis_vals.items():
                                if model == "svr":
                                    continue
                                abs_across_categories = model_vals["shap_values"][
                                    dataset
                                ]["shap_abstr_broad_categories"]

                                for category, values in abs_across_categories.items():
                                    category_aggregates[category].append(values)

                        for category, aggregates in category_aggregates.items():
                            results[esm_sample][soc_int_var]["shap_values"][dataset][
                                "abs_avg_across_cat"
                            ][category] = self.calculate_mean(aggregates)
                    else:
                        for fis, fis_vals in soc_int_var_vals.items():
                            for model, model_vals in fis_vals.items():
                                if model == "svr":
                                    continue
                                if (
                                    model
                                    in self.shap_config["plots"]["importance_plot"][
                                        "models_to_include"
                                    ]
                                ):
                                    abs_across_categories = model_vals["shap_values"][
                                        dataset
                                    ]["shap_abstr_broad_categories"]

                                    if (
                                        model
                                        not in results[esm_sample][soc_int_var][
                                            "shap_values"
                                        ][dataset]["abs_avg_across_cat"]
                                    ):
                                        results[esm_sample][soc_int_var]["shap_values"][
                                            dataset
                                        ]["abs_avg_across_cat"][model] = {}
                                    for (
                                        category,
                                        values,
                                    ) in abs_across_categories.items():
                                        if (
                                            category
                                            not in results[esm_sample][soc_int_var][
                                                "shap_values"
                                            ][dataset]["abs_avg_across_cat"][model]
                                        ):
                                            results[esm_sample][soc_int_var][
                                                "shap_values"
                                            ][dataset]["abs_avg_across_cat"][model][
                                                category
                                            ] = []
                                        results[esm_sample][soc_int_var]["shap_values"][
                                            dataset
                                        ]["abs_avg_across_cat"][model][category].append(
                                            values
                                        )

                        for model in results[esm_sample][soc_int_var]["shap_values"][
                            dataset
                        ]["abs_avg_across_cat"]:
                            if model == "svr":
                                continue
                            for category, aggregates in results[esm_sample][
                                soc_int_var
                            ]["shap_values"][dataset]["abs_avg_across_cat"][
                                model
                            ].items():
                                results[esm_sample][soc_int_var]["shap_values"][
                                    dataset
                                ]["abs_avg_across_cat"][model][
                                    category
                                ] = self.calculate_mean(
                                    aggregates
                                )

        final_result = {k: dict(v) for k, v in results.items()}
        return final_result

    @staticmethod
    def calculate_mean(shap_val_lst):
        """
        This function aggregates the mean and sum values of features across analysis settings (e.g., models)

        Args:
            shap_val_lst: Nested list where the first value of the inner list represents the sum of the summarized
                shap values and the second value represents the mean of the summarized shap values

        Returns:
            The average of both values across models and feature inclusion strategies
        """
        total_mean = sum(pair[0] for pair in shap_val_lst) / len(shap_val_lst)
        total_sum = sum(pair[1] for pair in shap_val_lst) / len(shap_val_lst)
        return [total_mean, total_sum]

    def structure_shap_values_for_plots(self, data, study):
        """
        Function that restructure the dct hierarchy for plotting multiple plots in one root plot

        Args:
            data: Dict, contains the SHAP values
            study: str, ssc or mse, processing depends on this

        Returns:
            reordered_dct: Dict, contains the same values as data, but reordered
        """
        dct = data.copy()
        reordered_dct = {}
        if study == "ssc":
            # Create a new dictionary with the reordered structure
            for esm_sample, esm_sample_vals in dct.items():
                for fis, fis_vals in esm_sample_vals.items():
                    for model, model_vals in fis_vals.items():
                        for soc_int_var, soc_int_var_vals in model_vals.items():
                            if esm_sample not in reordered_dct:
                                reordered_dct[esm_sample] = {}
                            if soc_int_var not in reordered_dct[esm_sample]:
                                reordered_dct[esm_sample][soc_int_var] = {}
                            if fis not in reordered_dct[esm_sample][soc_int_var]:
                                reordered_dct[esm_sample][soc_int_var][fis] = {}
                            reordered_dct[esm_sample][soc_int_var][fis][
                                model
                            ] = soc_int_var_vals
        elif study == "mse":
            for esm_sample, esm_sample_vals in dct.items():
                if esm_sample not in reordered_dct:
                    reordered_dct[esm_sample] = {}
                reordered_dct[esm_sample]["dummy"] = dct[esm_sample]
        else:
            raise ValueError("Unknown study")
        return reordered_dct

    def adjust_feature_names(self, features, fis, esm_sample, plot):
        """
        This function
            reformats the feature names according to the feature_mapping_yaml
            insert new lines to make the feature names presentable in plots

        Args:
            features: pd.DataFrame or list containing the features for the current dataset
            fis: Feature inclusion strategy
            esm_sample: ESM-sample
            plot: Specific plot, used for defining the optimal stringh length. Current options are
                  "summary_plot", "ia_heatmap",

        Returns:
            [df_pretty, cols]: pd.DataFrame or list with updated column names ready for paper plots
        """
        if isinstance(features, pd.DataFrame):
            cols = features.columns
        elif isinstance(features, list):
            cols = features
        mapping = self.feature_mapping.copy()
        current_mapping_dct = {}
        for feature in cols:
            current_mapping_dct[feature] = self.find_feature_name(
                feature=feature, mapping=mapping, fis=fis, esm_sample=esm_sample
            )
        if isinstance(features, pd.DataFrame):
            df_pretty = features.rename(columns=current_mapping_dct)
            cols = df_pretty.columns
        elif isinstance(features, list):
            feature_lst_pretty = [current_mapping_dct[feature] for feature in features]
            cols = feature_lst_pretty
        # Split long strings
        max_length = self.shap_config["plots"][plot]["strng_length"]
        cols = [self.split_long_string(col, max_length=max_length) for col in cols]
        if isinstance(features, pd.DataFrame):
            df_pretty.columns = cols
            return df_pretty
        elif isinstance(features, list):
            return cols

    def find_feature_name(self, feature, mapping, fis, esm_sample):
        """
        This function attempts to match a given feature as it is represented in the code and the same feature
        formatted for display in the paper plots.
        It searches for the feature in the nested dictionary (the feature_mapping.yaml) by exploring
        all branches until the match is obtained.

        Args:
            feature: str, a given feature name as it is in the code we search a matching category for
            mapping: Dict, basically a copy of the feature_mapping_yaml
            fis: str, feature inclusion strategy, used to skip irrelevant branches of the mapping.
            esm_sample: str, given ESM sample

        Returns:
            ['', value]: Either the correct match is returned, or an empty string
        """
        # process feature so that clean is removed
        feature_pretty = feature.replace("_clean", "")
        stack = [(mapping, iter(mapping.items()))]
        # traverse through the nested dict
        while stack:
            current_mapping, iterator = stack[-1]
            try:
                key, value = next(iterator)
                if (
                    key != fis
                    and isinstance(current_mapping, dict)
                    and fis in current_mapping
                ):
                    continue
                if key in feature_pretty:
                    if isinstance(value, dict):
                        if key not in list(value.keys())[0]:
                            try:
                                return value[esm_sample][feature_pretty]
                            except:
                                return value[esm_sample]
                        else:
                            stack.append((value, iter(value.items())))
                    else:
                        if feature_pretty == key:
                            return value
                else:
                    # Ensure that value is a dictionary before trying to iterate over it
                    if isinstance(value, dict):
                        stack.append((value, iter(value.items())))
            except StopIteration:
                # No more items in the current level, go back up
                stack.pop()
        return ""

    @staticmethod
    def get_top_features(data, dataset, feature_df, num=5):
        """
        This function extracts the most important features (based on the abs shap values).

        Args:
            data: Dict, containing the SHAP values
            dataset: str, "train" or "test"
            feature_df: df, containing the features for a given analysis
            num: int, determines how many of the top features are returned

        Returns:
            most_important_features: Ordered List of Dicts containing feture:importance pairs.
                The most important feature is the first list item."""
        importances = data["shap_values"][dataset]["abs_avg_across_reps_samples"]
        indices = np.argsort(importances)[-num:][::-1]
        top_importances = np.array(importances)[indices]
        top_features = feature_df.iloc[:, indices].columns
        most_important_features = [
            {feature: importance}
            for feature, importance in zip(top_features, top_importances)
        ]
        return most_important_features

    @staticmethod
    def split_long_string(string, max_length):
        """
        Split a string into multiple left-aligned lines if it exceeds max_length

        Args:
            string: str, the given string to split
            max_length: int, if the string is longer than max_lenght, we split it

        Returns:
            results: str, the formatted string that contains new lines for the line splitting
        """

        def split_at_index(s, index):
            """
            Helper function to split and strip at a given index.

            Args:
                s: str, the string to split
                index: int, the index of the string

            Returns:
                first_line: str, the first line of s
                remaining: str, the rest of s
            """
            first_line = s[:index].strip()
            remaining = s[index:].strip()
            return first_line, remaining

        lines = []
        while len(string) > max_length:
            # Find the last space before max_length
            split_index = string.rfind(" ", 0, max_length)
            if split_index == -1:
                # If no space found, force split at max_length
                split_index = max_length
            first_line, string = split_at_index(string, split_index)
            lines.append(first_line)
        # Add the remaining part of the string
        if string:
            lines.append(string)
        results = "\n".join(lines)
        return results

    def create_summary_plot(self, data, dataset, features, model, fis):
        """
        This function creates a shap summary plot / beeswarm plot. Currently, this is wrapped so that multiple
        summary plots are plotted together in one root plot

        Args:
            data: Dict, containing the SHAP values
            dataset: str, "train" or "test"
            features: df, containing the features for correct display of the feature values in the plot
            model: str, specifies the model on the plot
            fis: str, feature inclusion strategy
        """
        max_display_summary_plot = self.shap_config["plots"]["summary_plot"][
            "max_display"
        ]
        # Generate the SHAP summary plot
        shap.summary_plot(
            shap_values=np.array(data["shap_values"][dataset]["avg_across_reps"]),
            features=features,
            max_display=max_display_summary_plot,
            show=False,
            cmap=self.shap_config["plots"]["summary_plot"]["cmap"],
            plot_size=[12, 6],
        )
        # Get the current figure and axes objects and reformat
        fig, ax = plt.gcf(), plt.gca()
        ax.tick_params(axis="y", labelsize=5, pad=0, direction="inout")
        ax.tick_params(axis="x", labelsize=5)
        ax.set_xlabel("SHAP Value", fontsize=6)
        model_pretty = self.shap_config["plots"]["str_mapping"]["models"][model]
        fis_pretty = self.shap_config["plots"]["str_mapping"]["fis"][fis]
        ax.set_title(f"{fis_pretty} - {model_pretty}", fontsize=6)
        fig.axes[-1].tick_params(labelsize=5)
        fig.axes[-1].set_ylabel("Feature Value", fontsize=6)

    def create_importance_plot(
        self, data, ax, shap_summary_type, esm_sample, soc_int_var
    ):
        """
        This creates an importance plot. Currently, we use this only on the abstraction level of
        broad categories. This function plots one plot on the current ax.object that is given.

        Args:
            data: Dict, containing the SHAP values
            shap_summary_type: type of aggregation, "sum" or "mean"
            ax: matplotlib.ax Object, determines where the plot is placed on the root plot
            esm_sample: str, given ESM sample
            soc_int_var: str, given soc_int_var
        """
        if shap_summary_type == "sum":
            val_dct = {cat: vals[0] for cat, vals in data.items()}
        elif shap_summary_type == "mean":
            val_dct = {cat: vals[1] for cat, vals in data.items()}
        else:
            raise NotImplementedError("Choose either 'sum' or 'mean'")
        sorted_val_dct = dict(
            sorted(val_dct.items(), key=lambda item: item[1], reverse=True)
        )
        top_features = list(sorted_val_dct.keys())
        color_mapping = self.shap_config["plots"]["cat_color_mapping"]
        color = [color_mapping[feature] for feature in top_features]
        top_features_pretty = [
            self.shap_config["plots"]["category_mapping"][feature]
            for feature in top_features
        ]
        top_importances = list(sorted_val_dct.values())
        bar_height = 0.42

        # Creating the bar plot with different colors and adjusted spacing and format
        ax.barh(
            top_features_pretty,
            top_importances,
            align="center",
            height=bar_height,
            color=color,
        )
        if self.study == "ssc":
            soc_int_var_pretty = self.shap_config["plots"]["importance_plot"][
                "soc_int_var_mapping"
            ][soc_int_var]
            title_pretty = self.shap_config["plots"]["importance_plot"][
                "esm_sample_mapping"
            ][esm_sample]
            y_x_fontsize = 12
            title_fontsize = 14
        elif self.study == "mse":
            title_pretty = self.config["analysis"]["cv_results_plots_tables"]["plot"][
                "title_mapping"
            ]["mse"][esm_sample]
            y_x_fontsize = 16
            title_fontsize = 18
        ax.set_aspect(aspect="auto")
        ax.invert_yaxis()
        ax.set_yticklabels(top_features_pretty, fontsize=y_x_fontsize)
        ax.locator_params(axis="x", nbins=4)
        if shap_summary_type == "sum":
            ax.set_xlabel(
                "Sum of Importances per Category ",
                fontsize=y_x_fontsize,
                labelpad=8,
                fontweight="bold",
            )
        elif shap_summary_type == "mean":
            ax.set_xlabel(
                "Mean of Importances per Category ",
                fontsize=y_x_fontsize,
                labelpad=8,
                fontweight="bold",
            )
            if self.study == "ssc":
                ax.set_title(
                    f"{title_pretty} - {soc_int_var_pretty}",
                    fontsize=title_fontsize,
                    pad=15,
                    fontweight="bold",
                )
            elif self.study == "mse":
                ax.set_title(
                    f"{title_pretty}",
                    fontsize=title_fontsize,
                    pad=15,
                    fontweight="bold",
                )

    def create_dependence_plot(
        self, data, dataset, features, ax, ia_pair=None, num_feature=None
    ):
        """
        This function creates a dependency plot for a given pair of interacting features.
        If no pair is given, it takes the most important features and assign the most interactive feature
        automatically. Note: This is not used in the final paper and the final results and is not tested.
        It should work, though.

        Args:
            data: Dict, containing the SHAP values
            dataset: str, "train" or "test"
            features: df, containing the feature values for correct display in the plot
            ia_pair: list, containing two features that interact
            ax: matplotlib.ax object, determines where on the root plot the current config is plotted
        """
        if ia_pair:
            feature_1 = list(ia_pair)[0]
            feature_2 = list(ia_pair)[1]
            max_x_value = max(
                np.abs(features[feature_1].min()), features[feature_1].max()
            )
            xmin, xmax = -max_x_value, max_x_value
        else:
            top_feature_lst = self.get_top_features(data, dataset, features)
            feature_1 = list(top_feature_lst[num_feature].keys())[0]
            feature_2 = "auto"
            xmin, xmax = None, None
        shap.dependence_plot(
            ind=feature_1,
            shap_values=np.array(data["shap_values"][dataset]["avg_across_reps"]),
            features=features,
            interaction_index=feature_2,
            ax=ax,
            alpha=0.8,
            cmap=plt.get_cmap(self.shap_config["plots"]["dependence_plot"]["cmap"]),
            show=False,
            xmin=xmin,
            xmax=xmax,
        )

    def store_plot(
        self,
        plot_type,
        dataset=None,
        soc_int_var=None,
        esm_sample=None,
        fis=None,
        name_suffix=None,
    ):
        """
        This function is a generic method used to store various SHAP plots in colour.

        Args:
            plot_type: str, type specifying the plot (e.g., "summary_plot")
            dataset: [str, None]: if str, "train" or "test", if None, than the plot contains content of train
                and test together
            soc_int_var: [str,None], soc_int_var if ssc or None is mse
            esm_sample: str, the given ESM sample
            fis: str, feature inclusion strategy
            name_suffix: str, given suffix for more adequately specifyin the result plot
        """

        filename = plot_type
        if fis:
            filename += f"_{fis}"
        if dataset:
            filename += f"_{dataset}"
        if name_suffix:
            filename += f"_{name_suffix}"
        if esm_sample:
            current_plot_path = os.path.join(self.plot_path, esm_sample)
        else:
            current_plot_path = self.plot_path
        if soc_int_var:
            current_plot_path = os.path.join(self.plot_path, esm_sample, soc_int_var)
        os.makedirs(current_plot_path, exist_ok=True)
        file_path = os.path.join(current_plot_path, filename)
        if self.shap_config["plots"]["filetype"] == "jpeg":
            filename_jpeg = file_path + ".jpg"
            plt.savefig(
                filename_jpeg, dpi=500, bbox_inches="tight"
            )  # 600 DPI, adjust for some Heatmaps
            self.check_plot_grayscale(
                fig=None, filename_raw=file_path, show_plot=False, data_ending=".jpg"
            )
            plt.close()

    @staticmethod
    def check_plot_grayscale(
        fig=None, filename_raw=None, show_plot=True, data_ending=".jpg"
    ):
        Image.MAX_IMAGE_PIXELS = 500000000
        """
        This function checks how a certain plot looks if it would be printed in grayscale (as it would be in 
        the printed version of a journal). 
        
        Args:
            fig: [None, Figure], Matplolib Figure object (then we store it and then do the conversion) or None 
                (then we load the Figure and to the conversion)
            filename_raw: [None, str], if str, we load the Figure and then do the conversion
            show_plot: bool, if True, plot will be shown
            data_ending: str, determines the format of the output file  
        """
        if show_plot:  # Convert the Matploblib Figure
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            img = Image.open(buf).convert("L")
        else:
            img = Image.open(filename_raw + data_ending).convert("L")
        dpi = 500  # 600
        figsize = (img.width / dpi, img.height / dpi)
        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(np.array(img), cmap="gray")
        plt.axis("off")
        if filename_raw:
            new_filename = f"{filename_raw}_grayscale" + data_ending
            plt.savefig(
                new_filename
            )  # just for comparing the colors -> will not be used
        if show_plot:
            plt.show()
