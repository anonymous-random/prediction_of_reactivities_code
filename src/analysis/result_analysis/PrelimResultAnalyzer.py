import json
import os
import pickle
from pathlib import Path, PurePath

import numpy as np
import openpyxl
import pandas as pd
import rpy2.robjects as robjects
import yaml
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from scipy.stats import pearsonr


class PrelimResultAnalyzer:
    """
    This class is used to summarize and analyze descriptive statistics (e.g., M, SD, correlations)
    for a given ESM sample and a given analysis.
    Which specific preliminary analysis are relevant for which analysis type differs, because
    sometimes the analyses are based on the same data (e.g., for the supplementary analysis
    "weighting_by_rel", the descriptives are identical to the main analysis, the weighting
    comes only into play in the machine learning-based analysis).

    Attributes:
        config: YAML config determining certain specifications of the analysis.
        esm_sample: A given ESM-sample for which the descriptives are computed.
        _data_base_path: Root path were the processed data for all analyses is stored.
        _study: The analysis study, either SSC or MSE.
    """

    def __init__(
        self,
        config_path,
        esm_sample,
    ):
        """
        Constructor method of the PrelimResultAnalyzer Class.

        Args:
            config_path: Path to the .YAML config file.
            esm_sample: A given ESM-sample for which the descriptives are computed.
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.esm_sample = esm_sample
        self._data_base_path = self.config["general"]["load_data"][
            "processed_data_path"
        ]
        self._study = self.config["general"]["study"]

    @property
    def esm_sample(self):
        """The esm_sample property."""
        return self._esm_sample

    @esm_sample.setter
    def esm_sample(self, value):
        """Set the esm_sample, ensuring it's in the list of used ESM samples in the config."""
        if value not in self.config["general"]["samples_for_analysis"]:
            raise ValueError(
                f"esm_sample must be one of {self.config['samples_for_analysis']}"
            )
        self._esm_sample = value

    @property
    def data_base_path(self):
        """The data_base_path property getter."""
        return self._data_base_path

    @data_base_path.setter
    def data_base_path(self, value):
        """Set data_base_path, ensure that the path exists."""
        if not os.path.exists(value):
            raise ValueError(f"The provided path does not exist: {value}")
        self._data_base_path = value

    @property
    def study(self):
        """The data_base_path property getter."""
        return self._study

    @study.setter
    def study(self, value):
        """Set data_base_path, ensure that the path exists."""
        if value not in ["ssc", "mse"]:
            raise ValueError(f"Unknown study: {value}, choose 'mse' or 'ssc'")
        self._study = value

    @property
    def person_id_col(self):
        """Name of the person id column in the ESM dataset."""
        return self.config["general"]["id_col"]["all"]

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
    def suppl_var(self):
        """Var of supplementary analysis, only defined if self.suppl_type exists, e.g. 'ftf'."""
        return (
            self.config["general"]["suppl_var"]
            if self.suppl_type and self.suppl_type != "add_wb_change"
            else None
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
            if self.suppl_type in ["sep_ftf_cmc", "sep_pa_na", "add_wb_change"]
            else "main"
        )

    @property
    def suppl_type_level_path(self):
        """
        Used for loading the requested data efficiently. This is not equivalent to 'analysis', because
        for the supplementary analysis 'weighting_by_rel', we use the same data as in the main analysis up to
        the Multilevel Modeling.
        """
        return self.suppl_type if self.suppl_type != "weighting_by_rel" else None

    @property
    def suppl_var_level_path(self):
        """
        Used for loading the requested data efficiently. This is not equivalent to 'analysis', because
        for the supplementary analysis 'weighting_by_rel', we use the same data as an in main analysis up to
        the Multilevel Modeling.
        """
        return self.suppl_var if self.suppl_type != "weighting_by_rel" else None

    @property
    def soc_int_vars(self):
        """
        This sets the social interaction variables for the current analysis setting as a property.
        The assignment is based on the config and certain sample specifics (i.e., in CoCo UT interaction
        quantity was only assessed for face-to-face interactions, therefore it is added in this type
        of supplementary analysis. If self.study == 'mse', the property is irrelevant.
        """
        soc_int_vars = [
            soc_int_var
            for soc_int_var, values in self.config["state_data"]["ssc"][
                "social_interaction_vars"
            ].items()
            if self.esm_sample in values["samples"] and self.study == "ssc"
        ]
        if self.esm_sample == "coco_ut" and self.suppl_var in ["ftf", "ftf_pa"]:
            soc_int_vars.append("interaction_quantity")
        return soc_int_vars

    @property
    def wb_items(self):
        """
        This sets the well-being items for a given ESM-sample as a property based on the config.
        It further differentiates positive and negative affect which is reflected in the Dict hierarchy.
        """
        return [
            wb_item
            for affect_type in self.config["state_data"]["general"]["well_being_items"][
                self.esm_sample
            ].keys()
            if affect_type in ["positive_affect", "negative_affect"]
            for wb_item in self.config["state_data"]["general"]["well_being_items"][
                self.esm_sample
            ][affect_type]
        ]

    @property
    def data_path_study(self):
        """
        Creates the data_path for the descriptive calculations according to the study, analysis and ESM-sample
        up to the degree of the correct study. An example path would be data/preprocessed/sep_ftf_cmc/ftf/ssc.
        path components that are None will be filtered out, so that the correct path depending on the specific
        analysis is returned.
        """
        path_components = [
            self.data_base_path,
            self.analysis_level_path,
            self.suppl_type_level_path,
            self.suppl_var_level_path,
            self.study,
        ]
        filtered_path_components = [comp for comp in path_components if comp]
        return os.path.normpath(os.path.join(*filtered_path_components))

    @property
    def store_base_path(self):
        """Base path were the preliminary results are stored."""
        return self.config["analysis"]["descriptive_results"]["store_base_path"]

    @property
    def traits(self):
        """Set the person-level variables for both feature inclusion strategies as a property."""
        trait_path = os.path.join(self.data_path_study, "traits")
        if not os.path.exists(trait_path):
            raise ValueError(f"The provided path does not exist: {trait_path}")
        trait_dct = dict()
        for feature_inclusion_strategy in ["single_items", "scale_means"]:
            specific_trait_path = os.path.join(trait_path, feature_inclusion_strategy)
            print(specific_trait_path)
            trait_dct[feature_inclusion_strategy] = self.load_files(
                specific_trait_path, self.esm_sample, "traits"
            )
        return trait_dct

    @property
    def states(self):
        """Set the state data containing wb_items, social_situation variables, etc. as property."""
        state_path = os.path.join(self.data_path_study, "states")
        if not os.path.exists(state_path):
            raise ValueError(f"The provided path does not exist: {state_path}")
        state_dct = self.load_files(state_path, self.esm_sample, "states")
        return state_dct

    @property
    def random_effects(self):
        """Set the random effects used as criterion in the machine learning analyses as properties"""
        random_slopes_path = os.path.join(self.data_path_study, "random_effects")
        if not os.path.exists(random_slopes_path):
            raise ValueError(f"The provided path does not exist: {random_slopes_path}")
        random_slopes_dct = self.load_files(
            random_slopes_path, self.esm_sample, "random_effects"
        )
        return random_slopes_dct

    def apply_methods(self):
        """This function applies the preprocessing methods specified in the config."""
        for method in self.config["analysis"]["descriptive_results"]["methods"]:
            if method not in dir(PrelimResultAnalyzer):
                raise ValueError(f"Method '{method}' is not implemented yet.")
            getattr(self, method)()

    def set_duplicate_result_warning(self):
        """
        This is used to indicate that the results for the supplementary analysis 'weighting_by_rel' and
        main analysis will be identical. Then the analysis would be irrelevant.
        """
        if self.suppl_type == "weighting_by_rel":
            print("### preliminary results are the same as for the main analysis ###")

    def calc_ssc_descriptives(self):
        """
        This function calculate descriptives (M, SD, correlations) for the social situation variables in a given
        ESM-sample. Correlations are only computed for CoCo International (because other social interaction variables
        are only assessed if they engaged in a social_interaction at all, so computing a correlation between e.g.
        interaction_quantity and social_interaction makes no sense.
        Further, the data basis for the 3 other social situation variables in CoCo International is identical
        (interaction quantity, interaction closeness, interaction depth). Therefore, no separate preliminary analysis
        for these variables need to be computed.
        """
        if self.study == "ssc":
            result_dct = dict()
            states = self.states.copy()
            soc_int_vars = self.soc_int_vars.copy()
            # calc correlations
            if self.esm_sample == "coco_int":
                df = states["interaction_depth"]
                within_person_df, between_person_df = self.calc_bp_wp_correlations(
                    df, soc_int_vars, self.person_id_col
                )
                result_dct["within_person_correlation"] = within_person_df.round(
                    2
                ).to_dict()
                result_dct["between_person_correlation"] = between_person_df.round(
                    2
                ).to_dict()
            # Calc means, sds
            for soc_int_var in soc_int_vars:
                result_dct[soc_int_var] = dict()
                df = states[soc_int_var]
                result_dct[soc_int_var]["mean"] = np.round(df[soc_int_var].mean(), 2)
                result_dct[soc_int_var]["sd"] = np.round(df[soc_int_var].std(), 2)
            if self.config["analysis"]["descriptive_results"]["store_results"]:
                self.store_results(result_dct, "soc_int_vars", "json")

    def calc_wb_items_descriptives(self):
        """
        This function calculates the descriptives for the wb_items.
        Note: In SSC, we potentially have different data sources for the different social interaction variables
        (e.g., the data basis for social interaction differs from the data basis for interaction quantity,
        because the first includes all esm-prompts (social interaction yes and no), whereas the second
        includes only the esm-prompts where the participants have engaged in a social interaction.
        Therefore, we add the social situation variable as another nesting level in the result dict for ssc.
        In the supplement, we always report all descriptive tables, if the data basis for different social
        interaction variables differ.
        """
        result_dct = dict()
        states = self.states.copy()
        wb_items = self.wb_items.copy()
        data_vars = self.soc_int_vars.copy()

        # If study == "mse", there are no soc_int_vars. Including a dummy maintains the same structure for mse and ssc
        if self.study == "mse":
            data_vars.append("dummy")
        for soc_int_var in data_vars:
            result_dct[soc_int_var] = dict()
            if self.study == "ssc":
                df = states[soc_int_var]
            elif self.study == "mse":
                df = states[
                    self.config["analysis"]["result_analysis"]["mse_assignment"][
                        self.esm_sample
                    ]
                ]
            else:
                raise ValueError("Study must be ssc or mse")
            # get correlations
            within_person_df, between_person_df = self.calc_bp_wp_correlations(
                df, wb_items, self.person_id_col
            )
            result_dct[soc_int_var][
                "within_person_correlation"
            ] = within_person_df.round(2).to_dict()
            result_dct[soc_int_var][
                "between_person_correlation"
            ] = between_person_df.round(2).to_dict()
            # get means, sds
            for wb_item in wb_items:
                result_dct[soc_int_var][wb_item] = dict()
                result_dct[soc_int_var][wb_item]["mean"] = np.round(
                    df[wb_item].mean(), 2
                )
                result_dct[soc_int_var][wb_item]["sd"] = np.round(df[wb_item].std(), 2)
            # store corr tables as xlsx files
            if self.config["analysis"]["descriptive_results"]["store_results"]:
                within_person_df = within_person_df.round(2)
                self.store_corr_tables(
                    within_person_df, "wb_items", "within_corr", soc_int_var
                )
                between_person_df = between_person_df.round(2)
                self.store_corr_tables(
                    between_person_df, "wb_items", "between_corr", soc_int_var
                )
        # store all descriptives (including corr tables) as json files
        if self.config["analysis"]["descriptive_results"]["store_results"]:
            self.store_results(result_dct, "wb_items", "json")

    def calc_random_effects_descriptives(self):
        """
        This function calculates the descriptives for the random effect coefficients.
        Note: Mean should be zero by assumption, instead of the SD we reported the random slopes
        variance from the multilevel models in the paper.
        Therefore, here we only analyzed the between-person correlations of the individual random slopes.
        """
        if self.study == "ssc":
            result_dct = dict()
            random_effects = self.random_effects.copy()
            soc_int_vars = self.soc_int_vars.copy()
            # Calculated pairwise correlations
            for i, var1 in enumerate(soc_int_vars):
                for var2 in soc_int_vars[i + 1:]:
                    aligned_series1, aligned_series2 = random_effects[var1].align(
                        random_effects[var2], join="inner"
                    )
                    correlation, p_value = pearsonr(aligned_series1, aligned_series2)
                    result_dct[f"{var1}_{var2}"] = {
                        "correlation": np.round(correlation, 3),
                        "p_value": np.round(p_value, 6),
                    }
            # Store re correlations
            if self.config["analysis"]["descriptive_results"]["store_results"]:
                self.store_results(result_dct, "random_effects_corr", "json")

    @staticmethod
    def calc_bp_wp_correlations(df, variables_to_correlate, id_col):
        """
        This function calculates within and between-person correlations for given variables using the common
        R package ("psych"). To do so, we use the package py2rpy which enables using R in Python.

        Args:
            df: pd.DataFrame, data basis for calculating the correlations, e.g. state_df of coco_int
            variables_to_correlate: list, contains all variables that should be correlated, e.g. wb-items in coco_int
            id_col: column containing the person-id of df

        Returns:
            rwg_df: pd.DataFrame, containing the within-person correlations for variables_to_correlate
            rbg_df: pd.DataFrame, containing the between-person correlations for variables_to_correlate
        """
        # R set-up
        pandas2ri.activate()
        try:
            psych = importr("psych")
        except:  # if someone runs the code and it is not installed
            utils = importr("utils")  # TODO: test
            utils.install_packages("psych")
            psych = importr("psych")
        robjects.r('Sys.setlocale("LC_ALL", "C")')  # for encoding compatibility
        # Data set-up
        vars_to_corr = variables_to_correlate.copy()
        if "social_interaction" in variables_to_correlate:
            vars_to_corr.remove("social_interaction")
        df = df[vars_to_corr + [id_col]]
        # Use R to calculate correlations and create dataframes
        r_data = pandas2ri.py2rpy(df)
        r_stats_by = psych.statsBy(r_data, group=robjects.StrVector([id_col]))
        rwg_array = r_stats_by.rx2("rwg")
        rbg_array = r_stats_by.rx2("rbg")
        rwg_df = pd.DataFrame(rwg_array, columns=vars_to_corr)
        rbg_df = pd.DataFrame(rbg_array, columns=vars_to_corr)
        return rwg_df, rbg_df

    def calc_ml_features_descriptives(self):
        """
        This function calculates descrptives of the features used in the machine learning analyses, which
        includes socio-demographics, personality variables, political and societal attitudes, and country-level
        variables.
        For continuous variables, it computes the mean and the sd.
        For categorical variables, it computes frequencies (absolute + proportions).
        Note: In SSC, we potentially have different data sources for the different social interaction variables
        (e.g., the data basis for social interaction differs from the data basis for interaction quantity,
        because the first includes all esm-prompts (social interaction yes and no), whereas the second
        includes only the esm-prompts where the participants have engaged in a social interaction.)
        Therefore, we add the social situation variable as another nesting level in the result dict for ssc.
        In the supplement, we always report all descriptive tables, if the data basis for different social
        interaction variables differ.
        Here, we should output the results as a xlsx file.
        """
        result_dct = dict()
        traits = self.traits.copy()
        data_vars = self.soc_int_vars.copy()

        # If study == "mse", there are no soc_int_vars. Including a dummy maintains the same structure for mse and ssc
        if self.study == "mse":  # TODO test
            data_vars.append("dummy")
        for soc_int_var in data_vars:
            result_dct[soc_int_var] = dict()
            for fis in ["scale_means", "single_items"]:
                result_dct[soc_int_var][fis] = dict()
                # iterate over var types to get single tables per broad feature category
                for trait_type in self.config["trait_data"]["broad_categories"]:
                    trait_type_cfg = self.config["trait_data"][trait_type]
                    result_dct[soc_int_var][fis][trait_type] = dict()
                    if self.study == "ssc":
                        df = traits[fis][soc_int_var]
                    else:  # mse
                        df = traits[fis][
                            self.config["analysis"]["result_analysis"][
                                "mse_assignment"
                            ][self.esm_sample]
                        ]
                    df.columns = df.columns.str.replace("_clean", "")

                    # Get variable names for each trait category for the given ESM Sample and filter df
                    if trait_type == "country_vars":
                        yaml_names = [
                            var_name
                            for item in trait_type_cfg
                            for var_name in item["var_names"]
                        ]
                    else:
                        yaml_names = {
                            var["name"]
                            for var in trait_type_cfg
                            if self.esm_sample in var.get("time_of_assessment", {})
                        }
                    cols = [
                        col
                        for col in df.columns
                        if any(yaml_name in col for yaml_name in yaml_names)
                    ]
                    df = df[cols]

                    # separate binary and non-binary cols (all categoricals are one-hot-encoded in df)
                    binary_cols = df.columns[(df.isin([0, 1])).all(axis=0)]
                    non_binary_cols = df.columns.difference(binary_cols)
                    df_binaries = df[binary_cols]
                    df_continuous = df[non_binary_cols]

                    # Process continuous variables (M / SD)
                    if len(df_continuous.columns) > 1:
                        df_continuous_stats = df_continuous.agg(
                            ["mean", "std"]
                        ).transpose()
                        df_continuous_stats.columns = ["Mean", "SD"]
                        df_continuous_stats = df_continuous_stats.round(1)
                        result_dct[soc_int_var][fis][trait_type][
                            "conts"
                        ] = df_continuous_stats

                    # Process categorical variables (frequencies per category, absolute and %)
                    # This includes some custom processing to assign the categories correctly
                    if len(df_binaries.columns) > 1:
                        df_categorical_freq = df_binaries.sum()
                        df_categorical_freq.index = pd.MultiIndex.from_tuples(
                            [
                                tuple(index.rsplit("_", 1))
                                for index in df_categorical_freq.index
                            ]
                        )
                        df_categorical_freq = df_categorical_freq.unstack()
                        sorted_columns = sorted(
                            df_categorical_freq.columns, key=lambda x: int(x)
                        )
                        df_categorical_freq = df_categorical_freq[sorted_columns]
                        row_sums = df_categorical_freq.sum(axis=1)
                        df_categorical_proportions = df_categorical_freq.div(
                            row_sums, axis=0
                        )
                        # Format for frequencies for one category: "absolute (proportion)"
                        df_combined = (
                            df_categorical_freq.astype(str)
                            + " ("
                            + df_categorical_proportions.round(3).astype(str)
                            + ")"
                        )
                        result_dct[soc_int_var][fis][trait_type]["cats"] = df_combined

        # Store results as xlsx file
        if self.config["analysis"]["descriptive_results"]["store_results"]:
            self.store_results(result_dct, "ml_features", file_type="excel")

    def store_corr_tables(self, df, folder, name, soc_int_var=None):
        """
        This function stores a given df (which is a correlation table, in this case) to a given folder as a xlsx file.

        Args:
            df: pd.DataFrame, in principle any df, is a correlation table, in this case.
            folder: str, one folder in the hierarchy that identifies for which variables the corrs were computed.
            name: str, denotes which type of correlation, either "within_corr" or "between_corr".
            soc_int_var: [str, None], If defined, will be part of the stored file name.
        """
        path_components = self.get_path_components(self.data_path_study)
        dir_path = os.path.join(
            self.store_base_path,
            folder,
            *path_components[3:],
            self.esm_sample,
        )
        os.makedirs(dir_path, exist_ok=True)
        if soc_int_var:
            file_name = f"{soc_int_var}_{name}.xlsx"
        else:
            file_name = f"{name}.xlsx"
        file_path = os.path.join(dir_path, file_name)
        df.to_excel(file_path)

    def store_results(self, result_dct, folder, file_type=None):
        """
        This is a general function that stores certain descriptive results in a given folder.
        Thereby, it combines the store_base_path (were all descriptive results are stored) with
        the variable, for which the descriptives are calculated (var "folder", e.g. "wb_items") with
        the filename that is defined in this function.

        Args:
            result_dct: Dict, containing certain descriptive results
            folder: str, one folder that identifies for which variables the descriptives were computed
            file_type: str, defining how the content from result_dct is stored, either "json" or "excel".
        """
        path_components = self.get_path_components(self.data_path_study)
        # Most information are in the filename, stored in the same directory
        if file_type == "json":
            file_name = (
                "_".join(part for part in path_components[3:])
                + f"_{self.esm_sample}.json"
            )
            file_path = os.path.join(self.store_base_path, folder, file_name)
            with open(file_path, "w") as f:
                json.dump(result_dct, f, indent=4)
        # The construction of the dir_path is more fine-grained here, e.g. for the ml features
        elif file_type == "excel":
            for soc_int_var, values in result_dct.items():
                for fis, values_2 in values.items():
                    for trait_type, values_3 in values_2.items():
                        # Construct the directory path starting with each component in path_components
                        dir_path = os.path.join(
                            self.store_base_path,
                            folder,
                            *path_components[3:],
                            self.esm_sample,
                            soc_int_var,
                            fis,
                        )
                        os.makedirs(dir_path, exist_ok=True)
                        for var_type, df in values_3.items():
                            file_name = f"{trait_type}_{var_type}_.xlsx"
                            file_path = os.path.join(dir_path, file_name)
                            df.to_excel(file_path)
                            print("saved excel")
        else:
            raise ValueError("file type not implemented")

    @staticmethod
    def get_path_components(path_str):
        """
        This function returns the path components from a given path provided as a string.

        Args:
            path_str: str, a given path, e.g. '/usr/bin/python3'

        Returns: PurePath(path_str).parts: tuple, individual path components, e.g. ('/', 'usr', 'bin', 'python3')
        """
        return PurePath(path_str).parts

    @staticmethod
    def load_files(parent_path, esm_sample, var_type):
        """
        This function is used to load the data for a given var_type that will be used to calculated descriptives.
        It does so by creating the right path and filename based on the given esm_sample, parent_path, and var_type.

        Args:
            parent_path: str, Path were the specified var_type is stored (e.g., traits)
            esm_sample: str, given esm-sample
            var_type: str, the var_type that is loaded, either "traits", "states", or "random_effects"

        Returns:
            data_dct: Dict, containing the loaded data.
        """
        data_dct = dict()
        data_path = Path(parent_path)
        # depending on var_type, general naming patterns between files differ
        if var_type == "traits":
            post_sample_pattern = "one_hot"
        elif var_type == "states":
            post_sample_pattern = "preprocessed"
        elif var_type == "random_effects":
            post_sample_pattern = ""
        else:
            raise ValueError("Please choose traits, states or random_effects")
        # Iterate through files in current folder
        for file_path in data_path.glob(pattern=f"{esm_sample}*"):
            filename = file_path.stem
            # Isolate the part of the filename after the esm_sample
            post_sample_filename = filename[
                len(esm_sample) + 1:
            ]  # +1 because of the underscore
            var_end_index = post_sample_filename.index(post_sample_pattern)
            # Extract the variable name
            if var_type == "random_effects":
                var = post_sample_filename
            else:
                var = post_sample_filename[
                    : var_end_index - 1
                ]  # -1 to remove the trailing underscore
            # Load the pickled file
            with file_path.open("rb") as file:
                data = pickle.load(file)
                data_dct[var] = data
                print(f"loaded {filename}")
        return data_dct
