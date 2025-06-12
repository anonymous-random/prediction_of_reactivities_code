import copy
import itertools
import json
import logging
import math
import os
import pickle
import warnings
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import pandas as pd
import pingouin as pg
import yaml
from pandas.api.types import is_string_dtype
from sklearn.preprocessing import MultiLabelBinarizer


class BasePreprocessor(ABC):
    """
    Abstract base class for data preprocessing. This class serves as a template for specific
    preprocessor implementations such as PreprocessorSSC. It encapsulates
    methods and attributes used across multiple preprocessor variants, e.g. sth like inverse coding of items.

    Attributes:
        config: YAML config providing details on the preprocessing logic
        sample_dct: Dictionary that is passed to the preprocessor containing trait and state df of the ESM samples
        study: Study 1 ("ssc"), defined by the corresponding subclass
        config_path: str, relative path to the config file
        trait_dct: Dictionary containing the raw traits as str:df pairs at the beginning (keys are e.g. "coco_int")
            and the processed traits as str:df pairs at the end (keys are e.g. "coco_int_social_interaction")
        state_dct: Dictionary containing the raw traits as str:df pairs at the beginning (keys are e.g. "coco_int")
            and the processed states as str:df pairs at the end (keys are e.g. "coco_int_social_interaction")
        proc_esm_sample_lst: Lst of processed sample names, that structured like this:
            (f'{esm_sample}_{soc_int_var}')
    """

    @abstractmethod
    def __init__(
        self,
        config_path,
        sample_dct,
        study,
    ):
        """
        Constructor method of the BasePreprocessor Class.

        Args:
            config_path: str, relative path to the config file
            sample_dct: Dictionary that is passed to the preprocessor containing trait and state df of the ESM samples
            study: Study 1 ("ssc"), defined by the corresponding subclass
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.sample_dct = sample_dct
        self.study = study  # Choose in inherited methods
        self.config_path = config_path
        self._setup_logger()

        self.trait_dct = dict()
        self.state_dct = dict()
        self.proc_esm_sample_lst = list()
        for sample_name in self.config["general"]["samples_for_analysis"]:
            self.trait_dct[sample_name] = self.sample_dct[sample_name][0]
            self.state_dct[sample_name] = self.sample_dct[sample_name][1]

    @property
    def trait_cfg(self):
        """Part of the global config that concern the processing of the traits."""
        return self.config["trait_data"]

    @property
    def state_cfg(self):
        """Part of the global config that concern the processing of the states."""
        return self.config["state_data"]

    @property
    def samples_for_analysis(self):
        """ESM-samples that are processed (e.g., ["coco_int", "emotions", "coco_ut"])."""
        return self.config["general"]["samples_for_analysis"]

    @property
    def person_id_col(self):
        """Name of the person id column in the ESM samples (currently, it is "id" in all samples)."""
        return self.config["general"]["id_col"]["all"]

    @property
    def analysis(self):
        """Type of analysis, can either be "main" or "suppl"."""
        return self.config["general"]["analysis"]

    @property
    def suppl_type(self):
        """
        Type of supplementary analysis, only defined if self.analysis == 'suppl', e.g. 'sep_ftf_cmc'.
        It checks if the given study / suppl_type combo makes sense and raises ValueError otherwise.
        """
        if self.analysis == "suppl":
            suppl_type = self.config["general"]["suppl_type"]
            suppl_type_mapping = self.config["general"].get("suppl_type_mapping", {})
            if suppl_type in suppl_type_mapping[self.study]:
                return suppl_type
            else:
                raise ValueError(
                    f"Suppl Type {suppl_type} not compatible with study {self.study}"
                )
        elif self.analysis == "main":
            return None
        else:
            raise ValueError("analysis must be main or suppl")

    @property
    def suppl_var(self):
        """
        Var of supplementary analysis, only defined if self.suppl_type exists, e.g. 'ftf'.
        It checks if the given suppl_type / suppl_var combo makes sense and raises ValueError otherwise.
        """
        suppl_var = self.config["general"]["suppl_var"]
        suppl_var_mapping = self.config["general"].get("suppl_var_mapping", {})
        if self.suppl_type in suppl_var_mapping.keys():
            valid_vars = suppl_var_mapping.get(self.suppl_type, [])
            if suppl_var in valid_vars:
                return suppl_var
            else:
                raise ValueError(
                    f"for {self.suppl_type} you need to specify one of {valid_vars}"
                )
        else:
            return None

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
            in ["sep_ftf_cmc", "sep_pa_na"]
            else "main"
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
    def suppl_var_level_path(self):
        """
        Used for loading the requested data efficiently. This is not equivalent to 'analysis', because
        for the supplementary analysis 'weighting_by_rel', we use the same data as an in main analysis up to
        the Multilevel Modeling.
        """
        return self.suppl_var if self.suppl_type != "weighting_by_rel" else None

    @property
    def store_base_path(self):
        """Root dir where the processed data is stored."""
        return self.config["general"]["store_data"]["data_path"]

    @property
    def store_path(self):
        """
        The specific path where the processed data is stored. Depending on the analysis (main/suppl), suppl_type,
        study, etc...
        An example would be store_base_path/main/ssc.
        """
        path_components = [
            self.store_base_path,
            self.analysis_level_path,
            self.suppl_type_level_path,
            self.suppl_var_level_path,
            self.study,
        ]
        # Filter out empty or None values for dynamic path length
        filtered_path_components = [comp for comp in path_components if comp]
        return os.path.normpath(os.path.join(*filtered_path_components))

    @property
    def wb_items(self):
        """Sets a Dict with esm_samples and corresponding wb_items as class attribute."""
        wb_item_dct = dict()
        for esm_sample in self.samples_for_analysis:
            wb_item_dct[esm_sample] = [
                wb_item
                for affect_type in self.config["state_data"]["general"][
                    "well_being_items"
                ][esm_sample].keys()
                if affect_type in ["positive_affect", "negative_affect"]
                for wb_item in self.config["state_data"]["general"]["well_being_items"][
                    esm_sample
                ][affect_type]
            ]
        return wb_item_dct

    @property
    def pa_items(self):
        """Sets a Dict with esm_samples and corresponding items assessing positive affect as class attribute."""
        pa_item_dct = dict()
        for esm_sample in self.samples_for_analysis:
            pa_item_dct[esm_sample] = self.state_cfg["general"]["well_being_items"][
                esm_sample
            ]["positive_affect"]
        return pa_item_dct

    @property
    def na_items(self):
        """Sets a Dict with esm_samples and corresponding items assessing negative affect as class attribute."""
        na_item_dct = dict()
        for esm_sample in self.samples_for_analysis:
            na_item_dct[esm_sample] = self.state_cfg["general"]["well_being_items"][
                esm_sample
            ]["negative_affect"]
        return na_item_dct

    @property
    def wb_items_scale_endpoints(self):
        """
        Sets a Dct containing the scale endpoints of the well-being items per ESM samples.
        Dct values are lists with two entries, the first is the minimum of the scale, the second is the maximum.
        """
        scale_endpoint_dct = dict()
        for esm_sample in self.samples_for_analysis:
            scale_endpoint_dct[esm_sample] = self.state_cfg["general"][
                "well_being_items"
            ][esm_sample]["scale_endpoints"]
        return scale_endpoint_dct

    @property
    def logs_base_path(self):
        """Base folder for logging, same for subclasses."""
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "logs")

    @property
    def logs_specific_path(self):
        """
        Specific folder for logging, depending on the subclass and the log_name specified in config.
        Because the processing is identical for 'main' and 'weighting_by_rel', we do not create a
        individual log path for the weighting_by_rel supplementary analysis.
        """
        path_components = [
            self.logs_base_path,
            self.analysis_level_path,
            self.suppl_type_level_path,
            self.suppl_var_level_path,
            self.study,
            self.config["general"]["log_name"],
        ]
        # Filter out empty or None values
        filtered_path_components = [comp for comp in path_components if comp]
        return os.path.normpath(os.path.join(*filtered_path_components))

    @property
    def log_filename(self):
        """This creates the filename of the logfile based on the Attr "log_specific_path" and the current time."""
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        return os.path.join(
            self.logs_specific_path, f"preprocessing_{current_time}.log"
        )

    @property
    @abstractmethod
    def preprocessing_methods(self):
        """Exact preprocessing methods that are carried out, subclass-specific."""
        pass

    @property
    @abstractmethod
    def soc_dem_dct(self):
        """Implement this in the subclasses."""
        pass

    @property
    @abstractmethod
    def pers_dct(self):
        """Implement this in the subclasses."""
        pass

    @property
    @abstractmethod
    def pol_soc_dct(self):
        """Implement this in the subclasses."""
        pass

    @property
    def flag_columns_dct(self):
        """
        Dictionary containing the columns indicating suspicious rows for each sample separately for the
        trait df and the states df. Flag column names are given in the config.
        """
        flag_dct = dict()
        for esm_sample in self.samples_for_analysis:
            flag_dct[esm_sample] = dict()
            for var_type in ["traits", "states"]:
                flag_dct[esm_sample][var_type] = self.config["general"][
                    "exclude_flagged_data"
                ][esm_sample][var_type]
        return flag_dct

    @property
    def country_vars_store_path(self):
        """Relative path were the additional country variables for the analysis of coco_int are stored."""
        return self.config["trait_data"]["country_vars_other"]["store_path"]

    def apply_preprocessing_methods(self):
        """This function applies the preprocessing methods specified in the config."""
        for method in self.preprocessing_methods:
            if method not in dir(self.__class__):
                raise ValueError(f"Method '{method}' is not implemented yet.")
            getattr(self, method)()
        print(">>>>>>Preprocessing Done<<<<<<")

    def _setup_logger(self):
        """
        This function sets up the logger, configures it and sets it as a class attribute. It shows the effect of
        single preprocessing steps on sample and feature sizes for all esm_sample in self.samples_for_analysis.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        os.makedirs(self.logs_specific_path, exist_ok=True)
        fh = logging.FileHandler(self.log_filename)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Write current config to log folder
        yaml_filename = os.path.join(
            self.logs_specific_path,
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_cfg.yaml",
        )
        with open(yaml_filename, "w") as yaml_name:
            yaml.dump(self.config, yaml_name)

    def filter_and_preprocess_traits(self):
        """Wrapper that contains the filtering and preprocessing steps applied to the traits."""
        print(">>>>>>filter_and_preprocess_traits<<<<<<")
        self.logger.info(">>>>>>filter_and_preprocess_traits<<<<<<")
        for esm_sample in self.samples_for_analysis:
            print(f"---{esm_sample}---")
            self.logger.info("-----------")
            self.logger.info(f"Trait filtering for {esm_sample}")
            self.filter_traits_for_nan(esm_sample)
            if esm_sample == "coco_int":
                self.add_country_level_vars(esm_sample)
            self.inverse_code_items(esm_sample)
            self.calc_scale_means(esm_sample)
            self.one_hot_encode_categoricals(esm_sample)
            self.return_feature_counts(esm_sample)

    def filter_traits_for_nan(self, sample_name):
        """
        This function filters the traits of ESM samples specified in the configs for missings.
            1) It creates a clean column without missing values independent of the specific point of time of
                data collection (e.g., if a trait was assessed before or after the ESM-sampling phase).
            2) It removes a) unnecessary columns, and b) individual samples with missing values
                on the chosen trait columns to get a clean trait df without any missing values.
        It updates the corresponding class attributes with the filtered data.

        Args:
            sample_name: Name of the ESM-sample to filter, e.g. "coco_int"
        """
        print(">> >> >> filter_traits_for_nan << << <<")
        df_traits_tmp = getattr(self, "trait_dct")[sample_name].copy()
        self.logger.info(f"original len {sample_name} trait data: {len(df_traits_tmp)}")
        for var_type in ["soc_dem", "pers", "pol_soc"]:
            df_traits_tmp = self.create_clean_trait_columns(
                sample_name=sample_name, df_traits=df_traits_tmp, var_type=var_type
            )
        df_traits_tmp = self.remove_columns_and_nan_rows(df_traits=df_traits_tmp)
        # Log how filtering step reduced sample size and adjust trait and state dfs (class attributes) accordingly
        self.logger.info(
            f"len {sample_name} after filtering rows with NaN values: {len(df_traits_tmp)}"
        )
        getattr(self, "trait_dct")[sample_name] = df_traits_tmp

    def create_clean_trait_columns(self, sample_name, df_traits, var_type):
        """
        This function creates clean columns of the chosen trait variables for all features.
        Specifically, it defines a consistent naming convention for all features that depend
        on the variable category (e.g., bfi2_1_clean for questionnaires, age_clean for socio-demographics)
        and searches for a valid entries of the feature in the data.

        Args:
            sample_name: Name of the ESM-sample to filter, e.g. "coco_int"
            df_traits: df containing the states of the corresponding ESM-sample
            var_type: str, should be in ['soc_dem', 'pers', 'pol_soc'], Error otherwise

        Returns:
            df_traits: df that contains the added "clean" columns of the trait variables
        """
        current_ques_lst = getattr(self, f"{var_type}_dct")[sample_name]
        if var_type == "soc_dem":
            col_name_mappings = {
                demo["name"]
                + "_"
                + self.trait_cfg["new_suffix"]: [
                    demo["name"] + "_" + tp
                    for tp in demo["time_of_assessment"][sample_name]
                ]
                for demo in current_ques_lst
            }
        elif var_type in ["pers", "pol_soc"]:
            col_name_mappings = {
                ques["name"]
                + "_"
                + str(item_nr)
                + "_"
                + self.trait_cfg["new_suffix"]: [
                    ques["name"] + "_" + str(item_nr) + "_" + tp
                    for tp in ques["time_of_assessment"][sample_name]
                ]
                for ques in current_ques_lst
                for item_nr in range(1, ques["number_of_items"][sample_name] + 1)
            }
        else:
            raise ValueError("Choose 'soc_dem', 'pers' or 'pol_soc' as var_type")

        for new_col_name, col_names in col_name_mappings.items():
            df_traits[new_col_name] = df_traits.apply(
                lambda row: self.first_valid_element(
                    *[row[col_name] for col_name in col_names]
                ),
                axis=1,
            )
        return df_traits

    @staticmethod
    def first_valid_element(*args):
        """
        This function takes all trait values for the same trait of one individual as input (e.g., because traits
        were collected on multiple timepoints during the ESM data collection period) and returns the first value
        that is considered valid as an output (or NaN, if all values are NaN).

        Args:
            *args: Any cells of df, in this use case multiple columns of one specific row (individual sample)

        Returns:
            item_val / np.nan: The corresponding entry of the first non-empty cell, NaN otherwise

        Examples:
        first_non_nan([5, 4, 3, 2])
        5
        first_non_nan(["", "a lot", np.nan])
        "a lot"
        first_non_nan([np.nan, "", []])
        np.nan
        """
        for item_val in args:
            try:
                if not math.isnan(item_val):
                    return item_val
            except (
                TypeError
            ):  # e.g. string values would throw a TypeError in math.isnan(), considered as valid
                if item_val:
                    return item_val
        return np.nan

    def remove_columns_and_nan_rows(self, df_traits):
        """
        This function removes unused columns and then removes rows with nan values in the given trait_df.
        Note: Flagged Cols indicating suspicion are already removed in the "exclude_flagged_samples" method.

        Args:
            df_traits: df containing the traits for a specific ESM samples

        Returns:
            df_traits[cols_to_keep].dropna(axis=0): df_traits with only the necessary columns and without
                rows with NaN values on that columns
        """
        cols_to_keep = self.config["general"]["trait_columns_to_keep"].copy()
        cols_to_keep.extend(
            [col for col in df_traits.columns if self.trait_cfg["new_suffix"] in col]
        )
        return df_traits[cols_to_keep].dropna(axis=0)

    def adjust_trait_state_attributes(
        self, sample_name, df_states, new_sample_name=None
    ):
        """
        This function adjusts the trait or state dfs for the specified samples based on the filtering applied
        in the current preprocessing method. Thus, if the method has e.g. reduced the sample size
        in the trait dataframe, this method updates the state dataframe accordingly, so that people that
        were removed from the trait df are also remove from the state df.
        Based on when this function is applied, the way in which the dataframes are stored as class
        attribute can differ.
        If "new_sample_name" is defined, the function takes into account that after certain functions,
        we potentially have multiple dfs for one esm_sample (e.g., because of different data used for different
        social situation variables, such as "coco_int_social_interaction" or "coco_int_interaction_quantity").

        Args:
            sample_name: str, ESM-sample name given
            df_states: df containing the states of a given ESM-sample
            new_sample_name: combo of esm_sample-ssc (SSC)

        """
        df_traits_scale_means = getattr(self, "trait_dct")["scale_means"][
            f"{sample_name}_one_hot_encoded"
        ]
        df_traits_single_items = getattr(self, "trait_dct")["single_items"][
            f"{sample_name}_one_hot_encoded"
        ]
        common_ids = set(df_traits_scale_means[self.person_id_col]).intersection(
            set(df_states[self.person_id_col])
        )

        if new_sample_name:
            dyn_sample_name = new_sample_name
        else:
            dyn_sample_name = sample_name

        getattr(self, "state_dct")[dyn_sample_name] = df_states[
            df_states[self.person_id_col].isin(common_ids)
        ]
        getattr(self, "trait_dct")["scale_means"][
            f"{dyn_sample_name}_one_hot_encoded"
        ] = df_traits_scale_means[
            df_traits_scale_means[self.person_id_col].isin(common_ids)
        ]
        getattr(self, "trait_dct")["single_items"][
            f"{dyn_sample_name}_one_hot_encoded"
        ] = df_traits_single_items[
            df_traits_single_items[self.person_id_col].isin(common_ids)
        ]

    def exclude_flagged_samples(self):
        """
        This function excludes suspicious samples based on the flag variables from the config.
        This is done for trait data and state data separately. It also drops the cols that indicate
        the flagging after the corresponding rows were removed. Currently, this is only used for coco_int.
        """
        print(">>>>>>exclude_flagged_samples<<<<<<")
        self.logger.info(">>>>>>exclude_flagged_samples<<<<<<")
        self.logger.info(
            "Log how excluding flagged samples reduces the sample size in the trait and state datasets"
        )
        for esm_sample in self.samples_for_analysis:
            self.logger.info(
                f"len {esm_sample} trait data before excluding flagged rows:"
                f"{len(getattr(self, 'trait_dct')[f'{esm_sample}'])}"
            )
            self.logger.info(
                f"len {esm_sample} state data before excluding flagged rows:"
                f"{len(getattr(self, 'state_dct')[f'{esm_sample}'])}"
            )
            flag_trait_cols = self.config["general"]["exclude_flagged_data"][
                esm_sample
            ]["traits"]
            flag_state_cols = self.config["general"]["exclude_flagged_data"][
                esm_sample
            ]["states"]
            if flag_trait_cols:
                trait_df = self.trait_dct[esm_sample].copy()
                flag_filtered_trait_df = trait_df[
                    (trait_df[flag_trait_cols] == 0).all(axis=1)
                ]
                flag_filtered_trait_df = flag_filtered_trait_df.drop(
                    flag_trait_cols, axis=1
                )
                # if a person is flagged, it is also removed from the state_df
                getattr(self, "trait_dct")[esm_sample] = flag_filtered_trait_df
            if flag_state_cols:
                state_df = self.state_dct[esm_sample].copy()
                flag_filtered_state_df = state_df[
                    (state_df[flag_state_cols] == 0).all(axis=1)
                ]
                flag_filtered_state_df = flag_filtered_state_df.drop(
                    flag_state_cols, axis=1
                )
                getattr(self, "state_dct")[esm_sample] = flag_filtered_state_df
            self.logger.info(
                f"len {esm_sample} trait data after excluding flagged rows:"
                f"{len(getattr(self, 'trait_dct')[f'{esm_sample}'])}"
            )
            self.logger.info(
                f"len {esm_sample} state data after excluding flagged rows:"
                f"{len(getattr(self, 'state_dct')[f'{esm_sample}'])}"
            )
            self.logger.info("\n")

    def add_country_level_vars(self, esm_sample):
        """
        This function adds the country-level variables to the given ESM-sample (i.e., coco_int).
        Specifically, it gets the lst of country-level variables from the config, it loads the
        pkl file that stores the country-level variables and merge the two dfs. Then it drops
        rows with missings on the country level variables and updates the corresponding class atrributes.

        Args:
            esm_sample: str, name of the ESM-sample (in this case, only "coco_int" is valid)
        """
        print(">>>>>>add_country_level_vars<<<<<<")
        self.logger.info(">>>>>>add_country_level_vars<<<<<<")
        df_traits_tmp = self.trait_dct[esm_sample].copy()
        country_data_lst = [
            country_data["name"] for country_data in self.trait_cfg["country_vars"]
        ]
        country_col = f"country_{self.config['trait_data']['new_suffix']}"

        for data_source in country_data_lst:
            with open(
                os.path.join(
                    self.country_vars_store_path,
                    f"{data_source}_preprocessed.pkl",
                ),
                "rb",
            ) as f:
                df_country_tmp = pickle.load(f)
            df_traits_tmp = pd.merge(
                left=df_traits_tmp,
                right=df_country_tmp,
                left_on=country_col,
                right_index=True,
                how="left",
            )
        df_traits_tmp = df_traits_tmp.dropna(axis=0)
        # Drop raw categorical country vars from features df
        df_traits_tmp = df_traits_tmp.drop(columns=country_col)
        self.logger.info(
            f"len {esm_sample} trait data after adding country variables and removing missing: "
            f"{len(df_traits_tmp)}"
        )
        getattr(self, "trait_dct")[esm_sample] = df_traits_tmp
        self.logger.info("\n")

    def filter_and_preprocess_states(self):
        """
        Wrapper that contains the filtering and preprocessing steps applied to the traits.
        First operations are applied using the raw sample names (esm_sample).
        Other operations are applied using the processed sample names (proc_sample_name)
        Note: All operations that include operations across multiple rows (e.g., person mean centering) have
        to use the filtered data.
        """
        print(">>>>>>filter_and_preprocess_states<<<<<<")
        self.logger.info(">>>>>>filter_and_preprocess_states<<<<<<")

        for esm_sample in self.samples_for_analysis:
            print(f"---{esm_sample}---")
            self.filter_states_wb_items(esm_sample)
            self.aggregate_well_being_for_measures(esm_sample)
            self.filter_states_num_iv(esm_sample)

        for proc_sample_name in self.proc_esm_sample_lst:
            print(f"---{proc_sample_name}---")

            self.person_mean_center_wb_score(proc_sample_name)
            self.person_mean_center_iv(proc_sample_name)

    def filter_states_wb_items(self, esm_sample):
        """
        This function checks whether at least one item assessing positive affect or one item
        assessing negative affect is present in a certain measurement in the state df.
        If not, rows will be excluded. The class attributes is updated with the filtered df.

        Args:
            esm_sample: str, raw name of a given esm_sample, e.g. coco_int
        """
        print(">> >> >> filter_states_wb_items << << <<")
        df_states_tmp = getattr(self, "state_dct")[esm_sample]
        pa_notna = df_states_tmp[self.pa_items[esm_sample]].notna().any(axis=1)
        na_notna = df_states_tmp[self.na_items[esm_sample]].notna().any(axis=1)
        df_states_tmp["wb_notna_pointer"] = np.where(pa_notna & na_notna, 1, 0)
        df_states_filtered = df_states_tmp[df_states_tmp["wb_notna_pointer"] == 1]
        self.logger.info(
            f"num persons in {esm_sample} state data after filtering out rows with no na or pa items: "
            f"{df_states_filtered[self.person_id_col].nunique()}"
        )
        getattr(self, "state_dct")[esm_sample] = df_states_filtered

    def aggregate_well_being_for_measures(self, esm_sample):
        """
        This function aggregates the well-being items. Because this includes only aggregating across rows
        and not across columns, it can take place before the filtering of the states based on the number
        of independent variables with filled out surveys. It updates the class attribute accordingly.
        if suppl_type == sep_pa_na, it includes only positive or negative affect items, depending on "suppl_var".

        Args:
            esm_sample: str, raw name of a given esm_sample, e.g. coco_int
        """
        print(">> >> >> filter_states_wb_items << << <<")
        df_states = getattr(self, "state_dct")[esm_sample].copy()
        scale_min, scale_max = self.wb_items_scale_endpoints[esm_sample]
        pa_mean_score = df_states[self.pa_items[esm_sample]].mean(axis=1)
        na_mean_score = df_states[self.na_items[esm_sample]].mean(axis=1)

        if self.suppl_type == "sep_pa_na":
            if self.suppl_var == "pa":
                wb_score_per_esm = pa_mean_score
            elif self.suppl_var == "na":
                wb_score_per_esm = na_mean_score
            else:
                raise ValueError("Specify pa or na when suppl_type == sep_pa_na")
        elif self.suppl_type == "sep_ftf_cmc" and self.suppl_var == "ftf_pa":
            wb_score_per_esm = pa_mean_score
        else:
            wb_score_per_esm = (
                pa_mean_score + (scale_min - na_mean_score + scale_max)
            ) / 2

        df_states["wb_score"] = wb_score_per_esm
        getattr(self, "state_dct")[esm_sample] = df_states

    def person_mean_center_wb_score(self, proc_sample_name):
        """
        This function person-mean centers the dependent variable for the multilevel models that we
        calculate to extract the individual reactivities, because it was shown that otherwise the
        empirical Bayes estimates correlate artificially high with the person-mean
        (see Kuper etal, 2022, Appendix C)

        Args:
            proc_sample_name: str, processed name of a given sample, e.g. coco_int_social_interaction
        """
        print(">> >> >> person_mean_center_wb_score << << <<")
        df_state_tmp = getattr(self, "state_dct")[proc_sample_name].copy()
        person_means_wb = df_state_tmp.groupby(self.person_id_col)[
            "wb_score"
        ].transform("mean")
        df_state_tmp[f"wb_pmc"] = df_state_tmp["wb_score"] - person_means_wb
        getattr(self, "state_dct")[proc_sample_name] = df_state_tmp

    @abstractmethod
    def filter_states_num_iv(self, esm_sample):
        """Implement this in the subclasses."""
        pass

    def person_mean_center_iv(self, proc_sample_name):
        """Person mean centering of the ssc, this method is only implemented in the ssc subclass."""
        pass

    def log_final_filtering_results(
        self, proc_sample_name, state_df, min_number_esm_measures, current_var
    ):
        """
        This function logs the final results of the filtering process. Thus, currently, this function
        is applied in the filtering of the state data based on the number of ESM-measurements in both studies
        and the following adjustments of the trait data.
        This is the final step that reduces the sample sizes ot the traits and states.

        Args:
            proc_sample_name: str, name of the processed samples, e.g. coco_int_social_interaction
            state_df: df containing the states corresponding to proc_sample_name
            min_number_esm_measures: Minimum number of complete ESM measurements for a given variable
            current_var: Variable for filtering (i.e., social situationm variable)
        """
        self.logger.info(
            f"len {proc_sample_name} state data after filtering out rows with less than "
            f"{min_number_esm_measures} filled out surveys of the variable {current_var}: "
            f"{len(state_df)}"
        )
        self.logger.info(
            f"number of persons in same dataset: {state_df[self.person_id_col].nunique()}"
        )
        self.logger.info(
            f"Mean EMS surveys per person: {np.round(state_df.groupby(self.person_id_col).size().mean(), 2)}"
        )
        self.logger.info(
            f"SD EMS surveys per person: {np.round(state_df.groupby(self.person_id_col).size().std(), 2)}"
        )

        self.logger.info("\n")

    def one_hot_encode_categoricals(self, esm_sample):
        """
        This function one-hot encodes all categorical person variables that are specified in config.
        It updates the corresponding trait attributes.

        Args:
            esm_sample: str, given raw esm_sample name, e.g. coco_int
        """
        print(">> >> >> one_hot_encode_categoricals << << <<")
        self.logger.info(">>>>>>one_hot_encode_categoricals<<<<<<")
        df_dct = dict()
        df_dct["scale_means"] = getattr(self, "trait_dct")["scale_means"][esm_sample]
        df_dct["single_items"] = getattr(self, "trait_dct")["single_items"][esm_sample]

        for fis, df in df_dct.items():
            var_types = ["soc_dem", "pers", "pol_soc"]
            cat_columns = {}
            for var_type in var_types:
                cat_columns[var_type] = self.get_cat_columns(
                    sample_name=esm_sample, var_type=var_type
                )
            cat_cols_total = list(itertools.chain(*cat_columns.values()))

            # Edge case, this is manual step is not elegant but ok for now and validated
            if fis == "scale_means" and any(
                "corona_aff_binaries" in cat for cat in cat_cols_total
            ):
                cat_cols_total = [
                    item for item in cat_cols_total if "corona_aff_binaries" not in item
                ]

            df_traits_tmp, cat_cols_updated = self.process_multilabel_columns(
                df_tmp=df, cat_cols=cat_cols_total
            )
            # Transform column to strings to prevent unwanted column names (e.g., x_1.0)
            one_hot_df = pd.get_dummies(
                data=df_traits_tmp, columns=cat_cols_updated, drop_first=False
            )
            # Remove column name artefacts due to processing float values and convert booleans to numeric
            one_hot_df.columns = [
                col.split(".")[0] if ".0" in col else col for col in one_hot_df.columns
            ]
            boolean_cols = one_hot_df.select_dtypes(include="bool").columns
            one_hot_df[boolean_cols] = one_hot_df[boolean_cols].astype(int)
            getattr(self, "trait_dct")[fis][
                f"{esm_sample}_one_hot_encoded"
            ] = one_hot_df

    def get_cat_columns(self, sample_name, var_type):
        """
        This function gets the categorical columns based on the config specifications for a given df and var type.
        Var_type should be in ['soc_dem', 'pers', 'pol_soc'].

        Args:
            sample_name: str, raw name of a given ESM-sample, e.g. coco_int
            var_type: Category of person-level variables, must be in ['soc_dem', 'pers', 'pol_soc']

        Returns:
            cat_cols: Columns containing categorical variables for a given ESM-sample
        """
        if var_type == "soc_dem":
            cat_cols = [
                col["name"] + "_" + self.trait_cfg["new_suffix"]
                for col in getattr(self, f"{var_type}_dct")[sample_name]
                if sample_name in col["time_of_assessment"].keys()
                and col["var_type"][sample_name] == "cat"
            ]
            if sample_name == "coco_int":
                cat_cols.remove(f"country_{self.trait_cfg['new_suffix']}")
        elif var_type in ["pers", "pol_soc"]:
            cat_cols = [
                col["name"] + "_" + str(num) + "_" + self.trait_cfg["new_suffix"]
                for col in getattr(self, f"{var_type}_dct")[sample_name]
                if sample_name in col["time_of_assessment"].keys()
                and col["var_type"][sample_name] == "cat"
                for num in range(1, col["number_of_items"][sample_name] + 1)
            ]
        else:
            raise ValueError(
                "var_type must be in ['soc_dem', 'pers', 'pol_soc'], country vars are all continuous"
            )
        return cat_cols

    @staticmethod
    def process_multilabel_columns(df_tmp, cat_cols):
        """
        This function is used to convert columns of type string that contain multiple entries to single columns
        where each column represents one of the entries (e.g. if a cell value is ["4,5,6"] -> binary columns for
        each value are created). Therefore, it uses sklearns "MultiLabelBinarizer".

        Args:
            df_tmp: df containing the person-level features for a given ESM-sample
            cat_cols: lst, All column names containing categorical variables of a certain feature category

        Returns:
            df_tmp: The updated df where the raw multilabel categoricals are removed and the one-hot
                encoded multilabel categoricals are added
            cat_cols: lst, The remaining cat_cols that were not processed in this function
        """
        mlb = MultiLabelBinarizer()
        for col_name in cat_cols:
            if is_string_dtype(df_tmp[col_name]):
                if df_tmp[col_name].str.contains(",").any():
                    transformed = df_tmp[col_name].dropna().str.split(",")
                    mlb_result = mlb.fit_transform(transformed)
                    mlb_df = pd.DataFrame(
                        mlb_result,
                        columns=[f"{col_name}_{c}" for c in mlb.classes_],
                        index=transformed.index,
                    )
                    assert (
                        df_tmp.index.is_unique and mlb_df.index.is_unique
                    ), "Indices must be unique for a proper join."
                    assert (
                        df_tmp.index == mlb_df.index
                    ).all(), "Indices of df_tmp and mlb_df do not match."
                    df_tmp = df_tmp.drop(col_name, axis=1).join(mlb_df)
                    cat_cols.remove(col_name)
            else:
                pass
        return df_tmp, cat_cols

    def inverse_code_items(self, esm_sample):
        """
        This function inverse codes the items specified in the config.
        Note: In some cases, the number of items per questionnaire differ between esm-samples and therefore
        the inverse coding mappings differ. This is handled through more specific defenitions in the config
        and is reflected in this function in the differences in "recoding_lst".
        Columns have to be numeric to inverse code the items.
        Note: For coco_int, most columns are already inverse coded, therefore only a subset of the items +
        is included in its "ques_lst".

        Args:
            esm_sample: str, raw ESM-sample name (e.g., coco_int)
        """
        print(">> >> >> inverse_code_items << << <<")
        self.logger.info(">>>>>>inverse_code_items<<<<<<")
        df_traits_tmp = getattr(self, "trait_dct")[esm_sample]
        ques_lst = [
            ques for ques in self.pers_dct[esm_sample] + self.pol_soc_dct[esm_sample]
        ]
        if esm_sample == "coco_int":  # other variables are already inverse coded
            ques_lst = [
                ques
                for ques in ques_lst
                if ques["name"] in ["stab", "political_efficacy"]
            ]
        df_traits_tmp = df_traits_tmp.apply(
            lambda series: pd.to_numeric(series, errors="ignore")
        )

        for ques in ques_lst:
            if isinstance(ques.get("to_recode"), list):
                recoding_lst = ques["to_recode"]
            elif isinstance(ques.get("to_recode"), dict):
                recoding_lst = ques["to_recode"][esm_sample]
            else:  # if there is nothing to inverse code in a questionnaire -> no mapping, continue the loop
                continue
            df_traits_tmp = self.inverse_coding(
                df_traits_tmp=df_traits_tmp.copy(),
                ques=ques,
                recoding_lst=recoding_lst,
                scale_min=ques["scale_endpoints"][0],
                scale_max=ques["scale_endpoints"][1],
            )

        getattr(self, "trait_dct")[esm_sample] = df_traits_tmp

    def inverse_coding(self, df_traits_tmp, ques, recoding_lst, scale_min, scale_max):
        """
        Function that inverse codes the subset of items from a given questionnaire that need to be recoded

        Args:
            df_traits_tmp: df containing the person-level variables of a given ESM-sample
            ques: str, Questionnaire name, taken from the config (e.g., bfi2)
            recoding_lst: lst, contains items that need to be recoded, as specified in the config
                (taken from the questionnaires manuals)
            scale_min: int, original min of the scale
            scale_max: int, original max of the scale

        Returns:
            df_traits_tmp: df containing the df with the inverse coded items that were specified in the "recoding_lst"
                for the questionnaire "ques"

                Example 1: Basic Recoding of Two Columns
        ----------------------------------------
        data = {'A': [0, 1, 2, 3, 4, 5, 6, 7],
                'B': [7, 6, 5, 4, 3, 2, 1, 0]}
        df_traits = pd.DataFrame(data)
        ques = {"name": "trait1"}
        recoding_lst = [1, 2]
        scale_min = 0
        scale_max = 7
        inverse_coding(df_traits, ques, recoding_lst, scale_min, scale_max)
                         A                B
        0                7                0
        1                6                1
        2                5                2
        3                4                3
        4                3                4
        5                2                5
        6                1                6
        7                0                7
        """
        for item_nr in recoding_lst:
            col_name = f'{ques["name"]}_{item_nr}_{self.trait_cfg["new_suffix"]}'
            df_traits_tmp[col_name] = scale_max + scale_min - df_traits_tmp[col_name]
        return df_traits_tmp

    def calc_scale_means(self, esm_sample):
        """
        This function calculates scale means for the questionnaires based on the dimension mapping in the config.
        It creates two separate dataframes, one only containing the single items and one containing only the scale
        means for the features of the categories "personality variables" and "political and societal attitudes".
        This separation makes the subsequent analyses easier to handle. Therefore, it also deletes the raw sample.
        In the class attribute trait_dct, it creates the new dct keys "scale_means" and "single_items" and inserts
        the corresponding dfs created in this function.
        Note: For a given questionnaire, the mapping to its scale means can differ (e.g., because only a subset of
            the items is used in a specific ESM-sample). Therefore, the value of the dict level "dimension_mapping"
            can either be a nested dict with custom mappings per esm_sample or a simple dict with the dimensions as keys
            and the corresponding items as values.
        Note: This function assumes that items are already inverted (inverse_code_items)

        Args:
            esm_sample: str, given raw ESM-sample name, e.g. coco_int
        """
        print(">> >> >> calc_scale_means << << <<")
        self.logger.info(">>>>>>calc_scale_means<<<<<<")
        df_traits_tmp = getattr(self, "trait_dct")[esm_sample].copy()
        df_single_items = copy.deepcopy(df_traits_tmp)
        if "single_items" not in getattr(self, "trait_dct"):
            getattr(self, "trait_dct")["single_items"] = dict()
        getattr(self, "trait_dct")["single_items"][esm_sample] = df_single_items

        ques_lst_cont = [
            ques
            for ques in self.pers_dct[esm_sample] + self.pol_soc_dct[esm_sample]
            if esm_sample in ques["time_of_assessment"].keys()
            and ques["var_type"][esm_sample] == "cont"
        ]
        for ques in ques_lst_cont:
            dimension_mapping = ques.get("dimension_mapping")

            if isinstance(dimension_mapping, dict) and all(
                isinstance(v, dict) for v in dimension_mapping.values()
            ):
                for dimension, item_nrs in ques["dimension_mapping"][
                    esm_sample
                ].items():
                    df_traits_tmp = self.replace_items_with_scale_means(
                        df_traits_tmp=df_traits_tmp,
                        ques=ques,
                        dimension=dimension,
                        item_nrs=item_nrs,
                        dim_mapping_exists=True,
                    )
            elif isinstance(dimension_mapping, dict):
                for dimension, item_nrs in ques["dimension_mapping"].items():
                    df_traits_tmp = self.replace_items_with_scale_means(
                        df_traits_tmp=df_traits_tmp,
                        ques=ques,
                        dimension=dimension,
                        item_nrs=item_nrs,
                        dim_mapping_exists=True,
                    )
            else:  # no dimension_mapping or a single number
                df_traits_tmp = self.replace_items_with_scale_means(
                    df_traits_tmp=df_traits_tmp,
                    ques=ques,
                    dimension=None,
                    item_nrs=None,
                    dim_mapping_exists=False,
                )
        # Edge case if questionnaire items are binaries but building a scale mean makes sense (corona_aff_binaries)
        ques_lst_cat = [
            ques
            for ques in self.pers_dct[esm_sample] + self.pol_soc_dct[esm_sample]
            if esm_sample in ques["time_of_assessment"].keys()
            and ques["var_type"][esm_sample] == "cat"
        ]

        for ques in ques_lst_cat:
            if ques["number_of_items"][esm_sample] > 1:
                sum_cat_cols = [
                    col for col in df_traits_tmp.columns if ques["name"] in col
                ]
                # Variables are not ohe yet, we therefore adjust the binary values for clear interpretation
                df_traits_tmp[sum_cat_cols] = df_traits_tmp[sum_cat_cols].apply(
                    lambda x: x.replace(2, 0)
                )
                df_traits_tmp[
                    f'{ques["name"]}_{self.trait_cfg["new_suffix"]}'
                ] = df_traits_tmp[sum_cat_cols].mean(axis=1)
                df_traits_tmp = df_traits_tmp.drop(sum_cat_cols, axis=1)
        if "scale_means" not in getattr(self, "trait_dct"):
            getattr(self, "trait_dct")["scale_means"] = dict()
        getattr(self, "trait_dct")["scale_means"][esm_sample] = df_traits_tmp
        # Remove base sample that are neither "single_items" nor "scale_means"
        del self.trait_dct[esm_sample]

    def replace_items_with_scale_means(
        self, df_traits_tmp, ques, dimension, item_nrs, dim_mapping_exists
    ):
        """
        This function is used in "calc_scale_means". For a given questionnaire and a dimension,
        it builds the scale mean by averaging across the items associated with the dimension (specified by item_nrs)
        and deletes the corresponding single items. It returns the updated df.
        Args:
            df_traits_tmp: df containing the traits for a given ESM-sample
            ques: str, questionnaire name as specified in the config, e.g., bfi2
            dimension: str, dimension name as specified in the config, e.g., extraversion
            item_nrs: lst, numbers mapping items to dimension, e.g., [2, 3]
            dim_mapping_exists: bool, indicates whether there is a dimension mapping in the config or not, if not,
                then the questionnaire has only one dimension (e.g., cmq)

        Returns:
            trait_df_proc: df where the scale mean for a certain dimension was formed and the associated
                single items were removed
        """
        if dim_mapping_exists:
            cols = [
                f"{ques['name']}_{nr}_{self.trait_cfg['new_suffix']}" for nr in item_nrs
            ]
            df_traits_tmp[
                f'{ques["name"]}_{dimension}_{self.trait_cfg["new_suffix"]}'
            ] = df_traits_tmp[cols].mean(axis=1)
        else:
            cols = [
                col for col in df_traits_tmp.columns if col.startswith(ques["name"])
            ]
            df_traits_tmp[
                f'{ques["name"]}_{self.trait_cfg["new_suffix"]}'
            ] = df_traits_tmp[cols].mean(axis=1)
        trait_df_proc = df_traits_tmp.drop(cols, axis=1)
        return trait_df_proc

    def return_feature_counts(self, esm_sample):
        """
        This function returns the feature count for the two feature inclusion strategies (single_items, scale_means)
        for a given ESM-sample. Therefore, it subtracts the number of columns that are in the dataframe
        but not used as features in the machine learning analysis (e.g., id) from the number of total columns
        in the dataframe and returns the corresponding counts.
        it further deletes the non OHE trait dataframe, because in the following, we will only operate on the
        OHE trait df.

        Args:
            esm_sample: str, raw name of a given esm_sample, e.g., coco_int
        """
        print(">> >> >> return_feature_counts << << <<")
        self.logger.info(">>>>>>return_feature_counts<<<<<<")
        no_feature_cols = self.config["general"]["trait_columns_to_keep"]
        for fis in ["single_items", "scale_means"]:
            # Assessing the df without one-hot encoding
            df = getattr(self, "trait_dct")[fis][esm_sample]
            num_features = len(df.columns) - len(no_feature_cols)
            self.logger.info(f"num features {esm_sample} {fis}: {num_features}")
            # Assessing the one-hot encoded df
            df_ohe = getattr(self, "trait_dct")[fis][f"{esm_sample}_one_hot_encoded"]
            num_features_ohe = len(df_ohe.columns) - len(no_feature_cols)
            self.logger.info(f"num features {esm_sample} {fis} ohe: {num_features_ohe}")
            self.logger.info(".")
        # Remove the non-one-hot encoded dfs
        self.trait_dct["single_items"].pop(esm_sample, None)
        self.trait_dct["scale_means"].pop(esm_sample, None)

    def set_id_as_index(self):
        """This function sets the id of the trait dfs as the index, which helps in further processing."""
        for sample, fis_dct in self.trait_dct.items():
            for fis, trait_df in fis_dct.items():
                trait_df = trait_df.set_index("id", drop=True)
                self.trait_dct[sample][fis] = trait_df

    def sanity_checks(self):
        """
        This function is used to validate the preprocessing process.
        Specifically it checks the correspondence of the trait and state dfs for a given sample, it checks the
        scales reliability and intercorrelations, and checks if expected patterns between traits and states
        show up. Suspicious results are logged (e.g., logging cronbachs alphas below .50) what might give
        a hint for wrong preprocessing.
        """
        self.logger.info(">>>>>>sanity_checks<<<<<<")
        print(">>>>>>sanity_checks<<<<<<")
        for fis, fis_dct in self.trait_dct.items():
            for proc_sample_name, state_df in self.state_dct.items():
                self.logger.info(".")
                self.logger.info(f"{fis}")
                self.logger.info(f"{proc_sample_name}")
                raw_sample_name = [
                    sample
                    for sample in self.samples_for_analysis
                    if sample in proc_sample_name
                ][0]
                trait_df = fis_dct[f"{proc_sample_name}_one_hot_encoded"]
                assert len(trait_df.index) == len(
                    set(state_df[self.person_id_col])
                ), "Different number of persons in trait and state df"
                assert set(trait_df.index) == set(
                    state_df[self.person_id_col]
                ), f"IDs differ between trait and state df for {proc_sample_name}"
                assert (
                    not trait_df.isnull().any().any()
                ), f"NAN values in trait df for {proc_sample_name}"

                # check Cronbachs alpha for all scales
                if fis == "single_items":
                    pers_lst = self.pers_dct[raw_sample_name]
                    alpha_dct = self.check_scale_reliability(
                        pers_lst=pers_lst,
                        trait_df=trait_df,
                        new_suffix=self.config["trait_data"]["new_suffix"],
                        raw_sample_name=raw_sample_name,
                    )
                    susp_alpha_dct = {
                        key: round(value, 2)
                        for key, value in alpha_dct.items()
                        if value < 0.50
                    }
                    json_dct = json.dumps(susp_alpha_dct, indent=4)
                    self.logger.info("Scales with cronbachs alpha below .50:")
                    self.logger.info(json_dct)
                # check of theory based correlations between states and traits can be replicated
                if fis == "scale_means":
                    corr_wb_extra, corr_wb_neuro = self.check_trait_state_convergence(
                        trait_df=trait_df,
                        state_df=state_df,
                        sample_raw_name=raw_sample_name,
                    )
                    self.logger.info(
                        f"Correlations between state-wb and personality traits for {proc_sample_name}"
                    )
                    self.logger.info(f"corr_wb_extra: {np.round(corr_wb_extra, 2)}")
                    self.logger.info(f"corr_wb_neuro: {np.round(corr_wb_neuro, 2)}")

                    # check cronbachs alpha of the well-being scores (does not depend on fis)
                    (
                        alpha_wb,
                        alpha_pa,
                        alpha_na,
                    ) = self.check_state_wb_score_reliability(
                        state_df=state_df, raw_sample_name=raw_sample_name
                    )
                    if isinstance(alpha_wb, (int, float)):
                        self.logger.info(
                            f"cronbachs alpha well-being: {np.round(alpha_wb, 2)}"
                        )
                    if isinstance(alpha_pa, (int, float)):
                        self.logger.info(
                            f"cronbachs alpha positive affect: {np.round(alpha_pa, 2)}"
                        )
                    if isinstance(alpha_na, (int, float)):
                        self.logger.info(
                            f"cronbachs alpha negative affect: {np.round(alpha_na, 2)}"
                        )

    @staticmethod
    def check_scale_reliability(pers_lst, trait_df, new_suffix, raw_sample_name):
        """
        Check the scale reliability for each questionnaire and dimension.

        Args:
            pers_lst: List of person-level variables that are part of questionnaires
            trait_df: df containing the traits for a given ESM-sample
            new_suffix: Suffix defined in the config given to cleaned person-level variables
            raw_sample_name: raw ESM-sample name, e.g., coco_int

        Returns:
            alpha_results: Dict containing dimensions as keys and alphas as values
        """
        alpha_results = dict()
        for questionnaire in pers_lst:
            name = questionnaire["name"]
            num_items = questionnaire["number_of_items"][raw_sample_name]
            dimensions = questionnaire.get("dimension_mapping", None)
            if dimensions:
                for dimension, items in dimensions.items():
                    column_names = [f"{name}_{i}_{new_suffix}" for i in items]
                    df_dimension = trait_df[column_names]
                    if len(items) > 1:
                        alpha = pg.cronbach_alpha(data=df_dimension)[0]
                        alpha_results[f"{name}_{dimension}"] = alpha
            else:
                column_names = [
                    f"{name}_{i}_{new_suffix}" for i in range(1, num_items + 1)
                ]
                df_dimension = trait_df[column_names]
                if len(column_names) > 1:
                    alpha = pg.cronbach_alpha(data=df_dimension)[0]
                    alpha_results[f"{name}"] = alpha
        return alpha_results

    def check_trait_state_convergence(self, trait_df, state_df, sample_raw_name):
        """
        Try if expected patterns between traits and states can be found.
        As an example, we use the person-mean of well-being from the state df and
        the traits extraversion and neuroticism from the state df

        Args:
            trait_df: df containing the traits for a given ESM-sample
            state_df: df containing the states for a given ESM-sample
            sample_raw_name: raw ESM-sample name, e.g., coco_int

        Returns:
            corr_extra: corr between state level wb (person-mean) and trait extraversion
            corr_extra: corr between state level wb (person-mean) and trait neuroticism
        """
        if sample_raw_name == "emotions":
            bfi_name = "bfi2s"
        else:
            bfi_name = "bfi2"
        person_means = state_df.groupby(self.person_id_col)["wb_score"].mean()
        corr_extra = person_means.corr(
            trait_df[f'{bfi_name}_extra_{self.trait_cfg["new_suffix"]}']
        )
        corr_neuro = person_means.corr(
            trait_df[f'{bfi_name}_neuro_{self.trait_cfg["new_suffix"]}']
        )
        return corr_extra, corr_neuro

    def check_state_wb_score_reliability(self, state_df, raw_sample_name):
        """
        This function evaluates the reliability of the used well-being measure.

        Args:
            state_df: df containing the states for a given ESM-sample
            raw_sample_name: raw ESM-sample name, e.g., coco_int

        Returns:
            alpha_wb_score: Reliability of the well-being score including positive and negative affect items
            alpha_pa_score: Reliability of the well-being score including only positive affect
            alpha_na_score: Reliability of the well-being score including only negative affect
        """
        # Retrieve the scale endpoints for the current sample
        scale_min, scale_max = self.wb_items_scale_endpoints[raw_sample_name]
        state_df_wb_items = state_df[self.wb_items[raw_sample_name]]
        pa_wb_items = state_df[self.pa_items[raw_sample_name]]
        if len(pa_wb_items.columns) > 1:
            alpha_pa_score = pg.cronbach_alpha(data=pa_wb_items)[0]
        else:
            alpha_pa_score = ""
        na_wb_items = state_df[self.na_items[raw_sample_name]]
        if len(na_wb_items.columns) > 1:
            alpha_na_score = pg.cronbach_alpha(data=na_wb_items)[0]
        else:
            alpha_na_score = ""
        # Recode the na_items in state_df according to the scale endpoints
        state_wb_items_recoded = state_df[self.na_items[raw_sample_name]].apply(
            lambda x: (scale_min - x - scale_max)
        )
        # Drop the original na_items from state_df_wb_items
        state_df_wb_items = state_df_wb_items.drop(
            columns=self.na_items[raw_sample_name]
        )
        # Concatenate the recoded na_items back with the state_df_wb_items
        state_df_wb_combined = pd.concat(
            [state_df_wb_items, state_wb_items_recoded], axis=1
        )
        alpha_wb_score = pg.cronbach_alpha(data=state_df_wb_combined)[0]
        return alpha_wb_score, alpha_pa_score, alpha_na_score

    def store_processed_dfs(self):
        """
        This function stores the processed dfs as pickle files (.pkl) in a given folder.
            self.traits is a nested dict with the following hierarchy
                esm_sample: fis_dct
                    fis: trait_dfs
            self.states is dict with the following hierarchy
                esm_sample: state_dfs
        This structure will be partially mirrored in the storing process. Separate folders will be created for
            single_items
            scale means
        but the dfs for the different esm_samples will be stored in the same folder.

        Examples for the stored files are:
        "root_dir/main/ssc/traits/scale_means/coco_int_interaction_closeness_one_hot_encoded_preprocessed.pkl"
        "root_dir/main/ssc/states/coco_int_interaction_closeness_preprocessed.pkl"
        """
        print(">>>>>>store_processed_dfs<<<<<<")
        self.logger.info(">>>>>>store_processed_dfs<<<<<<")
        # store traits
        for fis, fis_dct in self.trait_dct.items():
            # create separate folders for single item and scale mean dfs
            for proc_sample_name, df_trait in fis_dct.items():
                file_path_traits = os.path.join(self.store_path, "traits", fis)
                file_name_traits = os.path.join(
                    file_path_traits, f"{proc_sample_name}_preprocessed.pkl"
                )
                self.store_file(
                    file_path=file_path_traits, file_name=file_name_traits, df=df_trait
                )
                print(f"Stored traits {proc_sample_name} in {file_path_traits}")
                self.logger.info(
                    f"Stored traits {proc_sample_name} in {file_path_traits}"
                )
        # store states
        for proc_sample_name, df_state in self.state_dct.items():
            file_path_states = os.path.join(self.store_path, "states")
            file_name_states = os.path.join(
                file_path_states, f"{proc_sample_name}_preprocessed.pkl"
            )
            self.store_file(
                file_path=file_path_states, file_name=file_name_states, df=df_state
            )
            print(f"Stored states {proc_sample_name} in {file_path_states}")
            self.logger.info(f"Stored traits {proc_sample_name}")

    @staticmethod
    def store_file(file_path, file_name, df):
        """
        This function is used to store a pd.DataFrame as a pickle file (.pkl)

        Args:
            file_path: Relative path were the preprocessed data is stored
            file_name: Name of the pkl file for the data stored
            df: state or trait df that is to be stored
        """
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if os.path.exists(file_name):
            warnings.warn("File already exists and will be overwritten")
        with open(file_name, "wb") as f:
            pickle.dump(df, f)
