from src.preprocessing.BasePreprocessor import BasePreprocessor


class PreprocessorSSC(BasePreprocessor):
    """
    Preprocessor class for the analysis of social situation characteristics.
    Inherits from BasePreprocessor. Methods unique to SSC are defined in this subclass.
    For attributes, see BasePreprocessor.
    """

    def __init__(self, config_path, sample_dct):
        """
        Constructor method of the PreprocessorSSC Class.

        Args:
            config_path: str, relative path to the config file
            sample_dct: Dictionary that is passed to the preprocessor containing trait and state df of the ESM samples
        """
        super().__init__(config_path, sample_dct, "ssc")

    @property
    def preprocessing_methods(self):
        """Preprocessing methods that are applied to the SSC analysis, specified in config."""
        return self.config["general"]["preprocessing_methods_ssc"]

    @property
    def soc_dem_dct(self):
        """Dict containing the socio-demographics in each ESM-sample."""
        soc_dem_dct = dict()
        for esm_sample in self.samples_for_analysis:
            # Apply additional filtering based on self.trait_cfg['mse_only']
            soc_dem_dct[esm_sample] = [
                dem
                for dem in self.trait_cfg["socio_demographics"]
                if esm_sample in dem["time_of_assessment"].keys()
                and not any(
                    dem["name"].startswith(prefix)
                    for prefix in self.trait_cfg["mse_only"][esm_sample]
                )
            ]
        return soc_dem_dct

    @property
    def pers_dct(self):
        """Dict containing the personality variables in each ESM-sample."""
        pers_dct = dict()
        for esm_sample in self.samples_for_analysis:
            pers_dct[esm_sample] = [
                pers
                for pers in self.trait_cfg["personality"]
                if esm_sample in pers["time_of_assessment"].keys()
                and not any(
                    pers["name"].startswith(prefix)
                    for prefix in self.trait_cfg["mse_only"][esm_sample]
                )
            ]
        return pers_dct

    @property
    def pol_soc_dct(self):
        """Dict containing the political and societal attitudes in each ESM-sample."""
        pol_soc_dct = dict()
        for esm_sample in self.samples_for_analysis:
            pol_soc_dct[esm_sample] = [
                pol_soc
                for pol_soc in self.trait_cfg["polit_soc_attitudes"]
                if esm_sample in pol_soc["time_of_assessment"].keys()
                and not any(
                    pol_soc["name"].startswith(prefix)
                    for prefix in self.trait_cfg["mse_only"][esm_sample]
                )
            ]
        return pol_soc_dct

    @property
    def soc_int_vars(self):
        """Set a Dict with esm_samples and corresponding soc_int_vars as attribute."""
        soc_int_var_dct = dict()
        for esm_sample in self.samples_for_analysis:
            soc_int_var_dct[esm_sample] = [
                soc_int_var
                for soc_int_var, vals in self.state_cfg["ssc"][
                    "social_interaction_vars"
                ].items()
                if esm_sample in vals["samples"]
            ]
            # add interaction_quantity in coco_ut / ftf
            if esm_sample == "coco_ut" and self.suppl_var in ["ftf", "ftf_pa"]:
                soc_int_var_dct[esm_sample].append("interaction_quantity")
        return soc_int_var_dct

    def remove_mse_specific_items(self):
        """
        This function removes the items that are specific for the analysis of major societal events
        from the trait dfs and from the dictionaries defining the traits for preprocessing in the config
        (i.e., election specific items in coco_ut and COVID-specific items in emotions).
        """
        self.logger.info(">>>>>>remove_mse_specific_items<<<<<<")
        for sample_name in self.samples_for_analysis:
            df_traits_tmp = getattr(self, "trait_dct")[sample_name]
            remove_lst = list()
            for ques in self.trait_cfg["mse_only"][sample_name]:
                remove_lst.extend(
                    [col for col in df_traits_tmp if col.startswith(ques)]
                )
            df_traits_tmp = df_traits_tmp.drop(remove_lst, axis=1)
            getattr(self, "trait_dct")[sample_name] = df_traits_tmp

    def filter_states_num_iv(self, esm_sample):
        """
        This function filters the state dfs based on a certain number of complete measurements that are
        presupposed per social_interaction_variable. The logic differs for binary (e.g., social interaction) and
        continuous (e.g., interaction quantity) measures.
        The specific numbers are defined in the config. For the paper, we have used 10 complete measurements
        for continuous social situation variables and 5 complete measurements per class for binary social
        situation variables.
        Because the data for one esm-sample can differ for different social situation variables, we create
        new dfs and corresponding names that are a esm-sample_social-situation-variable combination
        (e.g., coco_int_social_interaction). The function further deletes the raw sample df (e.g., coco_int)
        from the class attributes.

        Args:
            esm_sample: str, raw name of a given ESM-sample
        """
        print(">> >> >> filter_states_num_iv << << <<")
        sample_soc_int_vars = self.soc_int_vars[esm_sample]
        # interaction quantity was only assessed for ftf interactions in coco_ut
        if self.suppl_var in ["ftf", "ftf_pa"] and esm_sample == "coco_ut":
            sample_soc_int_vars.append("interaction_quantity")

        for soc_int_var in sample_soc_int_vars:
            df_states_tmp = getattr(self, "state_dct")[esm_sample].copy()
            df_states_tmp = df_states_tmp.dropna(subset=[soc_int_var])
            min_number_esm_measures = self.state_cfg["ssc"]["social_interaction_vars"][
                soc_int_var
            ]["min_number_esm_measures"]
            excl_no_int = self.state_cfg["ssc"]["social_interaction_vars"][soc_int_var][
                "exclude_no_interaction_data"
            ]
            soc_int_yes = self.state_cfg["ssc"]["social_interaction_vars"][
                "social_interaction"
            ]["raw_values"]["yes"]
            var_type = self.state_cfg["ssc"]["social_interaction_vars"][soc_int_var][
                "var_type"
            ]

            # Specific filtering for certain supplementary analyses
            if self.suppl_type == "sep_ftf_cmc":
                df_states_tmp = self.filter_states_ftf_cmc(
                    df_states_tmp=df_states_tmp,
                    soc_int_var=soc_int_var,
                    var_type=var_type,
                )

            if (
                excl_no_int
            ):  # use all data for social_interaction, use only interaction data for other vars
                df_states_tmp = df_states_tmp[
                    df_states_tmp["social_interaction"] == soc_int_yes
                ]

            if var_type == "binary":
                # replace original yes/no mapping with 0 and 1
                value_mapping = self.state_cfg["ssc"]["social_interaction_vars"][
                    soc_int_var
                ]["value_mapping"]
                df_states_tmp[soc_int_var] = df_states_tmp[soc_int_var].replace(
                    value_mapping
                )
                df_states_filtered = self.filter_binary_iv(
                    df_states=df_states_tmp,
                    soc_int_var=soc_int_var,
                    min_number_esm_measures=min_number_esm_measures,
                    id_col=self.person_id_col,
                    binary_vals=list(value_mapping.values()),
                )
            elif var_type == "continuous":
                df_states_filtered = self.filter_continuous_iv(
                    df_states=df_states_tmp,
                    soc_int_var=soc_int_var,
                    min_number_esm_measures=min_number_esm_measures,
                    id_col=self.person_id_col,
                )
            else:
                raise ValueError("var_type should be binary or continuous")

            # For logging the results, we use the new sample name that contains the esm-sample and the soc_int_var
            sample_name_new = f"{esm_sample}_{soc_int_var}"
            # We append the new sample name to the class attribute lst
            self.proc_esm_sample_lst.append(sample_name_new)
            self.adjust_trait_state_attributes(
                sample_name=esm_sample,
                df_states=df_states_filtered,
                new_sample_name=sample_name_new,
            )
            self.log_final_filtering_results(
                proc_sample_name=sample_name_new,
                state_df=getattr(self, "state_dct")[sample_name_new],
                min_number_esm_measures=min_number_esm_measures,
                current_var=soc_int_var,
            )
        # Remove Base Samples
        self.state_dct.pop(esm_sample, None)
        self.trait_dct["single_items"].pop(f"{esm_sample}_one_hot_encoded", None)
        self.trait_dct["scale_means"].pop(f"{esm_sample}_one_hot_encoded", None)

    def filter_states_ftf_cmc(self, df_states_tmp, soc_int_var, var_type):
        """
        This function is only applied in the supplementary analysis where face-to-face social interactions
        and computer-mediated interactions are separated (suppl_type == sep_ftf_cmc).
        It filters the dataframe so that either only ftf or only cmc interactions are included.
        NOte: if soc_int_var is social_interaction, one must take into account that the
        variable indicating the interaction_medium is NaN if no interaction took place at all.

        Args:
            df_states_tmp: df containing the states of a given ESM-sample
            soc_int_var: social situation variable, e.g. "social_interaction"
            var_type: variable type of the social situation variable, "binary" or "continuous"

        Returns:
            df_states_tmp: Filtered state df containing either only ftf or cmc interactions
        """
        soc_int_no = self.state_cfg["ssc"]["social_interaction_vars"][
            "social_interaction"
        ]["raw_values"]["no"]
        if self.suppl_var in ["ftf", "ftf_pa"]:
            if var_type == "binary":
                df_states_tmp = df_states_tmp[
                    (df_states_tmp["interaction_medium_binary"] == 1)
                    | (df_states_tmp[soc_int_var] == soc_int_no)
                ]
            else:
                df_states_tmp = df_states_tmp[
                    (df_states_tmp["interaction_medium_binary"] == 1)
                ]
            self.logger.info("include only face to face interactions")
        elif self.suppl_var == "cmc":
            if var_type == "binary":
                df_states_tmp = df_states_tmp[
                    (df_states_tmp["interaction_medium_binary"] == 0)
                    | (df_states_tmp[soc_int_var] == soc_int_no)
                ]
            else:
                df_states_tmp = df_states_tmp[
                    (df_states_tmp["interaction_medium_binary"] == 0)
                ]
            self.logger.info("include only computer mediated interactions")
        else:
            raise ValueError("suppl_var for sep_ftf_cmc should be ftf or cmc")
        return df_states_tmp

    @staticmethod
    def filter_binary_iv(
        df_states, soc_int_var, min_number_esm_measures, id_col, binary_vals
    ):
        """
        This function filters the binary social situation variables (currently only social_interaction) based
        on the specifications defined in the config. We presuppose a certain number of filled out surveys for both
        categories (social interaction took place, social interaction did not take place, currently 5 per binary var).
        All persons that do not fulfill these criteria will be removed from the dataframe.

        Args:
            df_states: df containing the states of a given ESM-sample
            soc_int_var: social situation variable, e.g. "social_interaction"
            min_number_esm_measures: min number of measurements per binary category, specified in config
            id_col: column indicating the person id in the state df
            binary_vals: values of the binary variable, specified in config

        Returns:
            df_states[df_states[id_col].isin(intersected_ids)]: Filtered state_df
        """
        tmp_dct = {
            f"var_lst_{val}": df_states[df_states[soc_int_var] == val]
            .groupby(id_col)
            .size()
            for val in binary_vals
        }
        tmp_dct = {
            key: value[value >= min_number_esm_measures].index.tolist()
            for key, value in tmp_dct.items()
        }
        intersected_ids = set(tmp_dct[f"var_lst_{binary_vals[0]}"]).intersection(
            tmp_dct[f"var_lst_{binary_vals[1]}"]
        )
        return df_states[df_states[id_col].isin(intersected_ids)]

    @staticmethod
    def filter_continuous_iv(df_states, soc_int_var, min_number_esm_measures, id_col):
        """
        This function filters the continuous social situation variables based on the number of complete
        ESM measurements specified in the config. For the paper, we used 10 measurements per continuous
        social situation variable.

        Args:
            df_states: df containing the states of a given ESM-sample
            soc_int_var: social situation variable, e.g. "social_interaction"
            min_number_esm_measures: min number of measurements for the continuous variable, specified in config
            id_col: column indicating the person id in the state df

        Returns:
            filtered_df: Filtered state df
        """
        count_series = df_states[df_states[soc_int_var].notna()].groupby(id_col).size()
        filtered_df = df_states[
            df_states[id_col].isin(
                count_series[count_series >= min_number_esm_measures].index
            )
        ]
        return filtered_df

    def person_mean_center_iv(self, proc_sample_name):
        """
        This method person mean centers the soc_int_var of a given esm_sample-soc_int_var combination.
        It updates the corresponding class attribute with the processed df.

        Args:
            proc_sample_name: str, name of a processed esm-sample, e.g., coco_int_social_interaction
        """
        print(">> >> >> person_mean_center_iv << << <<")
        df_state_tmp = getattr(self, "state_dct")[proc_sample_name]
        raw_sample_name = [
            sample for sample in self.samples_for_analysis if sample in proc_sample_name
        ][0]
        social_int_var = [
            var for var in self.soc_int_vars[raw_sample_name] if var in proc_sample_name
        ][0]
        person_means = df_state_tmp.groupby(self.person_id_col)[
            social_int_var
        ].transform("mean")
        df_state_tmp[f"{social_int_var}_pmc"] = (
            df_state_tmp[social_int_var] - person_means
        )
        getattr(self, "state_dct")[proc_sample_name] = df_state_tmp
