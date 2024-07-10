from datetime import datetime, timedelta

import pandas as pd

from src.preprocessing.BasePreprocessor import BasePreprocessor


class PreprocessorMSE(BasePreprocessor):
    def __init__(self, config_path, sample_dct):
        """
        Constructor method of the PreprocessorMSE Class.

        Args:
            config_path: str, relative path to the config file
            sample_dct: Dictionary that is passed to the preprocessor containing trait and state df of the ESM samples
        """
        super().__init__(config_path, sample_dct, "mse")

    @property
    def preprocessing_methods(self):
        """Preprocessing methods that are applied to the SSC analysis, specified in config."""
        return self.config["general"]["preprocessing_methods_mse"]

    @property
    def soc_dem_dct(self):
        """Dict containing the socio-demographics in each ESM-sample."""
        soc_dem_dct = dict()
        for esm_sample in self.samples_for_analysis:
            soc_dem_dct[esm_sample] = [
                dem
                for dem in self.trait_cfg["socio_demographics"]
                if esm_sample in dem["time_of_assessment"].keys()
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
            ]
        return pol_soc_dct

    @property
    def esm_tp_col(self):
        """Returns the column specifying the datetime of the esm survey in the state df as a class attribute."""
        return self.state_cfg["mse"]["esm_timepoint"]

    def aggregate_well_being_for_days(self, proc_sample_name):
        """
        This function aggregates the well-being score on a day level at updates the class attribute.

        Args:
            proc_sample_name: str, name of processed esm_sample, e.g. coco_ut_us_election_2020
        """
        print(">> >> >> aggregate_well_being_for_days << << <<")
        df_state_tmp = getattr(self, "state_dct")[proc_sample_name].copy()
        df_state_tmp[self.esm_tp_col] = pd.to_datetime(df_state_tmp[self.esm_tp_col])
        df_state_tmp["date"] = df_state_tmp[self.esm_tp_col].dt.date
        df_state_tmp["wb_score"] = df_state_tmp.groupby(["id", "date"])[
            "wb_score"
        ].transform("mean")
        getattr(self, "state_dct")[proc_sample_name] = df_state_tmp

    def assign_well_being_to_timevar(self, proc_sample_name):
        """
        This function creates a mapping between a linear time variable and the esm timepoints. This is not used
        in the paper or the supplements but was used for exploring if omitting the day-level aggregation changes
        the results (it does not).

        Args:
            proc_sample_name: str, name of processed esm_sample, e.g. coco_ut_us_election_2020
        """
        print(">> >> >> assign_well_being_to_timevar << << <<")
        df_state_tmp = getattr(self, "state_dct")[proc_sample_name].copy()
        df_state_tmp[self.esm_tp_col] = pd.to_datetime(df_state_tmp[self.esm_tp_col])
        start_date = df_state_tmp[self.esm_tp_col].min()
        df_state_tmp["hours_since_event"] = (
            ((df_state_tmp[self.esm_tp_col] - start_date).dt.total_seconds() / 3600)
            .round()
            .astype(int)
        )
        getattr(self, "state_dct")[proc_sample_name] = df_state_tmp

    def filter_states_num_iv(self, esm_sample):
        """
        This function filters the state dfs based on (a) a certain number of measurements per person in
        the defined time interval of post-event adjustment (given in the config, 14 days used in paper)
        and (b) a certain number of days per person with at least one measurement in the defined time
        interval of post-event adjustment, because well-being is aggregated on the day-level.
        The specific numbers are defined in the config. For the paper, we have used 10 measurements in total
        and measurements on at least 5 different days as the lower boarder.
        New dfs are created with the applied filtering and new df names in the class attributes that
        are a combination of esm-sample_event, such as "coco_ut_us_election_2020".
        If the well-being change across the day of the event is added (suppl_type == add_wb_change), this
        change is calculated, added as a feature and considered in the filtering process.

        Args:
            esm_sample: str, raw name of a given ESM-sample
        """
        print(">> >> >> filter_states_num_iv << << <<")
        # Get MSE corresponding to a given ESM-sample
        event_name, event_date = next(
            (
                (event, details["date"])
                for event, details in self.state_cfg["mse"][
                    "major_societal_events"
                ].items()
                if details["sample"] == esm_sample
            ),
            (None, None),
        )
        event_date = datetime.strptime(event_date, "%Y-%m-%d")
        timespan_start = event_date + timedelta(
            days=1
        )  # start: one day after the event at 0:00 (e.g. 23.3 0:00 for COVID-19 lockdown 2020)
        timespan_end = event_date + timedelta(
            days=(self.state_cfg["mse"]["period_investigated_in_days"] + 1)
        )
        min_number_esm_measures = self.state_cfg["mse"]["min_number_esm_measures"]
        df_states_tmp = getattr(self, "state_dct")[esm_sample]
        df_traits_scale_means = getattr(self, "trait_dct")["scale_means"][
            f"{esm_sample}_one_hot_encoded"
        ]
        df_traits_single_items = getattr(self, "trait_dct")["single_items"][
            f"{esm_sample}_one_hot_encoded"
        ]
        df_states_tmp[self.esm_tp_col] = pd.to_datetime(df_states_tmp[self.esm_tp_col])
        self.logger.info(
            f"len {esm_sample} state data before filtering out rows with less than "
            f"{min_number_esm_measures} filled out surveys of the variable {event_name}: "
            f"{len(df_states_tmp)}"
        )

        if self.suppl_type == "add_wb_change":
            pre_event_day = event_date - timedelta(days=1)
            df_states_tmp = self.filter_for_pre_post_event_data(
                state_df=df_states_tmp,
                esm_tp_col=self.esm_tp_col,
                pre_event_day=pre_event_day,
                timespan_start=timespan_start,
            )
            self.add_pre_post_wb_change(
                df_states=df_states_tmp,
                esm_tp_col=self.esm_tp_col,
                df_traits_single_items=df_traits_single_items,
                df_traits_scale_means=df_traits_scale_means,
                esm_sample=esm_sample,
                pre_event_day=pre_event_day,
                timespan_start=timespan_start,
            )

        df_states_filtered = self.filter_min_num_measurements(
            state_df=df_states_tmp,
            esm_tp_col=self.esm_tp_col,
            timespan_start=timespan_start,
            timespan_end=timespan_end,
            min_number_esm_measures=min_number_esm_measures,
        )

        if self.state_cfg["mse"]["filter_for_min_num_days"][
            "apply_filter_for_num_days"
        ]:
            df_states_filtered = self.filter_min_num_days(
                state_df=df_states_filtered, esm_tp_col=self.esm_tp_col
            )

        sample_name_new = f"{esm_sample}_{event_name}"
        self.proc_esm_sample_lst.append(sample_name_new)
        self.adjust_trait_state_attributes(
            sample_name=esm_sample,
            df_states=df_states_filtered,
            new_sample_name=sample_name_new,
        )

        self.log_final_filtering_results(
            proc_sample_name=esm_sample,
            state_df=getattr(self, "state_dct")[sample_name_new],
            min_number_esm_measures=min_number_esm_measures,
            current_var=event_name,
        )
        # remove base samples
        self.state_dct.pop(esm_sample, None)
        self.trait_dct["single_items"].pop(f"{esm_sample}_one_hot_encoded", None)
        self.trait_dct["scale_means"].pop(f"{esm_sample}_one_hot_encoded", None)

    def filter_min_num_measurements(
        self,
        state_df,
        esm_tp_col,
        timespan_start,
        timespan_end,
        min_number_esm_measures,
    ):
        """
        This function filters out individuals that do not have at least a minimum number of complete
        ESM-measurements in the given time interval (timespan_end - timespan_start).

        Args:
            state_df: df containing the states for a given ESM-sample
            esm_tp_col: The column in the state df defining the point in time of the measurement
            timespan_start: Start of the post-event adjustment (from 0:00 at the day after the event)
            timespan_end: End of the post-event adjustment (timespan_start + 14 days in the paper)
            min_number_esm_measures: Min number of complete Surveys in the timespan (10 in the paper)

        Returns:
            filtered_df = Filtered state df
        """
        # Count entries per ID within the timespan
        count_per_id = (
            state_df[
                (state_df[esm_tp_col] > timespan_start)
                & (state_df[esm_tp_col] <= timespan_end)
            ]
            .groupby(self.person_id_col)
            .count()
        )
        # Filter IDs meeting the minimum ESM measure requirement
        valid_ids = count_per_id[
            count_per_id[esm_tp_col] >= min_number_esm_measures
        ].index
        # Filter the DataFrame based on valid IDs and the timespan
        filtered_df = state_df[
            (state_df[self.person_id_col].isin(valid_ids))
            & (state_df[esm_tp_col] > timespan_start)
            & (state_df[esm_tp_col] <= timespan_end)
        ]
        # Add time column
        filtered_df["days_since_event"] = (
            filtered_df[esm_tp_col] - filtered_df[esm_tp_col].min()
        ).dt.days
        return filtered_df

    def filter_min_num_days(self, state_df, esm_tp_col):
        """
        Filters the DataFrame based on a minimum number of days with at least one ESM measurement.
        In the current implementation, this should take the df that was already  filtered for a minimum
        number of general measurements as an input.

        Args:
            state_df: df containing the states for a given ESM-sample
            esm_tp_col: The column in the state df defining the point in time of the measurement

        Returns:
            filtered_df: Filtered state df
        """
        min_num_days = self.state_cfg["mse"]["filter_for_min_num_days"]["min_num_days"]
        # Count unique days per ID in filtered_df
        unique_days_per_id = state_df.groupby(self.person_id_col)[esm_tp_col].apply(
            lambda x: x.dt.floor("d").nunique()
        )
        # Filter IDs meeting the minimum number of unique days requirement
        valid_ids = unique_days_per_id[unique_days_per_id >= min_num_days].index
        # Apply the filter to the original DataFrame
        filtered_df = state_df[state_df[self.person_id_col].isin(valid_ids)]
        return filtered_df

    def filter_for_pre_post_event_data(
        self, state_df, esm_tp_col, pre_event_day, timespan_start
    ):
        """
        Filters the DataFrame to include only IDs that have at least one survey filled out on the day before
        (pre_event_day) and on the day after the event (timespan_start). This is needed for calculating the
        initial well-being change as a feature.

        Args:
            state_df: df containing the states for a given ESM-sample
            esm_tp_col: str, name of the column containing the point in time of a given measurement
            pre_event_day: dt, Day before the event (as defined in the config)
            timespan_start: dt, The start of the post-event adjustment timespan (post-event day) to check for each ID.

        Returns:
            filtered_df: Filtered state df
        """
        # Determine valid IDs based on the presence of specific dates
        valid_ids = state_df.groupby(self.person_id_col).apply(
            lambda group: pre_event_day.date() in group[esm_tp_col].dt.date.values
            and timespan_start.date() in group[esm_tp_col].dt.date.values
        )
        # Get the list of IDs that meet the criteria
        wb_change_ids = valid_ids[valid_ids].index.tolist()
        # Filter the DataFrame to include only rows with valid IDs
        filtered_df = state_df[state_df[self.person_id_col].isin(wb_change_ids)]
        return filtered_df

    def add_pre_post_wb_change(
        self,
        df_states,
        esm_tp_col,
        df_traits_scale_means,
        df_traits_single_items,
        esm_sample,
        pre_event_day,
        timespan_start,
    ):
        """
        This function adds the wb_change (aggregated on a day level) from the day before the event to the day after
        the event as a person-level predictor for the predictive analysis.
        This is only executed if suppl_type == 'add_wb_change'.
        It filters the trait and state dfs accordingly and updates the class attributes.

        Args:
            df_states: df containing the states for a given ESM-sample
            esm_tp_col: str, name of the column containing the point in time of a given measurement
            df_traits_scale_means: df containing the person-level features as scale means for the esm_sample
            df_traits_single_items: df containing the person-level features as single items for the esm_sample
            esm_sample: Given esm_sample, e.g., "coco_int"
            pre_event_day: dt, Day before the event (as defined in the config)
            timespan_start: dt, The start of the post-event adjustment timespan (post-event day) to check for each ID.
        """
        df_states["date"] = df_states[esm_tp_col].dt.date
        filtered_df = df_states[
            df_states["date"].isin([timespan_start.date(), pre_event_day.date()])
        ]
        avg_scores = filtered_df.groupby([self.person_id_col, "date"]).mean()

        # Reset index to pivot
        avg_scores_reset = avg_scores.reset_index()
        pivot_df = avg_scores_reset.pivot(
            index=self.person_id_col, columns="date", values="wb_score"
        )
        # Calculate the change in wb_score between the two dates
        pivot_df["wb_change_pre_post_event"] = (
            pivot_df[timespan_start.date()] - pivot_df[pre_event_day.date()]
        )
        wb_change_df = pivot_df.reset_index()[["id", "wb_change_pre_post_event"]]

        # Merge with trait_df subsets
        df_traits_scale_means_filtered = df_traits_scale_means.merge(
            wb_change_df, on="id", how="left"
        )
        df_traits_scale_means_filtered = df_traits_scale_means_filtered.dropna(axis=0)
        getattr(self, "trait_dct")["scale_means"][
            f"{esm_sample}_one_hot_encoded"
        ] = df_traits_scale_means_filtered
        df_traits_single_items_filtered = df_traits_single_items.merge(
            wb_change_df, on="id", how="left"
        )
        df_traits_single_items_filtered = df_traits_single_items_filtered.dropna(axis=0)
        getattr(self, "trait_dct")["single_items"][
            f"{esm_sample}_one_hot_encoded"
        ] = df_traits_single_items_filtered
