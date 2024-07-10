import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import PurePath

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import yaml
from scipy.stats import chi2


class MultilevelModeling:
    """
    Class for the multilevel analysis. This class is mainly used for extracting the empirical
    Bayes estimates of the individual random slopes that will function as the criterion in the
    machine learning based prediction procedure.
    In Addition, it calculates descriptive statistics where multilevel models must be estimated
    (e.g., ICCs) and returns the reliability estimates of the individual reactivities that
    are used to weight the samples in the associated supplementary analysis.

    Attributes:
        base_path: str, rel path where the processed data (Output of the Preprocessor Class) is stored
        states: Dict where the preprocessed state dfs are stored for the ESM-samples specified in config
        random_effects: Dict where the individual EB estimates are stored for each ESM-Sample
        rel_sample_weights: Dict where the sample weights (the estimated individual reliabilities) are stored
        ols_slopes: Dict where the OLS slopes (results from linear regressions per person) are stored
        random_slope_only_results: Dict where key results of the multilevel models used for the analysis are stored.
            This includes average effects + credible intervals, random slope SD + credible intervals, standardized
            average effects and the average rel of individual reactivities across samples (see Table 5 in paper).
        fixed_slope_only_results: Stores the results of a OLS on the population level for conducting LR tests.
    """

    def __init__(self, config_path):
        """
        Constructor method of the MultilevelModeling Class.

        Args:
            config_path: str, relative path to the config file
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.base_path = self.config["general"]["load_data"]["processed_data_path"]
        self._setup_logger()
        self.states = dict()
        self.random_effects = dict()
        self.rel_sample_weights = dict()
        self.ols_slopes = dict()
        self.random_slope_only_results = dict()
        self.fixed_slope_only_results = dict()

    @property
    def samples_for_analysis(self):
        """List of samples used for the current analysis."""
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
            if self.suppl_type
            and self.suppl_type not in ["add_wb_change", "mse_no_day_agg"]
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
            if self.suppl_type
            in ["sep_ftf_cmc", "sep_pa_na", "add_wb_change", "mse_no_day_agg"]
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
        for the supplementary analysis 'weighting_by_rel', we use the same data as an in main analysis
        up to the Multilevel Modeling.
        """
        return self.suppl_var if self.suppl_type != "weighting_by_rel" else None

    @property
    def person_id_col(self):
        """Name of the person_id column in the datasets."""
        return self.config["general"]["id_col"]["all"]

    @property
    def mlm_solver(self):
        """Set mlm solver dynamically depending on whether we investigate mse or ssc."""
        return self.config["analysis"]["mlm_params"]["solver"][self.study]

    @property
    def data_path(self):
        """
        This method sets the path for
            a) loading the states for generating the multilevel models and
            b) storing the extracted random effects.
        This is based on the analysis type (main / suppl).
        An example of the level of abstraction would be "../data/preprocessed/add_wb_change/mse".
        """
        path_components = [
            self.base_path,
            self.analysis_level_path,
            self.suppl_type_level_path,
            self.suppl_var_level_path,
            self.study,
        ]
        filtered_path_components = [comp for comp in path_components if comp]
        return os.path.normpath(os.path.join(*filtered_path_components))

    @property
    def logs_base_path(self):
        """Base folder for logging, same for subclasses."""
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../..", "logs"
        )

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
        """This creates the filename of the log file based on self-log_specific_path and the current time."""
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        return os.path.join(
            self.logs_specific_path, f"multilevel_modeling_{current_time}.log"
        )

    @property
    def wb_items(self):
        """Sets a Dict with esm_samples and corresponding well-being items as class attribute."""
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
    def soc_int_vars(self):
        """Sets a Dict with esm_samples and corresponding soc_int_vars as class attribute (only in SSC)."""
        if self.study == "ssc":
            soc_int_var_dct = dict()
            for esm_sample in self.samples_for_analysis:
                soc_int_var_dct[esm_sample] = [
                    soc_int_var
                    for soc_int_var, vals in self.config["state_data"]["ssc"][
                        "social_interaction_vars"
                    ].items()
                    if esm_sample in vals["samples"]
                ]
                # add interaction_quantity in coco_ut / ftf
                if esm_sample == "coco_ut" and self.suppl_var == "ftf":
                    soc_int_var_dct[esm_sample].append("interaction_quantity")
            return soc_int_var_dct
        else:
            return None

    def apply_preprocessing_methods(self):
        """This function applies the preprocessing methods specified in the config."""
        for method in self.config["analysis"]["mlm_methods"]:
            if method not in dir(MultilevelModeling):
                raise ValueError(f"Method '{method}' is not implemented yet.")
            getattr(self, method)()

    def _setup_logger(self):
        """
        This sets up the logger, configures it and sets it as a class attribute. It logs e.g. the results
        of the multilevel models and the associated formulas.
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

    def get_data(self):
        """This function loads the dataframes stored as .pkl files in the given data directory."""
        data_path_analysis_mlm = os.path.join(self.data_path, "states")
        if not os.path.exists(data_path_analysis_mlm):
            raise ValueError(f"The path {data_path_analysis_mlm} does not exist.")
        # Iterate over all files in the directory
        for filename in os.listdir(data_path_analysis_mlm):
            if any(
                s in filename for s in self.config["general"]["samples_for_analysis"]
            ):
                filepath = os.path.join(data_path_analysis_mlm, filename)
                df = pd.read_pickle(filepath)
                key = filename.rsplit("_preprocessed", 1)[0]
                getattr(self, "states")[key] = df
        print(f"Loaded {len(self.states)} dataframes from {data_path_analysis_mlm}")

    def get_iccs(self):
        """
        This function is used to calculate a random intercept only model for (a) the well-being items and (b) social
        situation variables (if study == ssc) to get the ICCs for the table containing descriptive information.
        Therefore, we have to use the unstandardized variables. In ssc, we do this for all social situation variables.
        If specified, the ICCs are stored as a JSON file.
        """
        # wb_items
        result_dct_wb_items = dict()
        for (
            sample_name
        ) in self.states.keys():  # this includes all soc_int_var - sample combinations
            result_dct_wb_items[sample_name] = dict()
            raw_sample_name = [
                i for i in self.samples_for_analysis if i in sample_name
            ][0]
            df_states = getattr(self, "states")[sample_name]
            wb_items = self.wb_items[raw_sample_name]
            for item in wb_items:
                df_tmp = df_states.dropna(subset=[item])
                result_dct_wb_items[sample_name][item] = self.calc_rs_mlm(
                    df=df_tmp, var=item
                )
        if self.config["analysis"]["mlm_results"]["store_iccs"]:
            self.store_mlm_result_dct(
                result_dct=result_dct_wb_items, name_suffix=f"iccs_wb_items"
            )
        # soc_int_vars
        if self.study == "ssc":
            result_dct_soc_int_vars = dict()
            for (
                sample_name
            ) in (
                self.states.keys()
            ):  # this includes all soc_int_var - sample combinations
                result_dct_soc_int_vars[sample_name] = dict()
                raw_sample_name = [
                    i for i in self.samples_for_analysis if i in sample_name
                ][0]
                df_states = getattr(self, "states")[sample_name]
                soc_int_var = [
                    var
                    for var in self.soc_int_vars[raw_sample_name]
                    if var in sample_name
                ][0]
                result_dct_soc_int_vars[sample_name][soc_int_var] = self.calc_rs_mlm(
                    df=df_states, var=soc_int_var
                )
            if self.config["analysis"]["mlm_results"]["store_iccs"]:
                self.store_mlm_result_dct(
                    result_dct=result_dct_soc_int_vars, name_suffix=f"iccs_soc_int_vars"
                )

    def calc_ols_for_sample(self):
        """
        This calculates an OLS regression for the whole sample for comparison with the random slope-only model.
        IF specified, the results of the OLS regression are stored as a JSON file.
        """
        ols_result_dct = dict()
        for df_name, df in self.states.items():
            ols_result_dct[df_name] = dict()
            iv, dv = self.get_iv_dv(df_name)
            df_cleaned = df.dropna(subset=[iv, dv])
            if len(df_cleaned) != len(df):
                print(f"removed {len(df) - len(df_cleaned)} rows with NaN values")
            results_ols = self.calc_ols(iv, dv, df_cleaned, return_slope=False)
            results_to_json = self.create_ols_result_dct(
                ols_result_dct[df_name], results_ols
            )
            ols_result_dct[df_name] = results_to_json
            self.fixed_slope_only_results[df_name] = results_ols

        if self.config["analysis"]["mlm_results"]["store_ols_results_in_json"]:
            self.store_mlm_result_dct(ols_result_dct, "ols_results")

    def calc_ols_for_individuals(self):
        """
        This calculates a single OLS regression for each individual and extracts the slope. These slopes
        are used in the supplementary analysis where the individual reactivities are weighted by their estimated
        reliability, because the formula provided by Neubauer et al. (2020) is only an approximation of the
        reliability for the EB estimates, but it accurately describes the reliability of OLS slopes.
        If specified, the results of the individual OLS regressions are stored as a JSON file.
        """
        for df_name, df in self.states.items():
            iv, dv = self.get_iv_dv(df_name)
            df_cleaned = df.dropna(subset=[iv, dv])
            if len(df_cleaned) != len(df):
                print(f"removed {len(df) - len(df_cleaned)} rows with NaN values")

            slopes = pd.Series(dtype="float64")
            # Perform Regression for Each Group
            for person, person_data in df_cleaned.groupby(self.person_id_col):
                results_ols, slope = self.calc_ols(
                    iv, dv, person_data, return_slope=True
                )
                # Store the slope in the Series
                slopes.at[person] = slope
            # Add slopes to class attribute
            self.ols_slopes[df_name] = slopes

            if self.config["analysis"]["mlm_params"]["store"]["ols_slopes"]:
                self.store_files(file_type="ols_slopes")

    @staticmethod
    def calc_ols(iv, dv, df, return_slope=False):
        """
        This function calculates an ols model with the given parameters and returns the results.

        Args:
            iv: independent variable, defined in the config
            dv: dependent variable, defined in the config
            df: pd.DataFrame containing the state data of a given ESM-sample
            return_slope: if true, the coefficient of the slope will be returned

        Returns:
            results_ols: Results of the OLS model
            results_ols.params[iv]: Coefficient of the slope
        """
        ols_model = sm.OLS(df[dv], sm.add_constant(df[iv]))
        results_ols = ols_model.fit()
        if return_slope:
            return results_ols, results_ols.params[iv]
        else:
            return results_ols

    def calc_rs_mlm(self, df, var):
        """
        This function calculates a multilevel model with a random intercept to use it e.g. for ICC calculation.

        Args:
            df: pd.DataFrame containing the state data of a given ESM-sample
            var: State variable for which the model is calculated

        Returns:
            np.round(icc, 2): Two-decimal rounded ICC for the given configuration
        """
        mlm = smf.mixedlm(
            formula=f"{var} ~ 1",
            data=df,
            groups=df[self.person_id_col],
        )
        results_mlm = mlm.fit(
            maxiter=self.config["analysis"]["mlm_params"]["max_iter"], reml=True
        )
        icc = self.get_icc(results_mlm)
        return np.round(icc, 2)

    @staticmethod
    def get_icc(results):
        """
        Gets the Intraclass Correlation Coefficient (ICC) from the statsmodels results object.

        Args:
            results: Results from a given random intercept only multilevel model

        Returns:
            icc.values[0, 0]: ICC for a given model configuration
        """
        icc = results.cov_re / (results.cov_re + results.scale)
        return icc.values[0, 0]

    def calc_rs_only_mlm(self):
        """
        This function calculates a random slope only multilevel model for the given dv and iv.
        It further extracts the BLUB random effects that function as reactivities in the machine learning analysis,
        calculates standardized within-person effects, and calculates the reliability of the individual reactivities.
        All results are stored if specified.
        """
        self.logger.info(".")
        self.logger.info(
            ">>> Calculate random slop only multilevel models for all samples"
        )
        mlm_result_dct = dict()
        for df_name, df in self.states.items():
            self.logger.info(".")
            self.logger.info(
                f">>> Calculate random slop only multilevel model for {df_name}"
            )
            mlm_result_dct[df_name] = dict()
            iv, dv = self.get_iv_dv(df_name)

            # there should be no NaNs in the columns of importance, but if it is the case, remove them
            df_cleaned = df.dropna(subset=[iv, dv])
            if len(df_cleaned) != len(df):
                print(f"removed {len(df) - len(df_cleaned)} rows with NaN values")

            # set mlm formulas dynamically for more flexibility
            mlm_formula = f"{dv} ~ {iv}"
            mlm_re_formula = f"0 + {iv}"
            self.logger.info(f">>> mlm_formula: {mlm_formula}")
            self.logger.info(f">>> re_formula: {mlm_re_formula}")

            mlm = smf.mixedlm(
                formula=mlm_formula,
                data=df_cleaned,
                groups=df_cleaned[self.person_id_col],
                re_formula=mlm_re_formula,
            )
            results_mlm = mlm.fit(
                method=self.mlm_solver,  # set solver dynamically depending on the study
                maxiter=self.config["analysis"]["mlm_params"]["max_iter"],
                reml=self.config["analysis"]["mlm_params"]["reml"],
            )
            print(df_name)
            print(results_mlm.summary())
            self.logger.info(results_mlm.summary())
            self.logger.info("\n")
            self.random_slope_only_results[df_name] = results_mlm

            # Calculate and store EB of the individual random slopes
            eb_series = self.calc_empirical_bayes_estimates(results_mlm=results_mlm)
            self.random_effects[df_name] = eb_series
            if self.config["analysis"]["mlm_params"]["store"]["random_effects"]:
                self.store_files(file_type="random_effects")

            # Calculate reliabilities of the reactivities (see Neubauer et al., 2020), if specified
            if self.config["analysis"]["mlm_params"]["calc_rel"]:
                mean_rel = self.calc_rel(
                    df_name=df_name,
                    df=df_cleaned,
                    results_mlm=results_mlm,
                    person_id=self.person_id_col,
                    iv=iv,
                )
            else:
                mean_rel = None

            # Calculate standardized within person effects
            if self.config["analysis"]["mlm_params"]["calc_betas"]:
                beta = self.calc_betas(results_mlm=results_mlm, df=df_cleaned)
            else:
                beta = None

            # Create final result Dict
            mlm_result_dct[df_name] = self.create_mlm_result_dct(
                mlm_result_dct[df_name], results_mlm, iv, mean_rel, beta
            )
            self.logger.info(f"random_effects - mean: {np.round(eb_series.mean(), 6)}")
            self.logger.info(
                f"random_effects - descriptive sd: {np.round(eb_series.std(), 6)}"
            )
            self.logger.info(
                f"mlm - cov matrix of random effects: {results_mlm.cov_re}"
            )
            print(f"extracted random effects for {df_name}")

        if self.config["analysis"]["mlm_results"]["store_mlm_results_in_json"]:
            self.store_mlm_result_dct(mlm_result_dct, "mlm_results")

    @staticmethod
    def calc_empirical_bayes_estimates(results_mlm):
        """
        This function calculates the empirical Bayes estimates of the individual random slopes.

        Args:
            results_mlm: Results of a random slope only multilevel model

        Returns:
            eb_series: pd.Series containing the Person-ID as index and the EB estimate as data
        """
        fe = results_mlm.fe_params
        re = results_mlm.random_effects
        eb_estimates = {}
        for person, rand_effect in re.items():
            # The total effects for each group is the sum of fixed effects and its random effects
            eb_estimates[person] = [fe[k] + rand_effect[k] for k in rand_effect.keys()][
                0
            ]
        eb_series = pd.Series(
            data=[val for val in eb_estimates.values()],
            index=[person for person in eb_estimates],
        )
        return eb_series

    def calc_betas(self, df, results_mlm):
        """
        This function calculates the standardized within-person effect using the approach from
        Fitzmaurice et al. (2011). Persons that show no within-person variance on the variables of
        interest are excluded.

        Args:
            df: pd.Dataframe containing the states of a given ESM sample
            results_mlm: Results of a random slope only multilevel model

        Returns:
            np.mean(stand_slopes): the average standardized within-person effect
        """
        fe = results_mlm.fe_params
        re = results_mlm.random_effects
        stand_slopes = []
        dv = self.config["analysis"]["mlm_params"]["dv"]
        for person, rand_effect in re.items():
            ind_slope = [fe[k] + rand_effect[k] for k in rand_effect.keys()][0]
            within_person_var = np.std(df[df[self.person_id_col] == person][dv])
            if within_person_var == 0:
                print("no within person variance for person with the id", person)
                continue
            stand_slopes.append(ind_slope / within_person_var)
        return np.mean(stand_slopes)

    def calc_rel(self, df_name, df, results_mlm, person_id, iv):
        """
        This function calculates the reliability of the extracted random effects using the formula
        provided by Neubauer et al. (2020) using the terminology from the paper:
            o_e_2 -> Level-1 residual variance (in statsmodels: scale)
            t_2 -> random slope variance (if using param_object.cov_re, this has to be multiplied by scale in
                python to get the results from R.)
            o_x_2 -> within person variance of the predictor, has to be estimated from a RI-only model or from the data
                  Because of the person mean centering of the dependent variable, we estimate it from the data
                  This is equivalent to the residual variance of the non-fitting intercept only multilevel with the iv
                  as criterion.
        It stores the individual reliabilities (if specified), updates the corresponding class attribute,
        and returns the mean reliability for a given sample

        Args:
            df_name: Key of the dct of the class attribute storing the state data, contains the sample
                and the social situation variable (e.g., coco_int_social_interaction) or the major societal event
            df: pd.DataFrame containing the state data for a given ESM sample
            results_mlm: Results of a random slope only multilevel model
            person_id: Column that identifies if state observations belong to the same person
            iv: Independent variable used in results_mlm

        Returns:
            np.mean(rel_series): Average sample reliability of individual reactivities
        """
        o_e_2 = results_mlm.scale
        t_2 = results_mlm.params_object.cov_re[0][0] * results_mlm.scale
        o_x_2 = df[iv].std() ** 2

        result_dct = dict()
        for person in df[person_id].unique():
            num_measurements = (
                df[df[person_id] == person].groupby([person_id]).count().iloc[0, 0]
            )
            rel = t_2 / (t_2 + (o_e_2 / ((num_measurements - 1) * o_x_2)))
            result_dct[person] = rel
        rel_series = pd.Series(data=result_dct.values(), index=result_dct.keys())
        # set weights series corresponding to the reliabilities as attribute
        self.rel_sample_weights[df_name] = rel_series
        if self.config["analysis"]["mlm_params"]["store"]["sample_weights"]:
            self.store_files(file_type="rel_sample_weights")
        return np.mean(rel_series)

    def store_files(self, file_type):
        """
        This function stores a pd.Series as a pickle file. Currently, this is used for storing the extracted
        random effect coefficients, the individual ols slopes as well as the calculated samples weights.
        In contrast to the files that are stored as .JSON, these files are used in further analysis.
        It further checks if "REML" estimation was enabled when extracting the random slopes.

        Args:
            file_type: String specifying which files are stored, guided the storing process, should be in
                [rel_sample_weights, random_effects, ols_slopes] and of type pd.Series
        """
        file_path = os.path.join(self.data_path, file_type)
        if file_type in ["rel_sample_weights", "random_effects"]:
            assert self.config["analysis"]["mlm_params"][
                "reml"
            ], "We use REML for estimating the random effects we store"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for sample_name, series in getattr(
            self, f"{file_type}"
        ).items():  # self.rel_weights.items():
            file_name = os.path.join(file_path, f"{sample_name}.pkl")
            with open(file_name, "wb") as f:
                pickle.dump(series, f)
            print(
                f">>>> Stored pd.Series for {sample_name} containing {file_name} in {file_path} <<<<"
            )

    def compare_fs_rs_models(self):
        """
        This function conducts a likelihood ratio test for comparing fixed and random slopes models.
        Note: We have to compute the test manually, because the OLS model and the multilevel model are of
        different object type. For a correct likelihood ratio test, models must be fitted using maximum
        likelihood, not REML. If specified, it stores the results accordingly in a .JSON file.
        """
        lr_test_dct = dict()
        for df_name, mlm_results, ols_results in zip(
            self.random_slope_only_results.keys(),
            self.random_slope_only_results.values(),
            self.fixed_slope_only_results.values(),
        ):
            assert not self.config["analysis"]["mlm_params"][
                "reml"
            ], "For LR test, the multilevel model must be fitted with ML and not REML"
            lr_test_dct[df_name] = dict()
            print(f"LL ols: {ols_results.llf}, LL mlm: {mlm_results.llf}")
            LR_stat = -2 * (ols_results.llf - mlm_results.llf)
            df = 1  # ols _results do not count the intercept as a parameter, mixedlm does
            p_value = chi2.sf(LR_stat, df)
            lr_test_dct[df_name]["LR_stat"] = np.round(LR_stat, 2)
            lr_test_dct[df_name]["df"] = df
            lr_test_dct[df_name]["p_value"] = np.round(p_value, 4)  # achtung APA
        if self.config["analysis"]["mlm_results"]["store_lr_test_results_in_json"]:
            self.store_mlm_result_dct(lr_test_dct, "lr_test_results")

    @staticmethod
    def create_ols_result_dct(result_dct, results_ols):
        """
        This function adds ols parameters I report in the paper in a Dict and returns the Dict.

        Args:
            result_dct: Result Dict that will be changed inplace by this function
            results_ols: Results of an OLS regression from statsmodels

        Returns:
              result_dct: Result Dict containing the parameter names as keys and the corresponding values
        """
        result_dct["intercept"] = np.round(results_ols.params.tolist()[0], 5)
        result_dct["p_intercept"] = np.round(results_ols.pvalues.tolist()[0], 5)
        result_dct["slope"] = np.round(results_ols.params.tolist()[1], 5)
        result_dct["p_slope"] = np.round(results_ols.pvalues.tolist()[1], 5)
        result_dct["intercept_se"] = np.round(results_ols.bse.tolist()[0], 5)
        result_dct["slope_se"] = np.round(results_ols.bse.tolist()[1], 5)
        result_dct["residual_var"] = np.round(results_ols.scale.tolist(), 5)
        return result_dct

    @staticmethod
    def create_mlm_result_dct(result_dct, results_mlm, iv, mean_rel=None, beta=None):
        """
        This function adds multilevel results parameters I report in the paper in a Dict and returns the Dict.
        # Note: In contrast to the method "calc_rel", we used results_mlm.cov_re, this is already the scaled
        version (equivalent to R output), so we do not have to multiply by scale.

        Args:
            result_dct: Given Dict storing the results
            results_mlm: Results of a random slope only multilevel model
            iv: Column name of the independent variable of the multilevel model
            mean_rel: Average sample reliability of individual reactivities
            beta: Average standardized sample reactivities

        Returns:
            result_dct: Updated Dict containing the parameter names as keys and the corresponding values
        """
        result_dct["fe_slope"] = np.round(
            results_mlm.params_object.fe_params.tolist()[1], 3
        )
        result_dct["cred_interval_slope"] = [
            np.round(var, 5) for var in results_mlm.conf_int().loc[iv]
        ]
        result_dct["p_slope"] = np.round(results_mlm.pvalues.tolist()[1], 5)

        result_dct["random_slope_sd"] = np.round(
            np.sqrt(results_mlm.cov_re.iloc[0, 0]), 3
        )
        result_dct["slope_sd_cred_interval"] = [
            np.round(np.sqrt(var * results_mlm.scale), 3)
            for var in results_mlm.conf_int().loc[f"{iv} Var"]
        ]
        result_dct["residual_var"] = np.round(results_mlm.scale.tolist(), 5)
        if mean_rel:
            result_dct["mean_rel"] = np.round(mean_rel, 2)
        if beta:
            result_dct["slope_beta"] = np.round(beta, 5)
        return result_dct

    def store_mlm_result_dct(self, result_dct, name_suffix):
        """
        This is used to store different mlm results as a .JSON file.

        Args:
            result_dct: Final result dict containing all results we want to store
            name_suffix: String for defining the filename of the results
        """
        path_components = self.get_path_components(self.data_path)
        file_name = (
            "_".join(part for part in path_components[3:]) + f"_{name_suffix}.json"
        )
        file_path = os.path.join(
            self.config["analysis"]["mlm_results"]["json_result_folder"], file_name
        )
        with open(file_path, "w") as f:
            json.dump(result_dct, f, indent=4)

    @staticmethod
    def get_path_components(path_str):
        """Returns the path components from a given path provided as a string as a tuple of strings.

        Args:
            path_str: Given path, e.g. "../data/preprocessed/main/"

        Returns:
            PurePath(path_str).parts: Path components as a tuple of separated strings, e.g. ("..", "data", "processed")
        """
        return PurePath(path_str).parts

    def get_iv_dv(self, df_name):
        """
        This function gets the independent and the dependent variable name for the multilevel/OLS models.

        Args:
            df_name: Key of the dct of the class attribute storing the state data, contains the sample
                and the social situation variable (e.g., coco_int_social_interaction) or the major societal event

        Returns:
            iv: Column name of the independent variable
            dv: Column name of the dependent variable
        """
        if self.study == "ssc":
            # extract social int var from filename
            iv = "_".join(df_name.split("_")[-2:]) + "_pmc"
        elif self.study == "mse":
            if self.suppl_type == "mse_no_day_agg":
                iv = "hours_since_event"
            else:
                iv = "days_since_event"
        else:
            raise ValueError("Study not implemented")
        dv = self.config["analysis"]["mlm_params"]["dv"]
        return iv, dv
