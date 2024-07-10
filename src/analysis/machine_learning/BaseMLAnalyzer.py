import json
import os
import time
import warnings
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn import get_config
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    make_scorer,
    mean_squared_error,
    get_scorer,
)
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.analysis.machine_learning.CustomScaler import CustomScaler


class BaseMLAnalyzer(ABC):
    """
    Abstract base class for the machine learning-based prediction procedure. This class serves as a template
    for the model specific class implementations. It encapsulates basic functionality of the repeated-nested
    cross validation procedure and the feature importance analysis that is model independent.

    Attributes:
        output_dir: str, Specific directory where the result files of the machine learning analysis are stored.
            This depends on multiple parameters, such as the analysis_type (main / suppl), the study (ssc/mse),
            the esm-sample, the feature inclusion strategy, the prediction model, and in SSC the social situation
            variable. An example would be "../results/main/ssc/coco_int/scale_means/lasso/social_interaction".
                If the models are computed locally, the root directory is defined in the config.
                If the models are computed on a cluster, the root directory is defined in the SLURM script.
            Construction of further path components (e.g. "main/ssc/coco_int/scale_means/lasso/social_interaction")
            follow the same logic, both locally and on a cluster.
        model: str, prediction model for a given analysis, defined through the subclasses.
        best_models: Dict that collects the number of repetitions of the repeated nested CV as keys and a list
            of best models (grid_search.best_estimator_.named_steps["model"]) obtained in the nested CV as values
            during the machine learning analysis.
        repeated_nested_scores: Nested Dict that collects the number of repetitions as keys in the outer nesting,
            the metrics as keys in the inner nesting and the values obtained in the hold-out test sets in one outer
            fold as a list of values in the inner nesting.
                Example: {"rep_1":{"r2":[0.05, 0.02, 0.03]}}
        shap_values: Nested Dict that collects the SHAP values summarized across outer folds per repetition seperated
            for the train and the test set. SHAP values are of shape (n_samples, n_features).
                Example: {"train": {"rep_1": [n_samples, n_features]}}
        shap_ia_values: Nested Dict that collects aggregates of SHAP interaction values summarized across outer folds
            and repetitions separated for the train and test set. Because storing the raw interaction SHAP values would
            have been to memory intensive (a n_samples x n_samples x n_features tensor), we already calculated
            meaningful aggregations on the cluster (e.g., summarizing across).
                Example: {"train": {"agg_ia_persons": {"age_clean": { "age_clean": 112.43, 1.59}}}}
        config: YAML config determining certain specifications of the analysis. Is used on the cluster and locally.
            In contrast to the other classes, the config gets updated dynamically on certain conditions (e.g.,
            number of inner and outer folds is always set to 10 when running on a cluster to prevent errors caused
            by manual adjusting of the config for local testing).
        sample_weights: [pd.Series, None], The individual reliabilities calculated in the MultilevelModeling Class
            used to weight the individual samples in the machine learning based prediction procedure. Is None, if
            suppl_type != weighting_by_rel. If suppl_type == weighting_by_rel, the sample weights are loaded from
            its directory and used in the repeated nested CV using sklearns metadata_routing.
        pipeline: sklearn.pipeline.Pipeline, pipeline defining the steps that are applied inside the repeated-nested
            CV. This includes preprocessing (i.e., scaling), recursive feature elimination (if feature inclusion
            strategy == feature_selection) and the prediction model
        X: pd.df, All features for a given ESM-sample and feature inclusion strategy.
        y: pd.Series, All criteria for a given ESM-sample (thus, the individual reactivity estimates).
    """

    @abstractmethod
    def __init__(
        self,
        config,
        output_dir,
    ):
        """
        Constructor method of the BaseMLAnalyzer Class.

        Args:
            config: YAML config determining specifics of the analysis
            output_dir: Specific directory where the results are stored
        """
        self.output_dir = output_dir
        self.model = None
        self.best_models = dict()
        self.repeated_nested_scores = dict()
        self.shap_values = {"train": {}, "test": {}}
        self.shap_ia_values = {"train": {}, "test": {}}
        self.config = config  # passed in main.py
        self.sample_weights = None
        self.pipeline = None
        self.X = None
        self.y = None

    @property
    def data_base_path(self):
        """Root dir where the data for analysis is stored (features, criteria, sample_weights)."""
        return os.path.normpath(
            self.config["general"]["load_data"]["processed_data_path"]
        )

    @property
    def feature_inclusion_strategy(
        self,
    ):  # single_items, scale_means, feature_selection
        """Feature inclusion strategy for the current analysis."""
        return self.config["general"]["feature_inclusion_strategy"]

    @property
    def raw_sample_name(self):  # coco_int, emotions, coco_ut
        """Raw ESM-sample name of the current analysis."""
        return self.config["general"]["esm_sample"]

    @property
    def study_name(self):  # ssc or mse
        """Name of the Study of the current analysis, Study 1 (ssc) or Study 2 (mse)."""
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
    def crit_var(self):
        """Determines which criterion to use (random effects or ols slopes)."""
        return "ols_slopes" if self.suppl_var == "ols_slopes" else "random_effects"

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
        for the supplementary analysis 'weighting_by_rel', we use the same data as an in main analysis up to
        the Multilevel Modeling.
        """
        return self.suppl_var if self.suppl_type != "weighting_by_rel" else None

    @property
    def soc_int_var(self):
        """Define the social interaction variable if study is 'ssc', None if study is 'mse'."""
        return (
            self.config["general"]["social_interaction_variable"]
            if self.study_name == "ssc"
            else None
        )

    @property
    def data_inter_path(self):
        """
        Sets the intermediate path where the criterion (random_effects / ols_slopes) and the features (df with traits)
        are stored as a class attribute. Therefore, it combines the data_base_path (where all preprocessed data is
        stored) with the information on the analysis type and the Study.
        Examples would be "../data/preprocessed/main/mse" or "../data/preprocessed/sep_ftf_cmc/ftf/ssc".
        """
        path_components = [
            self.data_base_path,
            self.analysis_level_path,
            self.suppl_type_level_path,
            self.suppl_var_level_path,
            self.study_name,
        ]
        # Filter out empty or None values
        filtered_path_components = [comp for comp in path_components if comp]
        return os.path.normpath(os.path.join(*filtered_path_components))

    @property
    def feature_folder(self):
        """
        Sets the folder where the features for the specific analysis are stored as a class attribute.
        An example would be "../data/preprocessed/main/mse/traits/scale_means"
        """
        if self.feature_inclusion_strategy in ("single_items", "feature_selection"):
            fis_subfolder_name = "single_items"
        elif self.feature_inclusion_strategy == "scale_means":
            fis_subfolder_name = "scale_means"
        else:
            raise ValueError("Feature inclusion strategy not implemented")
        feature_folder = os.path.normpath(
            os.path.join(self.data_inter_path, "traits", fis_subfolder_name)
        )
        assert os.path.exists(feature_folder), f"{feature_folder} does not exist"
        return feature_folder

    @property
    def crit_folder(self):
        """
        Sets the folder where the criterion (e.g., random slopes) for the specific analysis are stored.
        An example would be "../data/preprocessed/main/mse/random_effects"
        """
        crit_folder = os.path.normpath(
            os.path.join(self.data_inter_path, self.crit_var)
        )
        assert os.path.exists(crit_folder), f"{crit_folder} does not exist"
        return crit_folder

    @property
    def model_name(self):
        """Get a string repr of the model name and sets it as class attribute (e.g., "lasso")."""
        return self.model.__class__.__name__.lower()

    @property
    def num_cv(self):
        """Number of inner and outer cv, always equal in our application (e.g., 10)."""
        return self.config["analysis"]["cross_validation"]["num_cv"]

    @property
    def num_repeats(self):
        """Number of repetitions of the nested cv procedure with different data partition (e.g., 10)."""
        return self.config["analysis"]["cross_validation"]["repetitions"]

    @property
    def hyperparameter_grid(self):
        """Set hyperparameter grid defined in config for the specified model as class attribute."""
        return self.config["analysis"]["model_hyperparameters"][self.model_name]

    @property
    def use_sample_weights(self):
        """Bool, True if weighting_by_rel, False otherwise, also set as class attribute."""
        return (
            True
            if self.config["general"]["analysis"] == "suppl"
            and self.config["general"]["suppl_type"] == "weighting_by_rel"
            and self.config["general"]["feature_inclusion_strategy"]
            in ["single_items", "scale_means"]
            else False
        )

    @property
    def weights_folder(self):
        """Specify folder to get the sample weights (if self.use_sample_weights is True)."""
        return os.path.normpath(
            os.path.join(self.data_inter_path, "rel_sample_weights")
        )

    def apply_methods(self):
        """This function applies the preprocessing methods specified in the config."""
        for method in self.config["analysis"]["machine_learning_methods"]:
            if method not in dir(BaseMLAnalyzer):
                raise ValueError(f"Method '{method}' is not implemented yet.")
            getattr(self, method)()

    @staticmethod
    def time_function_execution(class_method):
        """
        This function keep track of the time of other class methods. It can be used as a decorator.
        Currently, this is only used for repeated-nested CV (because this includes all other computationally
        intensive functions.). The results are stored as hours:minutes:seconds and also include the current
        time when the decorated function is finished.
        Example: {
            "execution_time": "repeated_nested_cv executed in 00h 03m 17.00s",
            "datetime": "2024-02-16 14:01:04"
            }

        Args:
            class_method: str, the method that is decorated by this function.
        Returns:
            wrapper: Inside function of the decorator that takes the arguments of the function
                that is decorated
        """

        def wrapper(self, *args, **kwargs):
            start = time.time()
            result = class_method(self, *args, **kwargs)
            end = time.time()
            hours, remainder = divmod(end - start, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_statement = f"{class_method.__name__} executed in {int(hours):02}h {int(minutes):02}m {seconds:.2f}s"
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            time_data = {
                "execution_time": time_statement,
                "datetime": formatted_datetime,
            }
            time_filename = os.path.join(self.output_dir, "time.json")
            os.makedirs(os.path.dirname(time_filename), exist_ok=True)
            with open(time_filename, "w") as file:
                json.dump(time_data, file, indent=4)
            return result

        return wrapper

    def adjust_metadata_routing(self):
        """This function sets metadata_routing to true or false in sklearn depending on the fis."""
        if self.feature_inclusion_strategy == "feature_selection":
            sklearn.set_config(enable_metadata_routing=False)
        else:
            sklearn.set_config(enable_metadata_routing=True)

    def get_features(self):
        """
        This method loads the machine learning features (traits) according to the specifications in the config.
        It gets the specific name of the file that contains the data, connect it to the feature path for the
        current analysis and sets the loaded features as a class attribute "X".
        """
        dataset_name = self.get_dataset_name(folder_att="feature_folder")
        spec_feature_path = os.path.join(self.feature_folder, dataset_name).replace(
            "\\", "/"
        )
        X = pd.read_pickle(spec_feature_path)
        print(f"loaded features from {spec_feature_path}")
        setattr(self, "X", X)

    def get_criterion(self):
        """
        This method loads the criterion (reactivities, either EB estimates of random slopes or OLS slopes)
        according to the specifications in the config.
        It gets the specific name of the file that contains the data, connect it to the feature path for the
        current analysis and sets the loaded features as a class attribute "y".
        """
        dataset_name = self.get_dataset_name(folder_att="crit_folder")
        spec_criterion_path = os.path.join(self.crit_folder, dataset_name).replace(
            "\\", "/"
        )
        y = pd.read_pickle(spec_criterion_path)
        assert len(self.X) == len(
            y
        ), f"Features and criterion differ in length, len(X) == {len(self.X)}, len(y) == {len(y)}"
        print(f"loaded criterion from {spec_criterion_path}")
        setattr(self, "y", y)

    def get_sample_weights(self):
        """Loads the sample weights we use when suppl == weighting_by_rel"""
        """
        This method loads the sample_weights (only if suppl_type == "weighting_by_rel") according to the 
        specifications in the config. It gets the specific name of the file that contains the data, connect 
        it to the feature path for the current analysis and sets the loaded features as a class attribute 
        "sample_weights".
        """
        if self.suppl_type == "weighting_by_rel":
            dataset_name = self.get_dataset_name(folder_att="weights_folder")
            spec_weights_path = os.path.join(self.weights_folder, dataset_name).replace(
                "\\", "/"
            )
            sample_weights = pd.read_pickle(spec_weights_path)
            assert len(self.X) == len(
                sample_weights
            ), "sample weights and Features differ in length"
            setattr(self, "sample_weights", sample_weights)

    def get_dataset_name(self, folder_att):
        """
        This function gets the name of the specific dataset based on the sample name and the soc_int_var in ssc.
        It is used in the functions that load the data relevant for machine learning analysis.

        Args:
            folder_att: str, is used to determine the right class attribute for getting the right path to
                the folder of interest for either loading features, criteria, or sample_weights.
                Therefore, it should be in ['feature_folder', 'weights_folder', 'crit_folder']
        """
        if self.study_name == "ssc":
            sample_soc_int_var = "_".join([self.raw_sample_name, self.soc_int_var])
            dataset_name = [
                str(file)
                for file in os.listdir(getattr(self, folder_att))
                if file.startswith(sample_soc_int_var)
            ][0]
        elif self.study_name == "mse":
            dataset_name = [
                str(file)
                for file in os.listdir(getattr(self, folder_att))
                if file.startswith(self.raw_sample_name)
            ][0]
        else:
            raise ValueError("Study not implemented")
        return dataset_name

    def select_features(self):
        """This method selects a subset of features, implemented in the linear baseline model subclass."""
        pass

    def fit(self, X, y):
        """Scikit-Learns "Fit" method of the machine learning model, model dependent, implemented in the subclasses."""
        self.model.fit(X, y)

    def predict(self, X):
        """Scikit-Learns "Predict" method of the machine learning model, model dependent, implemented in the subclasses."""
        return self.model.predict(X)

    def create_pipeline(self):
        """
        This function creates a pipeline with preprocessing steps (e.g., scaling and feature selection) and
        the estimator used in the repeated nested CV procedure. It sets the pipeline as a class attribute.
        """
        preprocessor = self.get_custom_scaler(data=self.X)

        if self.feature_inclusion_strategy == "feature_selection":
            selector = self.get_feature_selector()
            pipe = Pipeline(
                [
                    ("preprocess", preprocessor),
                    ("feature_selection", selector),
                    (
                        "model",
                        self.model,
                    ),  # sample_weights not implemented for RFECV
                ]
            )
        else:
            pipe = Pipeline(
                [
                    ("preprocess", preprocessor),
                    (
                        "model",
                        self.model.set_fit_request(
                            sample_weight=self.use_sample_weights
                        ),
                    ),
                ]
            )
        setattr(self, "pipeline", pipe)

    def get_custom_scaler(self, data):
        """
        This function creates a custom scaler that scales only continuous columns. Because the
        data.columns.difference() method unexpectedly ordered the features by alphabet, we
        preserved the new order of the columns for reassigning feature importances after the
        prediction procedure.

        Args:
            data: df, containing the features for a given analysis setting (binary and continuous)

        Returns:
            preprocessor: CustomScaler object that scales only continuous columns. Binary columns are still 0/1.

        """
        binary_cols = data.columns[(data.isin([0, 1])).all(axis=0)]  # ordered as in df
        continuous_cols = data.columns.difference(binary_cols)  # ordered by alphabet
        preprocessor = CustomScaler(cols_to_scale=continuous_cols)
        # set current column order as class attribute for assigning the right coefficients and shap values
        current_order = continuous_cols.tolist() + binary_cols.tolist()
        setattr(self, "current_feature_col_order", current_order)
        return preprocessor

    def get_feature_selector(self):
        """
        This function returns a RFECV (Recursive Feature Elimination with Cross-Validation) object
        for the pipeline if feature_inclusion_strategy == feature_selection. Specifications of the RFECV
        are all defined in the config.

        Returns:
            selector: RFECV object with the specified configurations
        """
        rfe_num_inner_cv = self.config["analysis"]["feature_selection"]["num_cv"]
        rfe_scoring = get_scorer(
            self.config["analysis"]["feature_selection"]["scoring_metric"]["name"]
        )
        rfe_stepsize = self.config["analysis"]["feature_selection"]["stepsize"]
        rfe_min_n_features = self.config["analysis"]["feature_selection"][
            "min_features_to_select"
        ]
        if self.config["analysis"]["feature_selection"]["estimator"] == "decision_tree":
            decision_tree = DecisionTreeRegressor(
                random_state=self.config["analysis"]["random_state"]
            )
            if isinstance(rfe_num_inner_cv, int):
                rfe_inner_cv = KFold(
                    n_splits=rfe_num_inner_cv,
                    shuffle=True,
                    random_state=self.config["analysis"]["random_state"] * 2,
                )
                # random state should differ from the inner cv spliting procedure
            else:
                raise TypeError("rfe_num_inner_cv must be an integer")
            selector = RFECV(
                estimator=decision_tree,
                step=rfe_stepsize,
                cv=rfe_inner_cv,
                scoring=rfe_scoring,
                min_features_to_select=rfe_min_n_features,
                verbose=0,
                n_jobs=self.config["analysis"]["parallelize"]["rfe_n_jobs"],
            )
        else:
            raise ValueError(
                "Only DecisionTree is currently implemented as RFE estimator."
            )
        return selector

    def nested_cross_val(
        self,
        rep=None,  # do not remove, used in single_nested_cv
        X=None,
        y=None,
        fix_rs=None,
        dynamic_rs=None,
        use_sample_weights=False,
        sample_weights=None,
    ):
        """
        This method represents the fundamental aspect of the BassMLAnalyzer class.
        This function performs nested cross-validation for a given partition of the total data in a train and test set.
        In the current analysis, this is only used in combination with the "repeated_nested_cross_val" method, repeating
        the CV procedure with 10 different data partitions (as defined by the dynamic random state that is passed
        to this method).
        In all specifications, this method
            - iterates over the CV splits
            - splits X and y in train and test data accordingly
            - performs GridSearchCV on train data (which repeatedly splits the train data into train/validation data)
            - evaluates the best performing models in GridSearchCV on the test data for multiple metrics
            - summarizes the model-specific SHAP calculations across outer folds
        Depending on the specification, this method might
            - enable metadata routing and aligns the sample_weights with the features ("weighting_by_rel")
            - implement a sanity check if the feature selection (RFECV) worked as expected
            - contain SHAP interaction values (only in certain analyses and if model == rfr), empty Dicts otherwise

        Args:
            rep: int, representing the number of repetitions of the nested CV procedure, for 10x between 1 and 10
            X: df, features for the machine learning analysis according to the current specification
            y: pd.Series, criteria for the machine learning analysis according to the current specification
            fix_rs: int, random_state parameter, which should remain the same for all analyses
            dynamic_rs: int, dynamic random_state parameter that changes across repetitions (so that the splits into
                train and test set differ between different repetitions)
            use_sample_weights: bool, true or false, indicates if sample_weights are used in the prediction process
            sample_weights: pd.Series or None, containing the sample weights in "weighting_by_rel"

        Returns:
            nested_scores_rep: Dict that collects the metrics evaluated on the test sets as keys and the
                values obtained in each test set as a list of values.
                Example: {"r2":[0.05, 0.02, 0.03], "spearman": [.20, .15, .25]}
            np.mean(nested_score): Mean of the scores in the outer test sets for the current data partition. Only
                displayed for the metric that was used to evaluated GridSearchCV (currently R²)
            np.std(nested_score): SD of the scores in the outer test sets for the current data partition. Only
                displayed for the metric that was used to evaluated GridSearchCV (currently R²)
            ml_models: list of the best models and their hyperparameter configuration obtained in the GridSearchCV
                (grid_search.best_estimator_.named_steps["model"])
            shap_values_train: ndarray, SHAP values obtained for the train set, summarized across outer folds.
                Of shape (n_samples x n_features).
            shap_values_test: ndarray, SHAP values obtained for the test set, summarized across outer folds.
                Of shape (n_samples x n_features).
            shap_ia_values_train: [None, ndarray], SHAP interaction values obtained for the train set, summarized
                across outer folds. Of shape (n_samples x n_features x n_features). Only if model == rfr and
                calc_ia_values is enabled in the config, None otherwise.
            shap_ia_values_test: [None, ndarray], SHAP interaction values obtained for the test set, summarized
                across outer folds. Of shape (n_samples x n_features x n_features). Only if model == rfr and
                calc_ia_values is enabled in the config, None otherwise.
        """
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        if use_sample_weights and sample_weights is None:
            sample_weights = self.sample_weights
        estimator = self.pipeline

        X, y = X.align(
            y, axis=0, join="inner"
        )  # Note: join='outer' resorts keys, join='inner' preserves the order
        inner_cv = KFold(n_splits=self.num_cv, random_state=fix_rs, shuffle=True)
        outer_cv = KFold(
            n_splits=self.num_cv, random_state=dynamic_rs, shuffle=True
        )  # dynamic_rs: one random_state per repetition

        nested_scores_rep = dict()
        ml_pipelines = []
        ml_models = []
        # Joblib somehow resets this -> therefore it has to repeated directly before ml analysis
        current = get_config()["enable_metadata_routing"]
        print("metadata_routing before adjusting again", current)
        self.adjust_metadata_routing()
        current = get_config()["enable_metadata_routing"]
        print("metadata_routing after adjusting again", current)

        for train_index, test_index in outer_cv.split(X, y):
            # Convert numerical indices to string indices and select data based on these indices
            train_indices = X.index[train_index]
            test_indices = X.index[test_index]
            X_train, X_test = X.loc[train_indices], X.loc[test_indices]
            y_train, y_test = y.loc[train_indices], y.loc[test_indices]
            assert (
                X_train.index == y_train.index
            ).all(), "Indices between train data differ"
            assert (
                X_test.index == y_test.index
            ).all(), "Indices between test data differ"

            if use_sample_weights:
                sample_weights_train, sample_weights_test = (
                    sample_weights.loc[train_indices],
                    sample_weights.loc[test_indices],
                )
                assert (
                    X_train.index == sample_weights_train.index
                ).all(), "Indices between train data differ"
                assert (
                    X_test.index == sample_weights_test.index
                ).all(), "Indices between train data differ"
                estimator.set_score_request(sample_weight=use_sample_weights)

            if (
                self.config["analysis"]["scoring_metric"]["inner_cv_loop"]["name"]
                == "spearman_corr"
            ):
                raise ValueError(
                    "inner cv scoring for spearman_corr not implemented yet."
                )
            else:
                scoring_inner_cv = get_scorer(
                    self.config["analysis"]["scoring_metric"]["inner_cv_loop"]["name"]
                )
                if use_sample_weights:
                    scoring_inner_cv.set_score_request(sample_weight=use_sample_weights)

            # print(self.config["analysis"]["parallelize"]["inner_cv_n_jobs"])
            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=self.hyperparameter_grid,
                cv=inner_cv,
                scoring=scoring_inner_cv,
                refit=True,
                verbose=1,
                n_jobs=self.config["analysis"]["parallelize"]["inner_cv_n_jobs"],
            )
            # Catch ConvergenceWarnings, e.g., for SVR, to prevent messing up the output logs
            if use_sample_weights:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    grid_search.fit(
                        X_train, y_train, sample_weight=sample_weights_train
                    )
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    grid_search.fit(X_train, y_train)

            # Get the best model / pipeline from GridSearchCV
            ml_models.append(grid_search.best_estimator_.named_steps["model"])
            ml_pipelines.append(grid_search.best_estimator_)

            # Sanity check for feature selection
            if self.feature_inclusion_strategy == "feature_selection":
                n_before_selection = len(X_train)
                n_selected_features = grid_search.best_estimator_.named_steps[
                    "feature_selection"
                ].n_features_
                print(
                    "Number of features before feature selection:", n_before_selection
                )
                print("Number of selected features:", n_selected_features)

            # Evaluate multiple metrics on the outer folds
            if use_sample_weights:
                scoring_functions = {
                    "r2": get_scorer("r2").set_score_request(
                        sample_weight=use_sample_weights
                    ),
                    "rmse": make_scorer(
                        mean_squared_error, greater_is_better=False
                    ).set_score_request(sample_weight=use_sample_weights),
                    "spearman": make_scorer(self.spearman_corr).set_score_request(
                        sample_weight=use_sample_weights
                    ),  # Assuming spearman_scorer is already defined
                }
            else:
                scoring_functions = {
                    "r2": get_scorer("r2"),
                    "rmse": make_scorer(
                        mean_squared_error, greater_is_better=True
                    ),  # this should have been "root_mean_squared_error", is corrected in the results analysis
                    "spearman": make_scorer(
                        self.spearman_corr
                    ),  # Assuming spearman_scorer is already defined
                }
            scorers = {
                metric: scoring_functions[metric]
                for metric in self.config["analysis"]["scoring_metric"]["outer_cv_loop"]
            }

            # Evaluate the best model on the outer folds test set and append the results to the score Dict
            scores = {
                metric: scorer(grid_search.best_estimator_, X_test, y_test)
                for metric, scorer in scorers.items()
            }
            for metric, score in scores.items():
                nested_scores_rep.setdefault(metric, []).append(score)

        (
            shap_values_train,
            shap_values_test,
            shap_ia_values_train,
            shap_ia_values_test,
        ) = self.summarize_shap_values_outer_cv(X, y, ml_pipelines, outer_cv)

        # Get score for GridSearchCV evaluation metric (i.e., R²)
        nested_score = nested_scores_rep[
            self.config["analysis"]["scoring_metric"]["inner_cv_loop"]["name"]
        ]
        return (
            nested_scores_rep,
            np.mean(nested_score),
            np.std(nested_score),
            ml_models,
            shap_values_train,
            shap_values_test,
            shap_ia_values_train,
            shap_ia_values_test,
        )

    def compute_shap_for_fold(
        self,
        num_cv_,
        pipeline,
        index_mapping,
        num_train_indices,
        num_test_indices,
        X,
        all_features,
    ):
        """
        Parallelization implementation of the method "summarize_shap_values_outer_cv". This enables parallel SHAP
        calculations for outer_folds if specified in the config.

        Args:
            num_cv_: int, indicating a certain run of the outer CV loop
            pipeline: Pipeline object, the best performing pipeline in the GridSearchCV associated with num_cv_
            index_mapping: Mapping from numerical indices to variable indices for unambiguous feature assignment
            num_train_indices: Numeric indices of the features that are in the train set in the outer loop "num_cv_"
            num_test_indices: Numeric indices of the features that are in the test set in the outer loop "num_cv_"
            X: df, features for the machine learning-based prediction
            all_features: Index object (X.columns), representing the feature names

        Returns:
            train_shap_template: ndarray, containing the SHAP values of the train set for the outer fold "num_cv_"
                Shape: (n_features x n_samples), array values that represent samples that were in the test set
                in the current outer fold are all zero.
            test_shap_template: ndarray, containing the SHAP values of the test set for the outer fold "num_cv_"
                Shape: (n_features x n_samples), array values that represent samples that were in the train set
                in the current outer fold are all zero.
            num_train_indices[num_cv_]: Numerical indices for the samples in the train set for the current outer fold
            num_test_indices[num_cv_]: Numerical indices for the samples in the test set for the current outer fold
            train_ia_shap_template: [None, ndarray], containing the SHAP interaction values of the train set for
                the current outer fold "num_cv_". Shape: (n_features x n_features x n_samples), array values that
                represent samples that were in the test set in the current outer fold are all zero.
                Only defined if calc_ia_values is specific in the config and model == 'rfr', None otherwise
            test_ia_shap_template: [None, ndarray], containing the SHAP interaction values of the test set for
                the current outer fold "num_cv_". Shape: (n_features x n_features x n_samples), array values that
                represent samples that were in the train set in the current outer fold are all zero.
                Only defined if calc_ia_values is specific in the config and model == 'rfr', None otherwise
        """
        str_train_indices = [index_mapping[idx] for idx in num_train_indices[num_cv_]]
        str_test_indices = [index_mapping[idx] for idx in num_test_indices[num_cv_]]
        X_train = X.loc[str_train_indices]
        X_test = X.loc[str_test_indices]

        (
            shap_values_train,
            selected_features_train,
            shap_ia_values_train,
        ) = self.calculate_shap_values(X_train, pipeline)
        (
            shap_values_test,
            selected_features_test,
            shap_ia_values_test,
        ) = self.calculate_shap_values(X_test, pipeline)
        # Prepare templates for this fold
        train_shap_template = np.zeros((len(str_train_indices), X.shape[1]))
        test_shap_template = np.zeros((len(str_test_indices), X.shape[1]))

        # Update the templates with actual SHAP values for the selected features
        for i, feature in enumerate(selected_features_train):
            col_index = all_features.get_loc(feature)
            train_shap_template[:, col_index] = shap_values_train[:, i]
        for i, feature in enumerate(selected_features_test):
            col_index = all_features.get_loc(feature)
            test_shap_template[:, col_index] = shap_values_test[:, i]

        if (
            self.model_name == "randomforestregressor"
            and self.config["analysis"]["calc_ia_values"]
        ):
            train_ia_shap_template = np.zeros(
                (len(str_train_indices), X.shape[1], X.shape[1])
            )
            test_ia_shap_template = np.zeros(
                (len(str_test_indices), X.shape[1], X.shape[1])
            )
            # Update the templates with actual SHAP interaction values for the selected features
            for i, feature_i in enumerate(selected_features_train):
                for j, feature_j in enumerate(selected_features_train):
                    col_index_i = all_features.get_loc(feature_i)
                    col_index_j = all_features.get_loc(feature_j)
                    train_ia_shap_template[
                        :, col_index_i, col_index_j
                    ] = shap_ia_values_train[:, i, j]
            for i, feature_i in enumerate(selected_features_test):
                for j, feature_j in enumerate(selected_features_test):
                    col_index_i = all_features.get_loc(feature_i)
                    col_index_j = all_features.get_loc(feature_j)
                    test_ia_shap_template[
                        :, col_index_i, col_index_j
                    ] = shap_ia_values_test[:, i, j]
        else:
            train_ia_shap_template, test_ia_shap_template = None, None

        return (
            train_shap_template,
            test_shap_template,
            num_train_indices[num_cv_],
            num_test_indices[num_cv_],
            train_ia_shap_template,
            test_ia_shap_template,
        )

    def summarize_shap_values_outer_cv(self, X, y, pipelines, outer_cv):
        """
        Ths function summarizes the shap values of the repeated nested resampling scheme as described by
        Scheda & Diciotti (2022). Thus, it averages the individual shap values over the train sets in the
        outer CV and takes the individual shap values of the test sets (when doing a 10x10 CV, a sample is 9 times
        in the train set and only 1 time in the test set, so we have 9 individual training set shap values
        and only 1 test set shap value per individual sample).
        It does this by defining a template a zeros of the shap of the SHAP values (n_features x n_samples) and
        accumulates the SHAP values computed in the outer_folds in this template.
        Note: The paper did not describe a procedure for using feature selection. I have adapted this to
        feature selection by accumulating the shap values across folds setting shap values for non-selected
        features to zero.

        Args:
            X: df, features for the machine learning analysis according to the current specification
            y: pd.Series, criteria for the machine learning analysis according to the current specification
            pipelines: List of Pipeline objects, containing preprocessing steps and estimators that obtained the
                best performance in the GridSearchCV in "nested_cv"
            outer_cv: KFold object used in "nested_cv" to repeatedly split the data into train and test set

        Returns:
            avg_shap_values_train: ndarray, containing the mean SHAP values of the train set (n_features x n_samples)
            test_shap_values: ndarray, containing the SHAP values of the test set (n_features x n_samples)
            avg_ia_shap_values_train: [None, ndarray], containing the mean SHAP interaction values of the train set
                (n_features x n_features x n_samples) if calc_ia_values is specific in the config and model == 'rfr',
                None otherwise
            ia_test_shap_values: [None, ndarray], containing the SHAP interaction values of the test set
                (n_features x n_features x n_samples) if calc_ia_values is specific in the config and model == 'rfr',
                None otherwise
        """
        print('---------------------')
        print('Calculate SHAP values')
        # Create a mapping from numerical indices to variable indices
        index_mapping = dict(enumerate(X.index))
        # Get numerical indices of the samples in the outer cv
        num_train_indices, num_test_indices = zip(
            *[(train, test) for train, test in outer_cv.split(X, y)]
        )
        all_features = X.columns
        # Parallelization of shap value calculations, if specified in config
        parallel_results = Parallel(
            n_jobs=self.config["analysis"]["parallelize"]["shap_outer_cv_n_jobs"],
            verbose=1,
            backend="loky",
        )(
            delayed(self.compute_shap_for_fold)(
                num_cv_=num_cv_,
                pipeline=pipeline,
                index_mapping=index_mapping,
                num_train_indices=num_train_indices,
                num_test_indices=num_test_indices,
                X=X,
                all_features=all_features,
            )
            for num_cv_, pipeline in enumerate(pipelines)
        )

        train_shap_accumulator = np.zeros((X.shape[0], X.shape[1]))
        test_shap_values = np.zeros((X.shape[0], X.shape[1]))
        # Aggregate results from all folds
        for (
            train_shap_template,
            test_shap_template,
            train_idx,
            test_idx,
            _,
            _,
        ) in parallel_results:
            train_shap_accumulator[train_idx, :] += train_shap_template
            test_shap_values[test_idx, :] += test_shap_template
        # Compute average SHAP values for training set
        avg_shap_values_train = train_shap_accumulator / (self.num_cv - 1)

        if (
            self.model_name == "randomforestregressor"
            and self.config["analysis"]["calc_ia_values"]
        ):
            ia_train_shap_accumulator = np.zeros((X.shape[0], X.shape[1], X.shape[1]))
            ia_test_shap_values = np.zeros((X.shape[0], X.shape[1], X.shape[1]))
            # Aggregate results from all folds
            for (
                _,
                _,
                train_idx,
                test_idx,
                train_ia_shap_template,
                test_ia_shap_template,
            ) in parallel_results:
                ia_train_shap_accumulator[train_idx, :, :] += train_ia_shap_template
                ia_test_shap_values[test_idx, :, :] += test_ia_shap_template
            avg_ia_shap_values_train = ia_train_shap_accumulator / (self.num_cv - 1)
        else:
            avg_ia_shap_values_train = None
            ia_test_shap_values = None

        return (
            avg_shap_values_train,
            test_shap_values,
            avg_ia_shap_values_train,
            ia_test_shap_values,
        )

    def single_nested_cv_rep(
        self, rep, X, y, fix_rs, dynamic_rs, use_sample_weights, sample_weights=None
    ):
        """
        Parallelized version of nested_cv. This enables the parallel execution of repetitions, which
        is really efficient (i.e., if 10 Cores are available, each Core will handle one of the 10 different
        data partitions). The downside is that nested parallelism is not so easy using e.g. Joblib.

        Args:
            rep: int, indicating the number of data partitioning, if 10xCV, rep is between 1 and 10
            X: df, features for the machine learning analysis according to the current specification
            y: pd.Series, criteria for the machine learning analysis according to the current specification
            fix_rs: int, random_state parameter, which should remain the same for all analyses
            dynamic_rs: int, dynamic random_state parameter that changes across repetitions (so that the splits into
                train and test set differ between different repetitions)
            use_sample_weights: bool, true or false, indicates if sample_weights are used in the prediction process
            sample_weights: pd.Series or None, containing the sample weights in "weighting_by_rel"

        Returns:
            nested_scores_rep: Dict that collects the metrics evaluated on the test sets as keys and the
                values obtained in each test set as a list of values.
                Example: {"r2":[0.05, 0.02, 0.03], "spearman": [.20, .15, .25]}
            rep: int, indicating the number of data partitioning, if 10xCV, rep is between 1 and 10
            mean_score: Mean of the scores in the outer test sets for the current data partition. Only
                displayed for the metric that was used to evaluated GridSearchCV (currently R²)
            best_models: list of the best models and their hyperparameter configuration obtained in the GridSearchCV
                (grid_search.best_estimator_.named_steps["model"])
            shap_values_train: ndarray, SHAP values obtained for the train set, summarized across outer folds.
                Of shape (n_samples x n_features).
            shap_values_test: ndarray, SHAP values obtained for the test set, summarized across outer folds.
                Of shape (n_samples x n_features).
            shap_ia_values_train: [None, ndarray], SHAP interaction values obtained for the train set, summarized
                across outer folds. Of shape (n_samples x n_features x n_features). Only if model == rfr and
                calc_ia_values is enabled in the config, None otherwise.
            shap_ia_values_test: [None, ndarray], SHAP interaction values obtained for the test set, summarized
                across outer folds. Of shape (n_samples x n_features x n_features). Only if model == rfr and
                calc_ia_values is enabled in the config, None otherwise.
        """
        print("---------------------------")
        print("nested cv repetition: ", rep)
        print("random_state: ", dynamic_rs)
        (
            nested_scores_rep,
            mean_score,
            std_score,
            best_models,
            shap_values_train,
            shap_values_test,
            shap_ia_values_train,
            shap_ia_values_test,
        ) = self.nested_cross_val(
            rep=rep,
            X=X,
            y=y,
            fix_rs=fix_rs,
            dynamic_rs=dynamic_rs,
            use_sample_weights=use_sample_weights,
            sample_weights=sample_weights,
        )
        return (
            nested_scores_rep,
            rep,
            mean_score,
            best_models,
            shap_values_train,
            shap_values_test,
            shap_ia_values_train,
            shap_ia_values_test,
        )

    @time_function_execution
    def repeated_nested_cv(self):
        """
        This function performs the nested cross-validation repeatedly using different data partitions into train
        and test sets. It does so by repeatedly executing the nested cross validation function using different
        dynamic random states that control the train-test partitioning. It also updates the class attributes
        with the corresponding analysis results.
        It further summarizes the SHAP values across repetitions and the SHAP interaction values (if specified)
        into meaningful aggregates (both steps are done on the cluster to save resources and memory).
        """
        X = self.X.copy()
        y = self.y.copy()
        sample_weights = None
        use_sample_weights = self.use_sample_weights
        if use_sample_weights:
            sample_weights = self.sample_weights.copy()
        fix_random_state = self.config["analysis"]["random_state"]
        n_jobs_reps = self.config["analysis"]["parallelize"]["reps_n_jobs"]
        nested_scores_total = list()
        current = get_config()["enable_metadata_routing"]
        print("metadata_routing", current)

        # Execute the nested cross-validation procedure in parralel
        results = Parallel(n_jobs=n_jobs_reps, verbose=10, backend="loky")(
            delayed(self.single_nested_cv_rep)(
                rep,
                X,
                y,
                fix_random_state,
                fix_random_state + rep,
                use_sample_weights,
                sample_weights,
            )
            for rep in range(1, self.num_repeats + 1)
        )
        for (
            nested_scores_rep,
            rep,
            mean_score,
            best_models,
            shap_values_train,
            shap_values_test,
            shap_ia_values_train,
            shap_ia_values_test,
        ) in results:
            self.best_models[f"rep_{rep}"] = best_models
            self.shap_values["train"][f"rep_{rep}"] = shap_values_train
            self.shap_values["test"][f"rep_{rep}"] = shap_values_test
            self.repeated_nested_scores[f"rep_{rep}"] = nested_scores_rep
            nested_scores_total.append(mean_score)

        # Aggregate SHAP values /  SHAP interaction values
        self.summarize_shap_values_repetitions()
        if (
            self.model_name == "randomforestregressor"
            and self.config["analysis"]["calc_ia_values"]
        ):
            self.summarize_shap_interaction_values(
                shap_ia_values=shap_ia_values_train, columns=X.columns, dataset="train"
            )
            self.summarize_shap_interaction_values(
                shap_ia_values=shap_ia_values_test, columns=X.columns, dataset="test"
            )
        print('Mean R2 score per repetition:', nested_scores_total)
        print('Mean R2 score across repetitions:', np.mean(nested_scores_total))

    def summarize_shap_values_repetitions(self):
        """
        This function summarizes the individual shap values per sample across repetitions. It computes the
        mean and the SD of the SHAP values across repetitions.
        """
        # Stack the SHAP values into a 3D array for train and test datasets
        stacked_shap_train = np.array(
            [self.shap_values["train"][key] for key in self.shap_values["train"]]
        )
        stacked_shap_test = np.array(
            [self.shap_values["test"][key] for key in self.shap_values["test"]]
        )
        # Compute the average across repetitions (axis 0) and std (based on the number of total outer folds)
        self.shap_values["train"]["avg_across_reps"] = np.mean(
            stacked_shap_train, axis=0
        )
        self.shap_values["train"]["std_across_reps"] = np.std(
            stacked_shap_train, axis=0
        )
        self.shap_values["test"]["avg_across_reps"] = np.mean(stacked_shap_test, axis=0)
        self.shap_values["test"]["std_across_reps"] = np.std(stacked_shap_test, axis=0)

    def summarize_shap_interaction_values(self, shap_ia_values, columns, dataset):
        """
        This function is used for summarizing the raw interactions values of shape (n_samples, n_features, n_features).
        1) It summarizes the raw and the absolute shap values across samples
        2) It identifies the most influential interaction per person and stores a) the variables and
           b) the pairs that interacts most in two dictionaries
        3) It summarizes the interaction values across features and samples, so that for each feature there
           is a score reflecting a "global" interaction value
        All results are stored as key:value pairs in the attribute self.shap_ia_values

        Args:
            shap_ia_values: ndarray, containing the IA values, Shap (n_samples x n_features xn_features)
            columns: pd.Index, Features corresponding to n_features in shap_ia_values
            dataset: str, "train" or "test"
        """
        # Rescale all shap_ia_values -> + 1 Mio -> is reverted in the ShapValueAnalyzer
        shap_ia_values = shap_ia_values * 1000000

        # 1) Summarize across persons
        ia_across_persons = shap_ia_values.mean(0)
        ia_across_persons_df = pd.DataFrame(
            ia_across_persons, index=columns, columns=columns
        )
        ia_across_persons_dct = {
            col: {
                row: value for row, value in ia_across_persons_df[col].items()
            }  # keep the main effects
            for col in ia_across_persons_df.columns
        }
        abs_ia_across_persons = np.abs(shap_ia_values).mean(0)
        abs_ia_across_persons_df = pd.DataFrame(
            abs_ia_across_persons, index=columns, columns=columns
        )
        abs_ia_across_persons_dct = {
            col: {
                row: value for row, value in abs_ia_across_persons_df[col].items()
            }  # keep the main effects
            for col in abs_ia_across_persons_df.columns
        }

        # 2) Most influential interactions
        mask = np.eye(len(columns), dtype=bool)
        most_influential_ia_per_person = np.array(
            [
                np.unravel_index(
                    np.nanargmax(
                        np.where(mask, np.nan, np.abs(shap_ia_values[i, :, :]))
                    ),
                    (len(columns), len(columns)),
                )
                for i in range(shap_ia_values.shape[0])
            ]
        )
        most_influential_interaction_df = pd.DataFrame(
            [
                (columns[row], columns[col])
                for row, col in most_influential_ia_per_person
            ],
            columns=["Feature 1", "Feature 2"],
        )
        feature_counts = most_influential_interaction_df.apply(
            pd.Series.value_counts
        ).sum(axis=1)
        feature_count_dict = feature_counts.sort_values(ascending=False).to_dict()
        unordered_pairs = most_influential_interaction_df.apply(
            lambda row: tuple(sorted([row["Feature 1"], row["Feature 2"]])), axis=1
        )
        unordered_pairs_as_strings = [
            str(pair) for pair in unordered_pairs
        ]  # Convert tuples to strings
        pair_counts_dct = pd.Series(unordered_pairs_as_strings).value_counts().to_dict()

        # 3) Summarize across persons and features
        agg_ia_across_persons_ias = np.nanmean(
            np.where(mask, np.nan, shap_ia_values), axis=(0, 1)
        )
        agg_ia_across_persons_ias_df = pd.DataFrame(
            agg_ia_across_persons_ias, index=columns, columns=["agg_ia_val"]
        )
        agg_ia_across_persons_ias_dct = agg_ia_across_persons_ias_df[
            "agg_ia_val"
        ].to_dict()
        abs_agg_ia_across_persons_ias = np.nanmean(
            np.abs(np.where(mask, np.nan, shap_ia_values)), axis=(0, 1)
        )
        abs_agg_ia_across_persons_ias_df = pd.DataFrame(
            abs_agg_ia_across_persons_ias, index=columns, columns=["agg_ia_val"]
        )
        abs_agg_ia_across_persons_ias_dct = abs_agg_ia_across_persons_ias_df[
            "agg_ia_val"
        ].to_dict()

        # Set the aggregates in the class attribute Dict
        self.shap_ia_values[dataset]["agg_ia_persons"] = ia_across_persons_dct
        self.shap_ia_values[dataset]["abs_agg_ia_persons"] = abs_ia_across_persons_dct
        self.shap_ia_values[dataset][
            "most_influential_ia_features"
        ] = feature_count_dict
        self.shap_ia_values[dataset]["most_influential_ia_pairs"] = pair_counts_dct
        self.shap_ia_values[dataset][
            "agg_ia_persons_ias"
        ] = agg_ia_across_persons_ias_dct
        self.shap_ia_values[dataset][
            "abs_agg_ia_persons_ias"
        ] = abs_agg_ia_across_persons_ias_dct

    def store_analysis_results(self):
        """This function stores the prediction results and the SHAP values / Shap interaction values as .JSON files."""
        results_filename = os.path.join(
            self.output_dir, self.config["analysis"]["output_filenames"]["performance"]
        )
        os.makedirs(os.path.dirname(results_filename), exist_ok=True)
        cv_results = self.repeated_nested_scores
        for rep, val in cv_results.items():
            for metric in val:
                cv_results[rep][metric] = cv_results[rep][metric]  # .tolist()
        with open(results_filename, "w") as file:
            json.dump(cv_results, file, indent=4)

        shap_values_filename = os.path.join(
            self.output_dir, self.config["analysis"]["output_filenames"]["shap_values"]
        )
        os.makedirs(os.path.dirname(shap_values_filename), exist_ok=True)
        feature_names = self.X.columns.tolist()
        shap_values = self.shap_values.copy()
        for train_test_set in list(shap_values.keys()):
            for key in list(shap_values[train_test_set].keys()):
                # Remove the single rep_ shap values for saving storage, only mean and sd across reps are stored
                if "rep_" in key:
                    del shap_values[train_test_set][key]
                    continue
                shap_values[train_test_set][key] = [
                    arr.tolist() for arr in shap_values[train_test_set][key]
                ]
        shap_values["feature_names"] = feature_names
        with open(shap_values_filename, "w") as file:
            json.dump(shap_values, file, indent=4)

        # store shap interaction values
        shap_ia_values_filename = os.path.join(
            self.output_dir,
            self.config["analysis"]["output_filenames"]["shap_ia_values"],
        )
        os.makedirs(os.path.dirname(shap_ia_values_filename), exist_ok=True)
        shap_ia_values = self.shap_ia_values.copy()
        for train_test_set in list(shap_ia_values.keys()):
            if shap_values[train_test_set]:
                shap_ia_values["feature_names"] = feature_names
                with open(shap_ia_values_filename, "w") as file:
                    json.dump(shap_ia_values, file, indent=4)

    def get_average_coefficients(self):
        """Gets the coefficients of linear models of the predictions, implemented in the linear model subclass."""
        pass

    def store_coefficients(self):
        """Stores the linear coefficients, implemented in the linear model subclass."""
        pass

    @abstractmethod
    def calculate_shap_values(self, X, best_model):
        """Calculation of individual SHAP values, model-dependent, implemented in the subclasses."""
        pass

    @abstractmethod
    def calculate_shap_for_instance(self, n_instance, instance, explainer):
        """Calculates shap values for a single instance for parallelization, implemented in the subclasses."""
        pass

    @staticmethod
    def spearman_corr(y_true_raw, y_pred_raw, sample_weight=None):
        """
        For using it as a scorer in the nested cv scheme, we need to manually calculate spearmans rank correlation.
        Note: Implementing sample_weights in the rank correlation is not trivial. Therefore, we approximate
        the weighting by replicating the samples according to their weights.

        Args:
            y_true_raw: pd.Series, the "true" reactivities that we predict in the machine learning-based prediction procedure
            y_pred_raw: pd.Series, the predicted reactivities that were outputted by the model
            sample_weight: [None, pd.Series], used for weighting by reliability (if suppl_type == weighting_by_rel)

        Returns:
            rank_corr: spearmans rank correlation (rho) for given y_true and y_pred values
        """
        if sample_weight is not None:
            # Normalize sample weights
            sample_weight = np.array(sample_weight)
            sample_weight = sample_weight / sample_weight.sum()
            # Replicate samples in proportion to their weights
            y_true = np.repeat(
                y_true_raw, np.ceil(sample_weight * len(sample_weight)).astype(int)
            )
            y_pred = np.repeat(
                y_pred_raw, np.ceil(sample_weight * len(sample_weight)).astype(int)
            )
        else:
            y_true = y_true_raw
            y_pred = y_pred_raw
        rank_corr, _ = spearmanr(y_true, y_pred)
        return rank_corr
