import os

import numpy as np
import shap
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor

from src.analysis.machine_learning.BaseMLAnalyzer import BaseMLAnalyzer


class RFRAnalyzer(BaseMLAnalyzer):
    """
    This class is the specific implementation of the random forest regression using the standard Sklearn implementation
    (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html). Inherits from
    BaseMLAnalyzer. For class attributes, see BaseMLAnalyzer. Hyperparameters to tune are defined in the config.
    """

    def __init__(self, config, output_dir):
        """
        Constructor method of the RFRAnalyzer class.

        Args:
            config: YAML config determining specifics of the analysis
            output_dir: Specific directory where the results are stored
        """
        super().__init__(config, output_dir)
        self.model = RandomForestRegressor(
            random_state=self.config["analysis"]["random_state"]
        )

    def calculate_shap_for_instance(self, n_instance, instance, explainer):
        """Calculates tree-based SHAP values for a single instance for parallelization.

        Args:
            n_instance: Number of a certain individual to calculate SHAP values for
            instance: 1d-array, represents the feature values for a single individual
            explainer: shap.TreeExplainer

        Returns:
            explainer(instance.reshape(1, -1)).values: array containing the SHAP values for "n_instance"
        """
        return explainer(instance.reshape(1, -1), check_additivity=False).values

    def compute_shap_interaction_values(self, explainer, X_subset):
        """
        This function computes (pairwise) SHAP interaction values for the rfr

        Args:
            explainer: shap.TreeExplainer
            X_subset: df, subset of X for which interaction values are computed, can be parallelized

        Returns:
            explainer.shap_interaction_values(X_subset): array containing SHAP interaction values for X_subset
        """
        print(
            "Currently processing these indices of the original df:",
            X_subset.index[:10],
        )
        print(
            f"Calculating SHAP for subset of length {len(X_subset)} in process {os.getpid()}"
        )
        return explainer.shap_interaction_values(X_subset)

    def calculate_shap_values(self, X, pipeline):
        """
        This function calculates tree-based SHAP values for a given analysis setting. This includes applying the
        preprocessing steps that were applied in the pipeline (e.g., scaling, RFECV if specified).
        It calculates the SHAP values using the explainers.TreeExplainer, the SHAP implementation that is
        suitable for tree-based models. SHAP calculations can be parallelized.
        Further, it calculates the SHAP interaction values based on the TreeExplainer, if specified

        Args:
            X: df, features for the machine learning analysis according to the current specification
            pipeline: Sklearn Pipeline object containing the steps of the ml-based prediction (i.e., preprocessing
                and estimation using the prediction model).

        Returns:
            shap_values_array: ndarray, obtained SHAP values, of shape (n_features x n_samples)
            columns: pd.Index, contains the names of the features in X associated with the SHAP values
            shap_interaction_values: SHAP interaction values, of shape (n_features x n_features x n_samples)
        """
        columns = X.columns
        # Note: columns = self.current_feature_col_order would fix the problem of wrong feature assignment right here
        # We implemented the assignment correction in the result processing
        X_processed = pipeline.named_steps["preprocess"].transform(X)
        if "feature_selection" in pipeline.named_steps:
            X_processed = pipeline.named_steps["feature_selection"].transform(
                X_processed
            )
            columns = columns[pipeline.named_steps["feature_selection"].get_support()]
        explainer_tree = shap.explainers.Tree(pipeline.named_steps["model"])
        if self.config["analysis"]["calc_ia_values"]:
            # Parallelize the calculations processing chunks of the data
            n_jobs = self.config["analysis"]["parallelize"]["shap_ia_values_n_jobs"]
            chunk_size = X.shape[0] // n_jobs + (X.shape[0] % n_jobs > 0)
            print("n_jobs shap ia _values")
            print("chunk_size:", chunk_size)
            results = Parallel(n_jobs=n_jobs, verbose=1, backend="multiprocessing")(
                delayed(self.compute_shap_interaction_values)(
                    explainer_tree, X[i : i + chunk_size]
                )
                for i in range(0, X.shape[0], chunk_size)
            )
            # Combine the results
            print("len results:", len(results))
            shap_ia_values_array = np.vstack(results)
        else:
            shap_ia_values_array = None

        print(self.config["analysis"]["parallelize"]["shap_n_jobs"])
        shap_values_list = Parallel(
            n_jobs=self.config["analysis"]["parallelize"]["shap_n_jobs"],
            verbose=0,
            backend="multiprocessing",
        )(
            delayed(self.calculate_shap_for_instance)(
                n_instance, instance, explainer_tree
            )
            for n_instance, instance in enumerate(X_processed)
        )
        # Convert list of arrays to single array
        shap_values_array = np.vstack(shap_values_list)
        return shap_values_array, columns, shap_ia_values_array
