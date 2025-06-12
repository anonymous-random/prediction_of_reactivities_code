import numpy as np
import shap
from joblib import Parallel, delayed
from sklearn.svm import SVR

from src.analysis.machine_learning.BaseMLAnalyzer import BaseMLAnalyzer


class SVRAnalyzer(BaseMLAnalyzer):
    """
    This class is the specific implementation of the support vector regression using the standard Sklearn implementation
    (https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). Inherits from BaseMLAnalyzer.
    For class attributes, see BaseMLAnalyzer. Hyperparameters to tune are defined in the config.
    """

    def __init__(self, config, output_dir):
        """
        Constructor method of the SVRAnalyzer class.

        Args:
            config: YAML config determining specifics of the analysis
            output_dir: Specific directory where the results are stored
        """
        super().__init__(config, output_dir)
        self.model = SVR()

    def calculate_shap_for_instance(self, n_instance, instance, explainer):
        """Calculates permutation SHAP values for a single instance for parallelization.

        Args:
            n_instance: Number of a certain individual to calculate SHAP values for
            instance: 1d-array, represents the feature values for a single individual
            explainer: SHAP.PermutationExplainer

        Returns:
            explainer(instance.reshape(1, -1)).values: array containing the SHAP values for "n_instance"
        """
        return explainer(instance.reshape(1, -1)).values

    def calculate_shap_values(self, X, pipeline):
        """
        This function calculates permutation SHAP values for a given analysis setting. This includes applying the
        preprocessing steps that were applied in the pipeline (e.g., scaling, RFECV if specified).
        It calculates the SHAP values using the explainers.Permutation, because is no model-specific
        SHAP implementation for SVR available. SHAP calculations can be parallelized.

        Args:
            X: df, features for the machine learning analysis according to the current specification
            pipeline: Sklearn Pipeline object containing the steps of the ml-based prediction (i.e., preprocessing
                and estimation using the prediction model).

        Returns:
            shap_values_array: ndarray, obtained SHAP values, of shape (n_features x n_samples)
            columns: pd.Index, contains the names of the features in X associated with the SHAP values
            shap_interaction_values: None, returned here for method consistency
        """
        columns = X.columns
        X_processed = pipeline.named_steps["preprocess"].transform(X)
        if "feature_selection" in pipeline.named_steps:
            X_processed = pipeline.named_steps["feature_selection"].transform(
                X_processed
            )
            columns = columns[pipeline.named_steps["feature_selection"].get_support()]

        # Define max_evals based on the number of features
        max_evals = 2 * len(X_processed[0, :]) + 1
        explainer_per = shap.explainers.Permutation(
            pipeline.named_steps["model"].predict,
            X_processed,
            max_evals=max_evals,
            npermutations=200,
        )
        shap_values_list = Parallel(
            n_jobs=self.config["analysis"]["parallelize"]["shap_n_jobs"],
            verbose=0,
            backend="loky",
        )(
            delayed(self.calculate_shap_for_instance)(
                n_instance, instance, explainer_per
            )
            for n_instance, instance in enumerate(X_processed)
        )
        # Convert list of arrays to single array
        shap_values_array = np.vstack(shap_values_list)
        shap_interaction_values = None  # Not implemented yet
        return shap_values_array, columns, shap_interaction_values
