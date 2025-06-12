import json
import os

import numpy as np
import shap
from joblib import Parallel, delayed

from src.analysis.machine_learning.BaseMLAnalyzer import BaseMLAnalyzer


class LinearAnalyzer(BaseMLAnalyzer):
    """
    This class serves as a template for the linear models (lasso, linear_baseline_model) and implements methods
    that do not differ between the both linear models. Inherits from BaseMLAnalyzer. For attributes, see
    BaseMLAnalyzer. The model attribute is defined in the subclasses.
    """

    def __init__(self, config, output_dir):
        """
        Constructor method of the LinearAnalyzer class.

        Args:
            config: YAML config determining specifics of the analysis
            output_dir: Specific directory where the results are stored
        """
        super().__init__(config, output_dir)
        self.model = None
        self.model_coefs = dict()

    def get_average_coefficients(self):
        """Calculate the average coefficients across all outer cv loops stored in self.best_models."""
        feature_names = self.X.columns.tolist()
        # feature_names = self.current_feature_col_order # this would fix the problem
        for rep in range(1, self.num_repeats + 1):
            coefs_dict_lst = []
            # print(self.best_models.keys())
            for model in self.best_models[f"rep_{rep}"]:
                # Create a dictionary with feature names as keys and coefficients as values
                coefs_dict = dict(zip(feature_names, model.coef_))
                sorted_coefs_dict = dict(
                    sorted(
                        coefs_dict.items(), key=lambda item: abs(item[1]), reverse=True
                    )
                )
                coefs_dict_lst.append(sorted_coefs_dict)
            self.model_coefs[f"rep_{rep}"] = coefs_dict_lst

    def store_coefficients(self):
        """This method stores the models coefficients as a .JSON using the folder and name specified in the config."""
        coefs_filename = os.path.join(
            self.output_dir, self.config["analysis"]["output_filenames"]["coefs"]
        )
        os.makedirs(os.path.dirname(coefs_filename), exist_ok=True)
        coef_dict_lst = self.model_coefs
        with open(coefs_filename, "w") as file:
            json.dump(coef_dict_lst, file, indent=4)

    def calculate_shap_for_instance(self, n_instance, instance, explainer):
        """
        Calculates linear SHAP values for a single instance for parallelization.

        Args:
            n_instance: Number of a certain individual to calculate SHAP values for
            instance: 1d-array, represents the feature values for a single individual
            explainer: shap.LinearExplainer

        Returns:
            explainer(instance.reshape(1, -1)).values: array containing the SHAP values for "n_instance"
        """
        return explainer(instance.reshape(1, -1)).values

    def calculate_shap_values(self, X, pipeline):
        """
        This function calculates linear SHAP values for a given analysis setting. This includes applying the
        preprocessing steps that were applied in the pipeline (e.g., scaling, RFECV if specified).
        It calculates the SHAP values using the explainers.Linear. SHAP calculations can be parallelized.

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
        explainer_lin_lasso = shap.explainers.Linear(
            pipeline.named_steps["model"], X_processed
        )
        shap_values_list = Parallel(
            n_jobs=self.config["analysis"]["parallelize"]["shap_n_jobs"], verbose=0
        )(
            delayed(self.calculate_shap_for_instance)(
                n_instance, instance, explainer_lin_lasso
            )
            for n_instance, instance in enumerate(X_processed)
        )
        # Convert list of arrays to single array
        shap_values_array = np.vstack(shap_values_list)
        shap_interaction_values = None
        return shap_values_array, columns, shap_interaction_values
