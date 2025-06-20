import json
import math
import os
import pickle
from json import JSONDecodeError
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler


class ResultAnalyzer:
    """
    This class is used to summarize and analyze the raw results obtained in the machine learning-based analysis.
    It averages the prediction results across repetitions and returns M and SD of the predictions results across
    repetitions and across outer folds for all metrics (R², RMSE, rho).
    Further, it calculates the average and the number of occurrences across outer folds of the linear model
    coefficients.
    Lastly, it averages the raw and absolute SHAP values and add these aggregates to the SHAP dct.
    Processed ML-results are stored in a separate folder.
    Due to memory constraints, this function does the result processing separately for a given
        analysis_type (e.g. main)
        study (e.g. ssc)
        esm_sample (e.g. coco_int)
    Thus, to process all results, we have to run this function repeatedly adjusting the specifications in the
    config or using loops in the main function.

    Attributes:
        config: YAML config determining certain specifications of the analysis.
        esm_sample: A given ESM-sample for which the results are processed.
        results: "str", root dir where all the raw machine learning results are stored
        col_order_before_scaling: Dct, containing the column order of the df containing the features before the
            custom scaler was applied, because applying this scaler changed the column order. The columns
            depend on the esm_sample, the feature inclusion strategy, and potentially the soc_int_var, therefore
            these parameters are mirrored in the Dict structure.
        col_order_after_scaling: Dct, containing the column order of the df containing the features before the
            custom scaler was applied, because applying this scaler changed the column order. The columns
            depend on the esm_sample, the feature inclusion strategy, and potentially the soc_int_var, therefore
            these parameters are mirrored in the Dict structure.
    """

    def __init__(self, config_path, esm_sample):
        """
        Constructor method of the ResultAnalyzer Class.

        Args:
            config_path: Path to the .YAML config file.
            esm_sample: A given ESM-sample for which the results are processed.
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.results = dict()
        self.esm_sample = esm_sample
        self.col_order_before_scaling = {}
        self.col_order_after_scaling = {}

    @property
    def result_config(self):
        """The part of the config file that concern the ResultAnalyzer class."""
        return self.config["analysis"]["result_analysis"]

    @property
    def result_base_path(self):
        """Data path of the raw results of the machine learning analysis."""
        return os.path.normpath(self.result_config["result_base_dir"])

    @property
    def feature_base_path(self):
        """Data path for the features of the specified analysis."""
        return os.path.normpath(self.result_config["features_base_dir"])

    @property
    def analysis_type(self):
        """Type of analysis. Is "main" or "None", because we did not include a "suppl" folder on the cluster."""
        return "main" if self.config["general"]["analysis"] == "main" else None

    @property
    def suppl_type(self):
        """Type of supplementary analysis, only defined if self.analysis == 'suppl', e.g. 'sep_ftf_cmc'."""
        return (
            None
            if self.analysis_type == "main"
            else self.config["general"]["suppl_type"]
        )

    @property
    def suppl_var(self):
        """Var of supplementary analysis, only defined if self.suppl_type exists, e.g. 'ftf'."""
        return (
            None
            if self.analysis_type == "main"
            else self.config["general"]["suppl_var"]
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
            if self.suppl_type in ["sep_ftf_cmc", "sep_pa_na"]
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
    def study(self):
        """Study, i.e., ssc."""
        return self.config["general"]["study"]

    @property
    def feature_result_inter_path(self):
        """
        Specific intermediate data path for the results and the features based on current analysis. Thus,
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
    def result_path(self):
        """Path for the cv results on the level of ESM-samples."""
        if self.suppl_type == "weighting_by_rel":
            return os.path.normpath(
                os.path.join(
                    self.result_base_path,
                    self.suppl_type,
                    self.suppl_var,
                    self.study,
                    self.esm_sample,
                )
            )
        else:
            return os.path.normpath(
                os.path.join(
                    self.result_base_path,
                    self.feature_result_inter_path,
                    self.esm_sample,
                )
            )

    @property
    def sum_results_folder(self):
        """Data path of the final results for the publication, specified in config."""
        return os.path.normpath(self.result_config["sum_results_folder"])

    @property
    def scoring_metric(self):
        """Scoring metric of the inner loop used for hyperparameter optimization, specified in config."""
        return self.config["analysis"]["scoring_metric"]["inner_cv_loop"]["name"]

    def apply_methods(self):
        """This function applies the processing methods specified in the config."""
        for method_info in self.result_config["methods"]:
            method_name = method_info["name"]
            args = [
                getattr(self, arg) if hasattr(self, arg) else arg
                for arg in method_info["args"]
            ]

            if method_name not in dir(ResultAnalyzer):
                raise ValueError(f"Method '{method_name}' is not implemented yet.")
            method = getattr(self, method_name)
            method(*args)

    def save_load_json_files(self, file_obj):
        """
        This function loads a given json file or checks if the json content from a file object is corrupted
        and attempts to fix it, and then loads the json file. We needed to implement this because occasionally
        the json_files were corrupted, with unexpected random characters.

        Args:
            file_obj: JSON file to process, contains e.g. the 10x10x10 CV results for each metric

        Returns:
            json_data: Dict, containing the (cleaned) content of the JSON file
        """
        try:
            # Reset file pointer to the start
            file_obj.seek(0)
            json_data = json.load(file_obj, object_hook=self.handle_nan)
            return json_data
        except JSONDecodeError:
            print("Unexpected characters, cleaning json content")
            # Reset file pointer to the start
            file_obj.seek(0)
            content = file_obj.read()
            clean_content = content.replace("\x00", "")
            # Reset the file and write the cleaned content
            file_obj.seek(0)
            file_obj.truncate()
            file_obj.write(clean_content)
            file_obj.flush()
            # Reset file pointer to the start to read again
            file_obj.seek(0)
            json_data = json.load(file_obj, object_hook=self.handle_nan)
            return json_data

    def load_data_from_folders(self):
        """
        This function walks through all subdirectories of a given root directory and extracts all raw machine learning
        results from these directories. It loads the corresponding JSON files and puts its content in a dictionary
        that mirrors the directory structure for unambiguous assignment.
        Create a nested dict that contains all machine learning results in a given root_dir
        based on the directory structure.
        """
        for root, dirs, files in os.walk(self.result_path):
            if not dirs:  # terminal directory
                relative_path = os.path.relpath(root, self.result_path)
                components = relative_path.split(os.path.sep)
                # Navigate and create nested dictionaries
                current_level = self.results
                for comp in components:
                    current_level = current_level.setdefault(comp, {})
                # Load and scale features
                features = self.get_and_scale_features(components)
                current_level["features"] = features
                # Load JSON files into the terminal dictionary
                for file in files:
                    print(components, file)
                    with open(os.path.join(root, file), "r+") as f:
                        current_level[file] = self.save_load_json_files(f)
        print("loaded data")

    def get_and_scale_features(self, components):
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
                or the sample (the relative_path representents the folder structure) of the data.

        Returns:
            features_result: df, containing the scaled features and its values with the correct assignment
        """
        # Construct the base path and filename
        dataset = self.esm_sample
        feature_inclusion_strategy = components[0]
        model = components[1]
        if self.study == "ssc":
            soc_int_var = components[2]
            file_name = f"{dataset}_{soc_int_var}_one_hot_encoded_preprocessed.pkl"
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
            self.feature_result_inter_path,
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
            dataset=dataset,
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
            dataset=dataset,
            fis=feature_inclusion_strategy,
            soc_int_var=soc_int_var,
            time_var="after",
        )
        return features_result

    def set_column_order_attributes(
        self, features, dataset, fis, soc_int_var, time_var
    ):
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
            dataset: str, esm-sample, e.g., "coco_int"
            fis: str, feature_inclusion_strategy, e.g., "scale_means"
            soc_int_var: str, social situation variavle, e.g., "social_interaction"
            time_var: str, determining if it is the column order before or after scaling, therefore, is must be
                "before" or "after"
        """
        if dataset not in getattr(self, f"col_order_{time_var}_scaling"):
            getattr(self, f"col_order_{time_var}_scaling")[dataset] = {}

        if self.study == "ssc":
            if fis not in getattr(self, f"col_order_{time_var}_scaling")[dataset]:
                getattr(self, f"col_order_{time_var}_scaling")[dataset][fis] = {}
            if (
                soc_int_var
                not in getattr(self, f"col_order_{time_var}_scaling")[dataset][fis]
            ):
                getattr(self, f"col_order_{time_var}_scaling")[dataset][fis][
                    soc_int_var
                ] = {}
            getattr(self, f"col_order_{time_var}_scaling")[dataset][fis][
                soc_int_var
            ] = features.columns.tolist()

    def summarize_results(self, current_level, current_path):
        """
        Wrapper function for the specific processing operations (summarizing cv_results, shap_values, lin_coefs).
        It is a recursive function that processes terminal level data if the key conditions are met.
        The file structure of the terminal directories is given to define a stopping criterion for
        the recursive function.
        If this processes a terminal directory that contains "shap_ia_values", this is added to the keys.
        Then, every directory must contain shap_ia_values. Therefore, the results with and without shap_ia_values
        are stored in different root directories and processed separately.

        Args:
            current_level: Dict, Current Dict that is processed. If is just contains other dicts, recursion continues
            current_path: str, current_path to current_level that is processed by the function
        """
        json_keys_non_linear = self.result_config["keys_non_linear"]
        # Copy nonlinear keys and add "lin_model_coefficients"
        json_keys_linear = (
            json_keys_non_linear + self.result_config["additional_keys_linear"]
        )

        # Replicate folder structure in the new base folder and save the results
        new_folder_path = self.get_new_path(current_path=current_path)
        print(new_folder_path)
        os.makedirs(new_folder_path, exist_ok=True)

        if set(current_level.keys()) == set(json_keys_linear) or set(
            current_level.keys()
        ) == set(json_keys_non_linear):
            print(current_path)
            if set(current_level.keys()) == set(json_keys_linear):
                processed_data = {
                    "cv_results.json": self.process_cv_results(
                        current_level["cv_results.json"]
                    ),
                    "lin_model_coefficients.json": self.process_lin_coefs(
                        current_level["lin_model_coefficients.json"],
                        new_folder_path,
                    ),
                    "shap_values.json": self.process_shap_values(
                        current_level["shap_values.json"],
                    ),
                }
            elif set(current_level.keys()) == set(json_keys_non_linear):
                processed_data = {
                    "cv_results.json": self.process_cv_results(
                        current_level["cv_results.json"]
                    ),
                    "shap_values.json": self.process_shap_values(
                        current_level["shap_values.json"],
                    ),
                }
                # just pass the shap ia values, processing did happen on the cluster due to its affordances
                if self.result_config["shap_ia_values"]:
                    processed_data["shap_ia_values.json"] = current_level[
                        "shap_ia_values.json"
                    ]

            # Save the processed data for each key
            if self.result_config["store_results"]:
                for key, value in processed_data.items():
                    output_file = os.path.join(new_folder_path, key)
                    with open(output_file, "w") as f:
                        json.dump(value, f, indent=4)
        else:
            # This level of the dictionary contains other dictionaries, so recurse further
            for key, sub_level in current_level.items():
                print(current_path, key)
                if not isinstance(key, str):
                    raise TypeError(
                        f"Key must be a string, got {type(key)} instead. "
                        f"Check JSON keys for stopping the recursion!"
                    )
                else:
                    new_path = os.path.join(current_path, key)
                self.summarize_results(sub_level, new_path)

    def get_new_path(self, current_path):
        """
        This function returns the path were the processed results are stored, based on the given path.

        Args:
            current_path: str, a given path corresponding to a certain Dict
        """
        if self.suppl_type == "weighting_by_rel":
            new_folder_path = os.path.normpath(
                os.path.join(
                    self.result_config["sum_results_folder"],
                    self.suppl_type,
                    self.suppl_var,
                    self.study,
                    self.esm_sample,
                    current_path,
                )
            )
        else:
            new_folder_path = os.path.normpath(
                os.path.join(
                    self.result_config["sum_results_folder"],
                    self.feature_result_inter_path,
                    self.esm_sample,
                    current_path,
                )
            )
        return new_folder_path

    def process_cv_results(self, data):
        """
        This function summarized the prediction results obtained using 10x10x10 CV.
        Specifically, it
            takes the average across repetitions, so that we get 1 predictability estimate per repetition.
            calculates M and SD across 100 outer folds and across 10 repetitions.

        Args:
            data: Dict containing the prediction results for a given analysis settings for all metrics.
        """
        result_dict = dict()
        metric_lst = list(
            set(
                [
                    metric
                    for rep, metrics in data.items()
                    for metric, values in metrics.items()
                ]
            )
        )
        for metric in metric_lst:
            if metric == "rmse":
                data = self.correct_rmse(data)
            result_dict[metric] = dict()
            result_dict[metric][f"mean_{metric}_per_rep"] = [
                np.nanmean(rep_vals[metric]) for rep, rep_vals in data.items()
            ]
            result_dict[metric][f"mean_{metric}_across_reps"] = np.nanmean(
                result_dict[metric][f"mean_{metric}_per_rep"]
            )
            result_dict[metric][f"mean_{metric}_across_outer_cvs"] = np.nanmean(
                [value for rep in data for value in data[rep][metric]]
            )
            result_dict[metric][f"std_{metric}_across_reps"] = np.std(
                result_dict[metric][f"mean_{metric}_per_rep"]
            )
            result_dict[metric][f"std_{metric}_across_outer_cvs"] = np.std(
                [value for rep in data for value in data[rep][metric]]
            )
        return result_dict

    @staticmethod
    def correct_rmse(data):
        """
        This method corrects the rmse metric obtained on the cluster. Due to a minor mistake, we always calculated
        the mse and also varied if it is displayed as positive or negative.
        To correct this, it is multiplied with -1 (if mse is negative) and its square root is taken.

        Args:
            data: Dict containing the prediction results for a given analysis settings for all metrics.

        Returns:
            data: Dict, data were the RMSE was corrected
        """
        for rep_key in data:
            rep_data = data[rep_key]
            for i, rmse_value in enumerate(rep_data["rmse"]):
                if rmse_value < 0:
                    rmse_value = -rmse_value
                rep_data["rmse"][i] = math.sqrt(rmse_value)
        return data

    def process_lin_coefs(self, data, path):
        """
        This function processes the coefficients obtained in the linear models (LASSO, LBM). It
            corrects the column order assignment
            calculates averages and number of non-zero occurrences across outer folds
            sets the results as key:value pairs in the result dict

        Args:
            data: Dict, containing the lin_model_coefficients
            path: str, Path to data

        Returns:
            result_dict: Dict containing the avg and non-zero occurrences of features coefficients
        """
        result_dict = {}

        # Fix coefficient name - coefficient value assignment
        col_order_before_scaling, col_order_after_scaling = self.get_column_orders(
            path=path
        )
        data_old = data.copy()
        data = self.fix_coefficient_name_value_assignment(
            data=data_old,
            col_order_before_scaling=col_order_before_scaling,
            col_order_after_scaling=col_order_after_scaling,
        )

        # Calculate the sum and non-zero counts of each coefficient
        all_coef_names = set(
            key for rep in data.values() for fold in rep for key in fold
        )
        sums = {coef_name: 0 for coef_name in all_coef_names}
        counts = {coef_name: 0 for coef_name in all_coef_names}
        total_counts = {coef_name: 0 for coef_name in all_coef_names}
        for rep in data.values():
            for fold in rep:
                for coef_name in all_coef_names:
                    coef_value = fold.get(coef_name, 0)
                    if not coef_value:  # otherwise coef_value can be None
                        coef_value = 0
                    sums[coef_name] += coef_value
                    total_counts[coef_name] += 1
                    if coef_value != 0:
                        counts[coef_name] += 1

        # Calculate averages and number of non-zero occurrences
        avg_coefs = {
            coef_name: sums[coef_name] / total_counts[coef_name]
            for coef_name in all_coef_names
        }
        non_zero_counts = counts

        # Store results in dictionary
        result_dict["avg_coefs_across_outer_folds"] = avg_coefs
        result_dict["num_occurences_across_outer_folds"] = non_zero_counts
        return result_dict

    def get_column_orders(self, path):
        """
        This function gets the right column order before and after scaling for a given path, using the
        predefined orders that are stored as class attributes and the given path.

        Args:
            path: str, Data path indicating the current analysis (which study, fis, soc_int_var)

        Returns:
            col_order_before_scaling: The column order of the features before applying the Scaler
            col_order_after_scaling: The column order of the features after applying the Scaler
        """
        path_parts = Path(path).parts
        study_index = (
            path_parts.index(self.esm_sample) if self.esm_sample in path_parts else None
        )
        if self.study == "ssc":
            esm_sample, fis, soc_int_var = path_parts[study_index : study_index + 2] + (
                path_parts[-1],
            )
            col_order_before_scaling = self.col_order_before_scaling[esm_sample][fis][
                soc_int_var
            ]
            col_order_after_scaling = self.col_order_after_scaling[esm_sample][fis][
                soc_int_var
            ]
        else:
            raise ValueError("Unknown study, must be ssc")

        return col_order_before_scaling, col_order_after_scaling

    def fix_coefficient_name_value_assignment(
        self, data, col_order_before_scaling, col_order_after_scaling
    ):
        """
        This function corrects the name value assignment for the linear model coefficients.
        First it reorders the name:value pairs to the column order before scaling.
        Then it reorders the name:value pairs according to the colum order after scaling.

        Args:
            data: A nested dictionary (reps (dct) -> outer folds (lst) -> coefficient_names: values)
            col_order_before_scaling: column order of the features before scaling
            col_order_after_scaling: column order of the features after scaling

        Returns:
            corrected_data: Dict with the same structure as data containing the corrected assignments
        """
        # this preserves the key-value assignment but reorders the keys in the old order
        dct_1 = {}
        for outer_key, list_of_dicts in data.items():
            new_list_of_dicts = []
            for inner_dict in list_of_dicts:
                new_inner_dict = {
                    new_key: inner_dict.get(new_key)
                    for new_key in col_order_before_scaling
                }
                new_list_of_dicts.append(new_inner_dict)
            dct_1[outer_key] = new_list_of_dicts
        # This corrects the wrong key value assignment by the order of the scales features
        data_corrected = {}
        for outer_key, list_of_dicts in dct_1.items():
            new_list_of_dicts = []
            for inner_dict in list_of_dicts:
                new_inner_dict = dict(zip(col_order_after_scaling, inner_dict.values()))
                new_list_of_dicts.append(new_inner_dict)
            data_corrected[outer_key] = new_list_of_dicts
        return data_corrected

    def process_shap_values(self, shap_data):
        """
        Custom processing for the shap values obtained. This includes summarizing the absolute shap values across
        samples to get a single importance score per variable for the training and the test set.

        Args:
            shap_data: Dict, containing the raw SHAP values

        Returns:
            result_dct: Dict, containing the raw and the processed SHAP values
        """
        result_dict = shap_data.copy()
        del result_dict["feature_names"]  # to prevent ambiguity
        result_dict["train"]["abs_avg_across_reps_samples"] = np.mean(
            np.abs(shap_data["train"]["avg_across_reps"]), axis=0
        ).tolist()
        result_dict["test"]["abs_avg_across_reps_samples"] = np.mean(
            np.abs(shap_data["test"]["avg_across_reps"]), axis=0
        ).tolist()
        return result_dict

    # Custom function to handle NaN values and convert them to None in JSON
    @staticmethod
    def handle_nan(obj):
        """
        Function to handle NaN values in JSON files. Used as object_hook in json.load().

        Args:
            obj: Any object

        Returns:
            [None, obj]: None if obj math.isnan, else obj
        """
        if isinstance(obj, float) and math.isnan(obj):
            return None
        return obj
