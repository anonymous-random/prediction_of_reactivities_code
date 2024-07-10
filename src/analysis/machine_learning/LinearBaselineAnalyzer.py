from sklearn.linear_model import LinearRegression

from src.analysis.machine_learning.LinearAnalyzer import LinearAnalyzer


class LinearBaselineAnalyzer(LinearAnalyzer):
    """
    This class is the specific implementation of the linear baseline model, which is a ordinary multiple regression
    containing only age, gender, educational attainment and the Big Five scale means as features.
    Inherits from Linear Analyzer. For class attributes, see BaseMLAnalyzer.
    """

    def __init__(self, config, output_dir):
        """
        Constructor method of the LinearAnalyzer class.

        Args:
            config: YAML config determining specifics of the analysis
            output_dir: Specific directory where the results are stored
        """
        super().__init__(config, output_dir)
        self.model = LinearRegression()

    @property
    def feature_inclusion_strategy(self):
        """
        This method sets feature_inclusion_strategy always to "scale_means" for the linear_baseline_model.
        This overrides the base implementation defined in the BaseMLAnalyzer and used in the other subclasses.
        """
        return "scale_means"

    def select_features(self):
        """
        This method selects the features used by the baseline model so that predictions are only based on this
        feature subset. It updates the feature attribute accordingly.
        Note: In the linear_baseline_model, we use dummy-coded features instead of one-hot encoded features,
        because otherwise multicollinearity issues may prevent meaningful and reproducible results.
        """
        X = getattr(self, "X")
        feature_lst = [
            col
            for col in X.columns
            if any(
                col.startswith(prefix)
                for prefix in ["age", "sex", "educational_attainment", "bfi2"]
            )
        ]
        X = X[feature_lst]
        X = X.drop("sex_clean_1", axis=1)  # dummy coding, remove reference category
        setattr(self, "X", X)
