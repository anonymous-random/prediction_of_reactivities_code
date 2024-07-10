from sklearn.linear_model import Lasso

from src.analysis.machine_learning.LinearAnalyzer import LinearAnalyzer


class LassoAnalyzer(LinearAnalyzer):
    """
    This class is the specific implementation of the Lasso regression model using the standard Sklearn implementation
    (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html). Inherits from LinearAnalyzer.
    For class attributes, see LinearAnalyzer. Hyperparameters to tune are defined in the config.
    """

    def __init__(self, config, output_dir):
        """
        Constructor method of the LassoAnalyzer class.

        Args:
            config: YAML config determining specifics of the analysis
            output_dir: Specific directory where the results are stored
        """
        super().__init__(config, output_dir)
        self.model = Lasso(random_state=self.config["analysis"]["random_state"])
