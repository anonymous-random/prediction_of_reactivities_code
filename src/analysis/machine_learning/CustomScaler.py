import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class CustomScaler(BaseEstimator, TransformerMixin):
    """
    This class scales only the continuous columns and ignore the binary columns. This custom class
    works with metadata_routing, which does not work with using the ColumnTransformer.

    Attributes:
         cols_to_scale: Columns indicating the continuous features that should be scaled.
         scaler: Standardscaler object from sklearn, calculates z scores [(x - M) / SD]
    """

    def __init__(self, cols_to_scale):
        """
        Constructor method of the CustomScaler class.

        Args:
            cols_to_scale: Columns indicating the continuous features that should be scaled.
        """
        self.cols_to_scale = cols_to_scale
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        """
        This method fits the scaler to the selected columns.

        Args:
            X: df, features to be scaled.
            y: criterion, not used in this implementation

        Returns:
            self: the CustomScaler object itself.
        """
        self.scaler.fit(X[self.cols_to_scale])
        return self

    def transform(self, X, y=None):
        """
        This method transforms the selected columns using the scaler object, drops the original columns
        and concatenate scaled and unscaled columns to a numpy array that is returned.

        Args:
            X: df, features to be scaled.
            y: criterion, not used in this implementation

        Returns:
            X_processed: ndarray, containing the scaled features.
        """
        X_scaled = self.scaler.transform(X[self.cols_to_scale])
        X_dropped = X.drop(self.cols_to_scale, axis=1)
        X_processed = np.hstack([X_scaled, X_dropped.to_numpy()])
        # org_col_order = X.columns.tolist()
        # current_order = self.cols_to_scale.tolist() + X_dropped.columns.tolist()
        # Note: Orders differ, is handled in the feature importance analysis
        return X_processed
