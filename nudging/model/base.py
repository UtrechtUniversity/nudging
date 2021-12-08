""" Model classes"""
from abc import ABC
from sklearn.base import clone


class BaseModel(ABC):
    """Base class for model"""
    predictors = None

    def __init__(self, model, predictors=None):
        self.model = model
        self.predictors = predictors

    @staticmethod
    def preprocess(data_frame):
        """Standardize outcome

        Arguments
        ---------
        data_frame: pd.DataFrame
            Input dataframe to be standardized. If not implemented by a
            derived class nothing will be done to it.

        Returns
        -------
        output_data_frame: pd.DataFrame
            Standardized dataframe
        """
        return data_frame

    def train(self, data):
        """Train model"""
        # Drop rows with missing values for predictors
        df_nonan = data.dropna(subset=self.predictors, inplace=False)
        self.model.fit(
            df_nonan[self.predictors],
            df_nonan["outcome"]
        )

    def predict_outcome(self, data):
        """Predict outcome of given samples.

        This method only works for a classifier that was initialized
        with "probability=True".

        Arguments
        ---------
        data: pd.DataFrame
            Input dataframe to compute the outcome of.

        Returns
        -------
        probabilities: np.ndarray
            Probabilities of a possitive outcome.
        """
        return self.model.predict_proba(data[self.predictors])[:, 1]

    def predict_cate(self, data):
        """Predict conditional average treatment effect"""
        if self.predictors is None:
            self.predictors = [x for x in list(data)
                               if x not in ["nudge", "outcome"]]
        nudge_data = data[self.predictors].copy(deep=True)
        nudge_data["nudge"] = 1
        control_data = data[self.predictors].copy(deep=True)
        control_data["nudge"] = 0
        nudge_pred = self.predict_outcome(nudge_data)
        control_pred = self.predict_outcome(control_data)
        cate = nudge_pred - control_pred
        return cate

    def _X_nudge_outcome(self, data):
        return *self._X_nudge(data),  data["outcome"].values

    def _X_nudge(self, data):
        if self.predictors is None:
            self.predictors = [x for x in list(data)
                               if x not in ["nudge", "outcome"]]
        X = data[self.predictors].values
        nudge = data["nudge"].values
        return X, nudge

    def clone(self):
        """Create a clone of itself.
        """
        new_clone = self.__class__(clone(self.model))
        new_clone.predictors = self.predictors
        return new_clone
