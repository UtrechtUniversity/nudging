"""Module for loading dataset from dataframe."""

import numpy as np
from sklearn.linear_model import ARDRegression

from nudging.dataset.real import RealDataset
from nudging.model.biregressor import BiRegressor


class FileDataset(RealDataset):
    """Class for creating a dataset from a dataframe."""
    truth = {}

    @classmethod
    def from_dataframe(cls, df):
        """Create a dataset from a dataframe"""
        standard_df = cls._preprocess(df)
        return cls(standard_df, df, None)

    # def train(self, xvalidate=False):
    #     self.xvalidate = xvalidate
    #     if xvalidate:
    #         self.model = []
    #         for train_data, test_data in dataset.kfolds(k=k):
    #             model.train(model.preprocess(train_data.standard_df))
    #             cur_cate = model.predict_cate(test_data.standard_df)
    #             results.append((cur_cate, test_data.idx))
    #
    #     self.model = BiRegressor(ARDRegression())
    #     self.model.train(self.model.preprocess(self.standard_df))

    def predict_cate(self, new_df=None):
        """Predict the cate of the current dataset.

        new_df can be another dataframe with the same structure.
        In that case no crossvalidation is done.
        """
        model = BiRegressor(ARDRegression())
        if new_df is None:
            # results = []
            cate = np.zeros(len(self))

            for train_data, test_data in self.kfolds(k=10):
                model.train(model.preprocess(train_data.standard_df))
                cur_cate = model.predict_cate(test_data.standard_df)
                cate[test_data.idx] = cur_cate
        else:
            new_dataset = FileDataset.from_dataframe(new_df)
            model.train(self.standard_df)
            cate = model.predict_cate(new_dataset.standard_df)

        return cate
