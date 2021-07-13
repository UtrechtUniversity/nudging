from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseDataSet(ABC):
    covariates = []

    def __init__(self, file_path):
        self.filename = file_path
        self.raw_df = self._load(file_path)
        self.df = self._preprocess(self.raw_df)

    @abstractmethod
    def _load(self, fp):
        raise NotImplementedError

    def _preprocess(self, df):
        # Remove unused columns
        used_columns = self.covariates + ["nudge", "outcome"]
        df = df.loc[:, used_columns]

        # df["nudge"] = df["nudge"].astype(int)
        # df["outcome"] = df["outcome"].astype(float)

        # Remove rows with NA in them
        df = df.dropna()

        # shuffle rows
        # np.random.seed(1234123)
        # df = df.sample(frac=1)
        return df

    @property
    def data(self):
        return split_df(self.df)

    def write_raw(self, path):
        self.raw_df.to_csv(path, index=False)


    def kfolds(self, k=10):
        zeros = np.where(self.df["nudge"].values == 0)[0]
        ones = np.where(self.df["nudge"].values == 1)[0]

        def balance(one, zero):
            if len(one) < len(zero):
                zero = np.random.choice(zero, size=len(one))
            elif len(zero) > len(one):
                one = np.random.choice(one, size=len(zero))
            combi = np.append(zero, one)
            np.random.shuffle(combi)
            return combi

        one_split = np.array_split(ones, k)
        zero_split = np.array_split(zeros, k)
        for i_split in range(k):
            test_idx = np.append(one_split[i_split], zero_split[i_split])
            train_idx = np.delete(np.arange(len(self.df)), test_idx)
#             df_clean = self.df.drop(["outcome", "nudge"], axis=1)
            df_train, df_test = self.df.iloc[train_idx], self.df.iloc[test_idx]
            yield split_df(df_train), split_df(df_test)


def split_df(df):
    nudge = df["nudge"].values
    outcome = df["outcome"].values
    X = df.drop(["nudge", "outcome"], axis=1).values.astype(float)
    return {"X": X, "nudge": nudge, "outcome": outcome}
