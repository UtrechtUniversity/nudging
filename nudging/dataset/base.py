"""Base class for DataSet """
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseDataSet(ABC):
    """Base class for DataSet """

    truth = {}
    goal = None

    def __init__(self, file_path=None, idx=None, standard_df=None):
        if standard_df is not None:
            self.standard_df = standard_df
        if file_path:
            self.filename = file_path
            self.raw_df = self._load(file_path)
            self.standard_df = self._preprocess(self.raw_df)
        if idx is None and standard_df is not None:
            idx = np.arange(len(standard_df))
        self.idx = idx

    @abstractmethod
    def _load(self, file_path):
        raise NotImplementedError

    def __getattr__(self, item):
        """Easier access to nudge and outcomes"""
        if item in ["nudge", "outcome"] and item in self.standard_df:
            return self.standard_df[item].values
        return self.truth[item]

    def _preprocess(self, data_frame):
        """Do some general preprocessing after reading the file"""
        # Remove unused columns

        result = data_frame.copy(deep=True)
        used_columns = self.covariates + ["nudge", "outcome"]
        result = result.loc[:, used_columns]

        # Convert to a numeric type
        result = result.apply(pd.to_numeric)
        result["nudge"] = result["nudge"].astype(int)
        result["outcome"] = result["outcome"].astype(float)

        # Remove rows with NA in them
        result = result.dropna()

        # Set nudge type and domain
        if "nudge_type" not in result.columns:
            result["nudge_type"] = self.truth["nudge_type"]
        if "nudge_domain" not in result.columns:
            result["nudge_domain"] = self.truth["nudge_domain"]

        # Remove duplicates
        result = remove_duplicate_cols(result)

        # shuffle rows
        np.random.seed(1234123)
        result = result.sample(frac=1)

        return result

    def write_raw(self, path):
        """Write raw data to csv file"""
        self.raw_df.to_csv(path, index=False)

    def write_interim(self, path):
        """Write interim data (standard format) to csv file"""
        if self.goal == "decrease":
            self.standard_df["outcome"] = - self.standard_df["outcome"]

        self.standard_df.to_csv(path, index=False)

    @property
    def ate(self):
        """Compute the Average Treatment Effect"""
        ones = np.where(self.nudge == 1)[0]
        zeros = np.where(self.nudge == 0)[0]
        return np.mean(self.outcome[ones])-np.mean(self.outcome[zeros])

    @property
    def shape(self):
        """Get shape of dataframe"""
        return (self.standard_df.shape[0], self.standard_df.shape[1])

    @property
    def covariates(self):
        """By defaults all columns (ex. nudge/outcome) are covariates."""
        return [x for x in list(self.standard_df) if x not in ["nudge", "outcome"]]

    def kfolds(self, k=10):
        """Generator for k-folds"""
        zeros = np.where(self.standard_df["nudge"].values == 0)[0]
        ones = np.where(self.standard_df["nudge"].values == 1)[0]
        np.random.shuffle(zeros)
        np.random.shuffle(ones)

        one_split = np.array_split(ones, k)
        zero_split = np.array_split(zeros, k)
        for i_split in range(k):
            test_idx = np.append(one_split[i_split], zero_split[i_split])
            train_idx = np.delete(np.arange(len(self)), test_idx)
            yield self.train_test_split(train_idx=train_idx, test_idx=test_idx)

    def __len__(self):
        """Number of samples in de dataset"""
        return len(self.standard_df)

    def split(self, *idx_sets):
        """Split dataset into multiple smaller datasets.

        Arguments
        ---------
        idx_sets: np.ndarray
            A list of index sets, which determine how the dataset is split.
            It should be given in row indices (not ID's). It is not technically
            necessary that these sets are non-overlapping.

        Returns
        -------
        list[BaseDataset]:
            A list of new datasets split off the current one.
        """
        if len(idx_sets) == 0:
            return [self]

        ret = []
        for idx in idx_sets:
            if getattr(self, "truth", None) is not None:
                truth = split_truth(self.truth, idx, len(self.idx))
            else:
                truth = None
            new_idx = self.idx[idx]
            new_df = self.standard_df.iloc[idx]
            ret.append(self.from_df(new_df, truth, new_idx, self.goal))
        return ret

    def train_test_split(self, train=0.7, train_idx=None, test_idx=None):
        """Split the data into training and test set"""
        if train_idx is None or test_idx is None:
            train_idx = np.random.choice(len(self), size=int(train*len(self)),
                                         replace=False)
            test_idx = np.delete(np.arange(len(self)), train_idx)
        train_data, test_data = self.split(train_idx, test_idx)
        test_data.truth["outcome"] = test_data.outcome
        test_data.standard_df.drop(columns=["outcome"], inplace=True)
        return train_data, test_data

    def obs_cate(self, features, min_count=10):
        """Calculate observed cate"""
        data_obs = self.standard_df[
            self.standard_df["nudge"] == 1].groupby(features, as_index=False)["outcome"].mean()
        data_obs["cate_obs"] = \
            data_obs["outcome"] - self.standard_df[self.standard_df["nudge"] == 0].groupby(
            features, as_index=False)["outcome"].mean()["outcome"]
        data_obs["count_nudge"] = self.standard_df[self.standard_df["nudge"] == 1].groupby(
            features, as_index=False)["outcome"].size()["size"]
        data_obs["count_control"] = self.standard_df[self.standard_df["nudge"] == 0].groupby(
            features, as_index=False)["outcome"].size()["size"]

        data_obs = data_obs[
            (data_obs["count_nudge"] > min_count) & (data_obs["count_control"] > min_count)]

        return data_obs["cate_obs"]

    @classmethod
    def from_df(cls, standard_df, truth, idx=None, goal=None):
        """Create dataset from dataframe.

        Arguments
        ---------
        standard_df: pd.DataFrame
            Input data to create the dataset from.
        truth: dict
            Dictionary with truths such as nudge_type, outcome, cate, etc.
        idx: np.ndarray
            Original indices, they can be used as ID's while propagating
            through multiple splits.
        goal: str
            Either increase or decrease for better outcomes.

        Returns
        -------
        BaseDataset:
            Return the dataset created.
        """
        dataset = cls(idx=idx, standard_df=standard_df)
        dataset.truth = truth
        dataset.goal = goal
        return dataset


def split_truth(truth, idx, n_total_idx):
    """Split the truth dictionary."""
    new_truth = {}
    for key, value in truth.items():
        # If it is an array with the same length as the original number of
        # samples then only return the idx indices.
        if isinstance(value, np.ndarray) and len(value) == n_total_idx:
            new_truth[key] = value[idx]
        else:
            new_truth[key] = value
    return new_truth


def remove_duplicate_cols(data_frame):
    """Some columns may be duplications of each other, remove them."""
    cols = list(data_frame.columns)
    unq_cols, counts = np.unique(cols, return_counts=True)
    if len(unq_cols) == len(cols):
        return data_frame

    result = data_frame.copy()
    unq_zip = dict(zip(unq_cols, counts))
    for col_name, count in unq_zip.items():
        if count == 1:
            continue
        col_pos = np.where(np.array(cols) == col_name)[0]
        new_cols = np.copy(cols)
        for i, i_pos in enumerate(col_pos):
            new_cols[i_pos] = col_name+"_"+str(i)
        result.columns = new_cols
        for i, i_pos in enumerate(col_pos):
            all_same = np.all(
                result[col_name+"_"+str(i)].values == result[col_name+"_"+str(0)].values)
            if not all_same:
                raise ValueError("Reading two columns with the same name, but different data")
            if i > 0:
                result.drop([col_name+"_"+str(i)], inplace=True, axis=1)
        result.rename(columns={col_name+"_0": col_name}, inplace=True)
        cols = result.columns
    return result
