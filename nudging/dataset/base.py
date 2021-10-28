"""Base class for DataSet """
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseDataSet(ABC):
    """Base class for DataSet """

    nudge_type = None
    nudge_domain = None
    goal = None

    def __init__(self, file_path=None):
        if file_path:
            self.filename = file_path
            self.raw_df = self._load(file_path)
            self.standard_df = self._preprocess(self.raw_df)

    @abstractmethod
    def _load(self, file_path):
        raise NotImplementedError

    def __getattr__(self, item):
        """Easier access to nudge and outcomes"""
        if item in ["nudge", "outcome"]:
            return self.standard_df[item].values
        raise AttributeError(f'{self.__class__.__name__}.{item} is invalid.')

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
            result["nudge_type"] = self.nudge_type
        if "nudge_domain" not in result.columns:
            result["nudge_domain"] = self.nudge_domain

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
    def data(self):
        """Split the data into FM + outcome + nudge"""
        return split_df(self.standard_df)

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
            yield split_df(self.standard_df, train_idx, test_idx)

    def __len__(self):
        """Number of samples in de dataset"""
        return len(self.standard_df)

    def split(self, train=0.7):
        """Split the data into training and test set"""
        train_idx = np.random.choice(len(self), size=int(train*len(self)),
                                     replace=False)
        test_idx = np.delete(np.arange(len(self)), train_idx)
        return split_df(self.standard_df, train_idx, test_idx)

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


def split_df(data_frame, *idx_set):
    """Split the dataset into multiple datasets."""
    if len(idx_set) == 0:
        return convert_df(data_frame)
    return [convert_df(data_frame.iloc[idx], idx) for idx in idx_set]


def convert_df(data_frame, idx=None):
    """Split the dataset into FM/outcome/nudge"""
    if idx is None:
        idx = np.arange(len(data_frame))
    nudge = data_frame["nudge"]
    outcome = data_frame["outcome"]
    data = data_frame.drop(["nudge", "outcome"], axis=1)
    return {"X": data, "outcome": outcome, "nudge": nudge, "idx": idx}


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
