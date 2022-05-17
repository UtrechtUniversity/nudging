"""Contains the classes for real datasets"""

from pathlib import Path
from enum import IntEnum

import numpy as np
import pandas as pd

from nudging.dataset.base import BaseDataSet


class Gender(IntEnum):
    """Gender enum"""
    MALE = 1
    FEMALE = 0


class Group(IntEnum):
    """Group enum"""
    CONTROL = 0
    NUDGE = 1


class RealDataset(BaseDataSet):
    """Base class for real world datasets"""
    def __init__(self, standard_df, raw_df, file_path, idx=None):
        self.raw_df = raw_df
        self.file_path = file_path
        super().__init__(standard_df, idx=idx)

    @classmethod
    def _load(cls, file_path, encoding=None):
        suffix = Path(file_path).suffix.lower()
        if suffix == ".dta":
            return pd.read_stata(file_path)
        if suffix == ".csv":
            return pd.read_csv(file_path, encoding=encoding)
        raise ValueError(f"Reader doesn't know how to read files with extension '{suffix}'")

    @classmethod
    def from_file(cls, file_path):
        """Create dataset from file"""
        if Path(file_path).is_dir():
            file_path = Path(file_path, cls._default_filename)
        raw_df = cls._load(file_path)
        standard_df = cls._preprocess(raw_df)
        return cls(standard_df, raw_df, file_path)

    @classmethod
    def _preprocess(cls, data_frame):
        """Do some general preprocessing after reading the file"""
        # Remove unused columns

        result = data_frame.copy(deep=True)
        used_columns = cls.truth["covariates"] + ["nudge", "outcome"]
        result = result.loc[:, used_columns]

        # Convert to a numeric type
        result = result.apply(pd.to_numeric)
        result["nudge"] = result["nudge"].astype(int)
        result["outcome"] = result["outcome"].astype(float)

        # Remove rows with NA in them
        result = result.dropna()

        # Set nudge type and domain
        try:
            if "nudge_type" not in result.columns:
                result["nudge_type"] = cls.truth["nudge_type"]
        except (AttributeError, KeyError):
            result["nudge_type"] = -1
        try:
            if "nudge_domain" not in result.columns:
                result["nudge_domain"] = cls.truth["nudge_domain"]
        except AttributeError:
            result["nudge_domain"] = -1

        # Remove duplicates
        result = remove_duplicate_cols(result)

        # shuffle rows
        np.random.seed(1234123)
        result = result.sample(frac=1)

        return result

    def _split_once(self, idx):
        new_idx = self.idx[idx]
        new_df = self.standard_df.iloc[idx]
        new_raw_df = self.raw_df.iloc[idx]
        return self.__class__(new_df, new_raw_df, self.file_path, new_idx)

    def write_raw(self, path):
        """Write raw data to csv file"""
        self.raw_df.to_csv(path, index=False)


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


def convert_categorical(df, col_old, conversion, col_new=None):
    """Convet categories"""
    if col_new is None:
        col_new = col_old
    orig_values = df[col_old].values
    good_rows = np.isin(orig_values, list(conversion))
    df = df.iloc[good_rows]
    orig_values = df[col_old].values
    cat_values = np.zeros(len(df), dtype=int)
    for src, dest in conversion.items():
        cat_values[orig_values == src] = dest
    df.loc[:, col_new] = cat_values
    return df
