"""DataSet class for Andreas Lieberoth et al 2018 (https://doi.org/10.1016/j.trf.2018.02.016)"""
import pandas as pd
import numpy as np

from nudging.dataset.real import RealDataset, Gender, Group


class Lieberoth(RealDataset):
    """DataSet class for Andreas Lieberoth et al 2018"""
    _default_filename = "lieberoth.csv"

    truth = {
        "covariates": ["age", "gender"],
        "nudge_type": 3,
        "nudge_domain": 3,
        # Control and nudge classes in original data:
        "control_value": "control",
        "nudge_value": "nudge",
        # Gender classes in original data:
        "male": "1",
        "female": "8",
        # nudge is successfull if outcome increased
        "goal": "increase",

    }

    @classmethod
    def _load(cls, file_path, encoding="iso-8859-1"):
        return super()._load(file_path, encoding=encoding)

    @classmethod
    def _preprocess(cls, data_frame):
        """Convert Lieberoth data to dataframe with standard format
        Args:
            data_frame (pandas.DataFrame): original data
        Returns:
            pandas.DataFrame: dataframe containing covariates, outcome, nudge
        """
        df = data_frame.copy()
        df = _convert_categorical(df, "group", {"control": Group.CONTROL, "nudge": Group.NUDGE},
                                  col_new="nudge")
        df = _convert_categorical(df, "gender", {"8": Gender.FEMALE, "1": Gender.MALE})
        df["age"] = pd.to_numeric(df["age"], errors='coerce').round()
        df["outcome"] = df["swtot"]

        # this removes unused columns of original data
        return super()._preprocess(df)


def _convert_categorical(df, col_old, conversion, col_new=None):
    if col_new is None:
        col_new = col_old
    orig_values = df[col_old].values
    good_rows = np.isin(orig_values, list(conversion))
    df = df.iloc[good_rows]
    orig_values = df[col_old].values
    cat_values = np.zeros(len(df), dtype=int)
    for src, dest in conversion.items():
        cat_values[orig_values == src] = dest
    df[col_new] = cat_values
    return df
