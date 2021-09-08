"""DataSet class for Pennycook et al 2020
(https://journals.sagepub.com/doi/10.1177/0956797620939054)"""
import numpy as np
import pandas as pd

from reader.base import BaseDataSet


class PennyCook1(BaseDataSet):
    """DataSet class for Pennycook et al 2020 """
    # Columns to keep as covariates
    covariates = ["age", "gender", "education", "hhi", "ethnicity", "political_party",
                  "SciKnow", "MMS", "CRT_ACC"]
    nudge_type = 8
    nudge_domain = 5
    # Control and nudge classes in original data:
    control = 2
    nudge = 1
    # Gender classes in original data:
    male = 1
    female = 2

    def _load(self, file_path):
        """ Read file and return data in dataframe
        Args:
            file_path (str): path of file
        Returns:
            pandas.DataFrame: raw data in dataframe
        """
        return pd.read_csv(file_path, encoding="iso-8859-1")

    def _preprocess(self, data_frame):
        """Convert original data to dataframe with standard format
        Args:
            data_frame (pandas.DataFrame): original data
        Returns:
            pandas.DataFrame: dataframe containing covariates, outcome, nudge
        """
        df_out = data_frame.copy()
        df_out.rename(columns={list(df_out)[0]: "nudge"}, inplace=True)
        df_out.rename(columns={"Discern": "outcome"}, inplace=True)
        control_idx = np.where(df_out["nudge"].values == self.control)[0]
        nudge_idx = np.where(df_out["nudge"].values == self.nudge)[0]
        df_out.loc[control_idx, "nudge"] = 0
        df_out.loc[nudge_idx, "nudge"] = 1
        # convert gender to female=0, male=1:
        male_idx = np.where(df_out["gender"].values == self.male)[0]
        female_idx = np.where(df_out["gender"].values == self.female)[0]
        df_out.loc[female_idx, "gender"] = 0
        df_out.loc[male_idx, "gender"] = 1
        return super()._preprocess(df_out)
