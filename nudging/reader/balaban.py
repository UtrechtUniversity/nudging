"""DataSet class for Balaban"""
import numpy as np
import pandas as pd

from nudging.reader.base import BaseDataSet


class Balaban(BaseDataSet):
    """DataSet class for Balaban and Conway 2020"""
    covariates = ["ACThybrid", "gender", "Fyear", "PofC"]
    nudge_type = 8
    nudge_domain = 5

    # nudge is successfull if outcome increased
    goal = "increase"

    def _load(self, file_path):

        return pd.read_stata(file_path)

    def _preprocess(self, data_frame):
        """Read raw csv and convert to standard format
        Args:
            filename (str): name of file to convert
        Returns:
            pandas.DataFrame: containing age, gender, outcome, nudge
        """
        person = np.array(data_frame['anon'])
        person_ids = np.unique(person)

        df_out = data_frame[data_frame['time'] == 3]
        df_out.rename(columns={"MDH": "outcome"}, inplace=True)
        df_out.rename(columns={"Nudge_EA": "nudge"}, inplace=True)
        df_out['gender'] = 1 - data_frame['Female']

        return super()._preprocess(df_out)
