"""DataSet class for Andreas Lieberoth et al 2018 (https://doi.org/10.1016/j.trf.2018.02.016)"""
import csv

import pandas as pd

from nudging.dataset.base import BaseDataSet


class Lieberoth(BaseDataSet):
    """DataSet class for Andreas Lieberoth et al 2018"""
    # Columns to keep as covariates
    covariates = ["age", "gender"]
    nudge_type = 3
    nudge_domain = 3
    # Control and nudge classes in original data:
    control = "control"
    nudge = "nudge"
    # Gender classes in original data:
    male = "1"
    female = "8"
    # nudge is successfull if outcome increased
    goal = "increase"

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
        # Put data in DataFrame
        df_out = pd.DataFrame(columns=('age', 'gender', 'outcome', 'nudge'))
        with open(self.filename, newline='') as csvfile:
            dataset = csv.dataset(csvfile)
            index = 0
            for row in dataset:
                if row[0] == self.control:
                    nudge = 0
                elif row[0] == self.nudge:
                    nudge = 1
                else:
                    continue
                if row[2] == self.male:
                    gender = 1
                elif row[2] == self.female:
                    gender = 0
                else:
                    gender = " "
                # remove data with undefined gender
                if row[1] == " " or gender == " ":
                    continue
                age = int(float(row[1]))
                outcome = int(row[9])
                df_out.loc[index] = [age, gender, outcome, nudge]
                index += 1

        # this removes unused columns of original data
        return super()._preprocess(df_out)
