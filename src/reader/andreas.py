import numpy as np
import pandas as pd
import csv

from reader.base import BaseDataSet


class Andreas(BaseDataSet):
    covariates = ["age", "gender"]
    nudge_type = "[3, 7, 8]"
    nudge_domain = 3
    control = "control"
    nudge = "nudge"
    male = 1
    female = 8

    def _load(self, file_path):
        return pd.read_csv(file_path, encoding="iso-8859-1")


    def _preprocess(self, df_dummy):
        """Read raw csv and convert to standard format
        Args:
            filename (str): name of file to convert
        Returns:
            pandas.DataFrame: containing age, gender, outcome, nudge
        """
        # Put data in DataFrame
        df = pd.DataFrame(columns=('age', 'gender', 'outcome', 'nudge'))
        with open(self.filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            swipes = {}
            index = 0
            for i, row in enumerate(reader):
                if row[0] == 'control':
                    nudge = 0
                elif row[0] == 'nudge':
                    nudge = 1
                else:
                    continue
                if row[2] == "1":
                    # male
                    gender = 1
                elif row[2] == "8":
                    # female
                    gender = 0
                else:
                    gender = " "
                if row[1] == " " or gender == " ":
                    continue
                age = int(round(float(row[1])/10, 0))
                outcome = int(row[9])
                df.loc[index] = [age, gender, outcome, nudge]
                index += 1

        df = df.apply(pd.to_numeric)

        return df



    def _write_interim(self, df, path):
        df["nudge_type"] = self.nudge_type
        df["nudge_domain"] = self.nudge_domain
        df.to_csv(path, index=False)