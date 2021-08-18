import numpy as np
import pandas as pd

from reader.base import BaseDataSet


class PennyCook1(BaseDataSet):
    file_name = "Pennycook et al._Study 1.csv"
    covariates = ["age", "gender", "education", "hhi", "ethnicity", "political_party",
                  "SciKnow", "MMS", "CRT_ACC"]
    nudge_type = 8
    nudge_domain = 5
    control = 2
    nudge = 1
    # TODO: check!
    male = 1 
    female = 2

    def _load(self, fp):
        return pd.read_csv(fp, encoding="iso-8859-1")

    def _preprocess(self, df):
        df.rename(columns={list(df)[0]: "nudge"}, inplace=True)
        df.rename(columns={"Discern": "outcome"}, inplace=True)
        control_idx = np.where(df["nudge"].values == self.control)[0]
        nudge_idx = np.where(df["nudge"].values == self.nudge)[0]
        df.loc[control_idx, "nudge"] = 0
        df.loc[nudge_idx, "nudge"] = 1
        # convert gender to female=0, male=1:
        male_idx = np.where(df["gender"].values == self.male)[0]
        female_idx = np.where(df["gender"].values == self.female)[0]
        df.loc[female_idx, "gender"] = 0
        df.loc[male_idx, "gender"] = 1        
        return super()._preprocess(df)


    def _write_interim(self, df, path):
        df["nudge_type"] = self.nudge_type
        df["nudge_domain"] = self.nudge_domain
        df.to_csv(path, index=False)


class PennyCook2(PennyCook1):
    file_name = "pennycook_2.csv"
    control = 1
    nudge = 2
