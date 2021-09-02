import numpy as np
import pandas as pd

from reader.base import BaseDataSet


class Hotard(BaseDataSet):

    file_name = "NNYFeeWaiverReplicationData.dta"
    # covariates = ["mtbany", "educ_HS", "marital_single", "age_Q",
    #               "hhinc_cap_Q", "hhsize_Q", "gender", "College",
    #               "lang_Eng", "yearsinus"]
    covariates = ["mtbany", "educ_HS", "marital_single",
                  "hhinc_cap_Q", "hhsize_Q",  "College",
                  "lang_Eng", "yearsinus"]
    nudge_type = 8
    nudge_domain = 5

    male = "Male"
    female = "Female"


    def _load(self, fp):
        print(pd.read_stata(fp))
        return pd.read_stata(fp)

    def _preprocess(self, df_dummy):
        """Read raw csv and convert to standard format
        Args:
            filename (str): name of file to convert
        Returns:
            pandas.DataFrame: containing age, gender, outcome, nudge
        """
        # Put data in DataFrame (remove missing values)
        df_temp = df_dummy.replace("", np.nan, inplace=False)
        df_temp = df_temp.dropna(subset=["submitted", "anyfeewaivernote"]).reset_index()

        df = pd.DataFrame(columns=('age', 'gender', 'outcome', 'nudge'))
        df["outcome"] = (df_temp["submitted"] == "Submitted").astype('int')
        df["nudge"] = (df_temp["anyfeewaivernote"] == 1).astype('int')
        df["age"] = df_temp["age_Q"]
        for feat in self.covariates:            
            df[feat] = df_temp[feat]

        print(df)
        # convert gender to female=0, male=1:
        male_idx = np.where(df_temp["gender_f"].values == self.male)[0]
        female_idx = np.where(df_temp["gender_f"].values == self.female)[0]
        df.loc[female_idx, "gender"] = 0
        df.loc[male_idx, "gender"] = 1  
        return df

    def write_interim(self, df, path):
        df["nudge_type"] = self.nudge_type
        df["nudge_domain"] = self.nudge_domain
        df.to_csv(path, index=False)