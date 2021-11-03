
"""DataSet class for Hotard et al 2019 (https://doi.org/10.1038/s41562-019-0572-z)"""
import numpy as np
import pandas as pd

from nudging.dataset.base import BaseDataSet


class Hotard(BaseDataSet):
    """DataSet class for Hotard et al 2019"""
    covariates = ["age_Q", "gender", "mtbany", "educ_HS", "marital_single",
                  "hhinc_cap_Q", "hhsize_Q",  "College",
                  "lang_Eng", "yearsinus"]
    nudge_type = 8
    nudge_domain = 5

    male = "Male"
    female = "Female"
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
        # Remove missing values
        df_out = data_frame.replace("", np.nan, inplace=False)
        df_out = df_out.dropna(subset=["submitted", "anyfeewaivernote"]).reset_index()

        # df_out = pd.DataFrame(columns=('age', 'gender', 'outcome', 'nudge'))
        df_out["outcome"] = (df_out["submitted"] == "Submitted").astype('int')
        df_out["nudge"] = (df_out["anyfeewaivernote"] == 1).astype('int')

        # Age is given in quartiles, hence this data cannot be used to combine with other studies
        # df_out.rename(columns={"age_Q": "age"}, inplace=True)

        # convert gender to female=0, male=1:
        male_idx = np.where(df_out["gender_f"].values == self.male)[0]
        female_idx = np.where(df_out["gender_f"].values == self.female)[0]
        df_out.loc[female_idx, "gender"] = 0
        df_out.loc[male_idx, "gender"] = 1
        return super()._preprocess(df_out)
