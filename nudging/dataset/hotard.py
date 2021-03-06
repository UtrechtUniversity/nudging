
"""DataSet class for Hotard et al 2019 (https://doi.org/10.1038/s41562-019-0572-z)"""
import numpy as np

from nudging.dataset.real import RealDataset


class Hotard(RealDataset):
    """DataSet class for Hotard et al 2019"""
    _default_filename = "hotard.dta"
    truth = {
        "covariates": ["age_Q", "gender", "mtbany", "educ_HS", "marital_single",
                       "hhinc_cap_Q", "hhsize_Q",  "College",
                       "lang_Eng", "yearsinus"],
        "nudge_type": 8,
        "nudge_domain": 5,
        "male": "Male",
        "female": "Female",
        # nudge is successfull if outcome increased
        "goal": "increase",
    }

#     @classmethod
#     def _load(cls, file_path):
#         return pd.read_stata(file_path)

    @classmethod
    def _preprocess(cls, data_frame):
        """Read raw csv for Hotard data and convert to standard format
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
        male_idx = np.where(df_out["gender_f"].values == cls.truth["male"])[0]
        female_idx = np.where(df_out["gender_f"].values == cls.truth["female"])[0]
        df_out.loc[female_idx, "gender"] = 0
        df_out.loc[male_idx, "gender"] = 1
        return super()._preprocess(df_out)
