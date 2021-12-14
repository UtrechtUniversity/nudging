"""DataSet class for Pennycook et al 2020,
"Fighting COVID-19 Misinformation on  Social Media:
Experimental Evidence for  a Scalable Accuracy-Nudge Intervention"
(https://journals.sagepub.com/doi/10.1177/0956797620939054)
Study 1:
- nudge group was asked about accuracy of headline,
- control group was asked if they would share a headline
- outcome is discernment defined as the difference in accuracy judgments
(or sharing intentions) between true and false headlines
Study 2:
- nudge group rated the accuracy of a single headline before sharing task,
- control group started news-sharing task without accuracy rating question.
- outcome is discernment defined as the difference in accuracy judgments
(or sharing intentions) between true and false headlines
"""
import numpy as np

from nudging.dataset.real import RealDataset


class BasePennyCook(RealDataset):
    """Base class for penny cook datasets"""
    @classmethod
    def _preprocess(cls, data_frame):
        df_out = data_frame.copy()
        df_out.rename(columns={list(df_out)[0]: "nudge"}, inplace=True)
        df_out.rename(columns={"Discern": "outcome"}, inplace=True)
        control_idx = np.where(df_out["nudge"].values == cls.truth["control_value"])[0]
        nudge_idx = np.where(df_out["nudge"].values == cls.truth["nudge_value"])[0]
        df_out.loc[control_idx, "nudge"] = 0
        df_out.loc[nudge_idx, "nudge"] = 1
        # convert gender to female=0, male=1:
        male_idx = np.where(df_out["gender"].values == cls.truth["male"])[0]
        female_idx = np.where(df_out["gender"].values == cls.truth["female"])[0]
        df_out.loc[female_idx, "gender"] = 0
        df_out.loc[male_idx, "gender"] = 1
        return super()._preprocess(df_out)

    @classmethod
    def _load(cls, file_path, encoding="iso-8859-1"):
        return super()._load(file_path, encoding=encoding)


class Pennycook1(BasePennyCook):
    """DataSet class for Pennycook et al 2020, study 1"""
    _default_filename = "pennycook_1.csv"
    truth = {
        "covariates": ["age", "gender", "hhi", "education", "ethnicity",
                       "political_party", "SciKnow", "MMS", "CRT_ACC"],
        "nudge_type": 8,
        "nudge_domain": 5,
        "control_value": 2,
        "nudge_value": 1,
        "male": 1,
        "female": 2,
        "goal": "increase",
    }


class Pennycook2(BasePennyCook):
    """DataSet class for Pennycook et al 2020, study 2"""
    _default_filename = "pennycook_2.csv"
    truth = {
        "covariates":  ["age", "gender", "region", "education", "ethnicity",
                        "political_party", "SciKnow", "MMS", "CRT_ACC"],
        "nudge_type": 8,
        "nudge_domain": 5,
        "control_value": 1,
        "nudge_value": 2,
        "male": 1,
        "female": 2,
        "goal": "increase",
    }
