"""DataSet class for Milkman et al 2021: Megastudies improve the impact of applied
behavioural science
Paper: https://www.nature.com/articles/s41586-021-04128-4
Data: https://osf.io/9av87/?view_only=8bb9282111c24f81a19c2237e7d7eba3
"""
from pathlib import Path
import numpy as np
import pandas as pd

from nudging.dataset.real import RealDataset, Gender, Group, convert_categorical


def get_change(data_frame):
    """ Comnpute change in visits"""
    change = data_frame['visits'][data_frame['phase'] == 'during'].to_numpy() - \
        data_frame['visits'][data_frame['phase'] == 'pre'].to_numpy()

    result = np.nan
    if change:
        result = change[0]

    return result


class Milkman(RealDataset):
    """DataSet class for Milkman et al 2021"""

    intervention = "Higher Incentives b"
    _default_filename = "pptdata.csv"
    truth = {
        "goal": "increase",
        "covariates": ["age", "gender"],  # "customer_state"],
        "nudge_type": None,
        "nudge_domain": 3,
    }

    @classmethod
    def _load(cls, file_path, encoding="iso-8859-1"):
        return super()._load(file_path, encoding=encoding)

    @classmethod
    def from_file(  # pylint: disable=arguments-differ
            cls, file_path, nudge_type=None, intervention=None):
        """Create dataset from file"""
        if intervention:
            cls.intervention = intervention
        if nudge_type:
            cls.truth["nudge_type"] = nudge_type
        if Path(file_path).is_dir():
            file_path = Path(file_path, cls._default_filename)
        raw_df = cls._load(file_path)
        standard_df = cls._preprocess(raw_df)

        return cls(standard_df, raw_df, file_path)

    @classmethod
    def _preprocess(cls, data_frame):
        """Convert original data to standard format
        Args:
            data_frame (pandas.DataFrame): dataframe with original data
        Returns:
            pandas.DataFrame: containing age, gender, outcome, nudge
        """
        data = data_frame[
            ['participant_id', 'week', 'visits', 'age',
                'customer_state', 'gender', 'exp_condition']].copy()
        data.loc[:, 'phase'] = "post"
        data.loc[data.week < 5, 'phase'] = 'during'
        data.loc[data.week < 0, 'phase'] = 'pre'

        data_new = data.groupby(['participant_id', 'phase'], as_index=False).mean()
        data_new = data_new.groupby(['participant_id'], as_index=False).apply(get_change)
        data_new.columns = ['participant_id', 'outcome']
        data_unique = data.drop(columns=['week', 'visits', 'phase']).drop_duplicates()

        df = pd.merge(data_unique, data_new, on="participant_id")
        df = convert_categorical(
            df,
            "exp_condition",
            {"Placebo Control": Group.CONTROL, cls.intervention: Group.NUDGE},
            col_new="nudge")
        df = convert_categorical(df, "gender", {"F": Gender.FEMALE, "M": Gender.MALE})

        return super()._preprocess(df)
