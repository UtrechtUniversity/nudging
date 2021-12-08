"""DataSet class for Balaban"""
import pandas as pd

from nudging.dataset.real import RealDataset


class Balaban(RealDataset):
    """DataSet class for Balaban and Conway 2020"""
    _default_filename = "balaban.dta"
    truth = {
        "covariates": ["ACThybrid", "gender", "Fyear", "PofC"],
        "nudge_type": 8,
        "nudge_domain": 5,
        "goal": "increase"
    }
#     nudge is successfull if outcome increased
#     goal = "increase"

    @classmethod
    def _load(cls, file_path, _encoding=None):
        return pd.read_stata(file_path)

    @classmethod
    def _preprocess(cls, data_frame):
        """Read raw csv and convert to standard format
        Args:
            filename (str): name of file to convert
        Returns:
            pandas.DataFrame: containing age, gender, outcome, nudge
        """
        # person = np.array(data_frame['anon'])
        # person_ids = np.unique(person)
        df_out = data_frame[data_frame['time'] == 3]
        df_out.rename(columns={"MDH": "outcome"}, inplace=True)
        df_out.rename(columns={"Nudge_EA": "nudge"}, inplace=True)
        df_out['gender'] = 1 - data_frame['Female']

        return super()._preprocess(df_out)
