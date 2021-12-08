"""DataSet class for Andreas Lieberoth et al 2018 (https://doi.org/10.1016/j.trf.2018.02.016)"""
import pandas as pd
import numpy as np

from nudging.dataset.real import RealDataset, Gender, Group


class Lieberoth(RealDataset):
    """DataSet class for Andreas Lieberoth et al 2018"""
    _default_filename = "lieberoth.csv"

    truth = {
        "covariates": ["age", "gender"],
        "nudge_type": 3,
        "nudge_domain": 3,
        # Control and nudge classes in original data:
        "control_value": "control",
        "nudge_value": "nudge",
        # Gender classes in original data:
        "male": "1",
        "female": "8",
        # nudge is successfull if outcome increased
        "goal": "increase",

    }
#     # Columns to keep as covariates
#     covariates = ["age", "gender"]
#     nudge_type = 3
#     nudge_domain = 3
#     # Control and nudge classes in original data:
#     control = "control"
#     nudge = "nudge"
#     # Gender classes in original data:
#     male = "1"
#     female = "8"
#     # nudge is successfull if outcome increased
#     goal = "increase"

    @classmethod
    def _load(cls, file_path):
        """ Read file and return data in dataframe
        Args:
            file_path (str): path of file
        Returns:
            pandas.DataFrame: raw data in dataframe
        """
        return pd.read_csv(file_path, encoding="iso-8859-1")

    @classmethod
    def _preprocess(cls, data_frame):
        """Convert original data to dataframe with standard format
        Args:
            data_frame (pandas.DataFrame): original data
        Returns:
            pandas.DataFrame: dataframe containing covariates, outcome, nudge
        """
        # Put data in DataFrame
#         df_out = pd.DataFrame(columns=('age', 'gender', 'outcome', 'nudge'))
        df = data_frame.copy()
        df = convert_categorical(df, "group", {"control": Group.CONTROL, "nudge": Group.NUDGE},
                                 col_new="nudge")
        df = convert_categorical(df, "gender", {"8": Gender.FEMALE, "1": Gender.MALE})
        df["age"] = pd.to_numeric(df["age"], errors='coerce').round()
        df["outcome"] = df["swtot"]
#         nudge_idx = np.where(data_frame["group"].values == cls.truth["nudge_value"])[0]
#         control_idx = np.where(data_frame["group"].values == cls.truth["control_value"])[0]
#         all_idx = np.append(nudge_idx)
#         remove_columns = set(np.arange(len(df_out)))
#         with open(self.filename, newline='') as csvfile:
#             dataset = csv.reader(csvfile)
#         index = 0
#         for row in dataset:
#             if row[0] == self.control:
#                 nudge = 0
#             elif row[0] == self.nudge:
#                 nudge = 1
#             else:
#                 continue
#             if row[2] == self.male:
#                 gender = 1
#             elif row[2] == self.female:
#                 gender = 0
#             else:
#                 gender = " "
#             # remove data with undefined gender
#             if row[1] == " " or gender == " ":
#                 continue
#             age = int(float(row[1]))
#             outcome = int(row[9])
#             df_out.loc[index] = [age, gender, outcome, nudge]
#             index += 1

        # this removes unused columns of original data
        return super()._preprocess(df)


def convert_categorical(df, col_old, conversion, col_new=None):
    if col_new is None:
        col_new = col_old
    orig_values = df[col_old].values
    good_rows = np.isin(orig_values, list(conversion))
    df = df.iloc[good_rows]
    orig_values = df[col_old].values
    cat_values = np.zeros(len(df), dtype=int)
    for src, dest in conversion.items():
        cat_values[orig_values == src] = dest
    df[col_new] = cat_values
    return df
