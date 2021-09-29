"""DataSet class for Vandenbroele  et al 2021 (https://doi.org/10.1016/j.obhdp.2019.09.004)
The data used here is from study 2 of this paper:
"Effects of product visibility and pairwise presentation on meat and meat substitute sales"
The outcome variable is the percentage of vegetarion purchases
The nudge applied is product visibilitu and/or pairwise presentation
There are 231 participants (undergraduate students), of which 111 male)
"""
import pyreadstat
import numpy as np
import pandas as pd

from nudging.reader.base import BaseDataSet


class Vandenbroele(BaseDataSet):
    """DataSet class for Vandenbroele et al 2020"""
    covariates = ["age", "gender"]
    nudge_type = 4
    nudge_domain = 3

    # nudge is successfull if outcome increased
    goal = "increase"

    def _load(self, file_path):

        input_data, _ = pyreadstat.read_sav(file_path)
        return input_data

    def _preprocess(self, data_frame):
        """Convert original data to standard format
        Args:
            data_frame (pandas.DataFrame): dataframe with original data
        Returns:
            pandas.DataFrame: containing age, gender, outcome, nudge
        """
        person = np.array(data_frame['ParticipantID'])
        visibility = np.array(data_frame['Visibility'])
        pair = np.array(data_frame['PairwisePresentation'])
        gender_array = np.array(data_frame['Sex'])
        age_array = np.array(data_frame['Age'])
        exclude = np.array(data_frame['Participant_VegetarianorVegan'])
        meat = np.array(data_frame['MeatPurchase'])
        veg = np.array(data_frame['VegeterianPurchases'])

        # Exclude vegetarions (veg != 2)
        # index  of control group
        control = np.bitwise_and(np.bitwise_and(visibility == 0, pair == 0), exclude == 2)

        # index of nudge group
        nudge = np.bitwise_and(np.bitwise_or(visibility == 1, pair == 1), exclude == 2)

        person_ids = np.unique(person)
        data = []
        veg1 = 0
        meat1 = 0
        for person_id in person_ids:
            person_index = person == person_id
            index_nudge = np.bitwise_and(nudge, person_index)
            index_control = np.bitwise_and(control, person_index)
            # control
            meat_purchases = np.nansum(meat[index_control])
            veg_purchases = np.nansum(veg[index_control])
            veg1 = veg1 + veg_purchases
            meat1 = meat1 + meat_purchases
            ratio_control = veg_purchases/(meat_purchases + veg_purchases)

            # nudge
            meat_purchases = np.nansum(meat[index_nudge])
            veg_purchases = np.nansum(veg[index_nudge])
            ratio_nudge = veg_purchases/(meat_purchases + veg_purchases)

            age_list = age_array[person_index][np.isfinite(age_array[person_index])]
            if len(age_list) > 0:
                age = int(
                    round(age_array[person_index][np.isfinite(age_array[person_index])][0], 0))
            else:
                continue
            gender = gender_array[person_index][np.isfinite(gender_array[person_index])][0]
            if gender == 1:
                gender = 0
            elif gender == 0:
                gender = 1
            else:
                gender = ""

            data.append([age, gender, ratio_nudge, 1])
            data.append([age, gender, ratio_control, 0])

        df_out = pd.DataFrame(data, columns=["age", "gender", "outcome", "nudge"])

        return super()._preprocess(df_out)
