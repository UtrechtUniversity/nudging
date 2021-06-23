#!/usr/bin/env python
"""Convert raw dataset 16 """
import csv
import pyreadstat
import numpy as np

NUDGE_DOMAIN = 3
NUDGE_TYPE = [4]

THRESHOLD = 0.

def convert():
    """Convert raw sav"""

    input_data, _ = pyreadstat.read_sav('data/external/016_vandenbroele/S2_OpenAccess.sav')
    # write csv file with raw data
    input_data.to_csv("data/raw/original_16.csv")

    person = np.array(input_data['ParticipantID'])
    visibility = np.array(input_data['Visibility'])
    pair = np.array(input_data['PairwisePresentation'])
    gender_array = np.array(input_data['Sex'])
    age_array = np.array(input_data['Age'])
    exclude = np.array(input_data['Participant_VegetarianorVegan'])
    meat = np.array(input_data['MeatPurchase'])
    veg = np.array(input_data['VegeterianPurchases'])

    # Exlude vegetarions (veg != 2)
    # index  of control group
    control = np.bitwise_and(np.bitwise_and(visibility == 0, pair == 0), exclude == 2)

    # index of nudge group
    nudge = np.bitwise_and(np.bitwise_or(visibility == 1, pair == 1), exclude == 2)

    person_ids = np.unique(person)
    ratios = {}
    # Get mean and std for control group
    for person_id in person_ids:
        person_index = person == person_id
        index = np.bitwise_and(nudge, person_index)
        # individual control
        meat_purchases_control = np.nansum(meat[index])
        veg_purchases_control = np.nansum(veg[index])
        if (meat_purchases_control + veg_purchases_control) == 0:
            continue

        ratio_control = veg_purchases_control/(meat_purchases_control + veg_purchases_control)
        age_list = age_array[index][np.isfinite(age_array[index])]
        if len(age_list) > 0:
            age = int(round(age_array[index][np.isfinite(age_array[index])][0]/10, 0))
        else:
            continue
        gender = gender_array[index][np.isfinite(gender_array[index])][0]
        if gender == 1:
            gender = 0
        elif gender == 0:
            gender = 1
        else:
            gender = ""

        if (age, gender) in ratios: 
            ratios[(age, gender)].append(ratio_control)
        else:
            ratios[(age, gender)] = [ratio_control]
    
    # New dataset
    new_file = "data/processed/dataset_16.csv"
    # Assume nudge_succes=1 when z_score>=1
    with open(new_file, mode='w') as outfile:
        writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # writer.writerow(
        #     ["age", "gender", "nudge_domain", "nudge_type", "success", "z_score"])
        # Determine nudge success
        for person_id in person_ids:
            person_index = person == person_id
            index = np.bitwise_and(nudge, person_index)
            meat_purchases = np.nansum(meat[index])
            veg_purchases = np.nansum(veg[index])
            if (meat_purchases + veg_purchases) == 0:
                continue
            ratio = veg_purchases/(meat_purchases + veg_purchases)
            age = int(round(age_array[index][np.isfinite(age_array[index])][0]/10, 0))
            gender = gender_array[index][np.isfinite(gender_array[index])][0]
            if gender == 1:
                gender = 0
            elif gender == 0:
                gender = 1
            else:
                gender = ""

            if (age, gender) in ratios:
                mean = np.array(ratios[(age, gender)]).mean() 
                std = np.array(ratios[(age, gender)]).std()
                z_score = (ratio - mean) / std
                success = 1 if z_score >= THRESHOLD else 0
                z_score = round(z_score, 2)

            writer.writerow([age, gender, NUDGE_DOMAIN, NUDGE_TYPE, success, z_score])


if __name__ == "__main__":
    convert()
