#!/usr/bin/env python
"""Convert raw dataset 11 """
from shutil import copyfile

import csv
import numpy as np


NUDGE_DOMAIN = 3
NUDGE_TYPE = [3, 7, 8]

THRESHOLD = 0.


def convert():
    """Convert raw csv"""

    # Raw dataset
    filename = "data/external/011_andreas/Commuter experiment_simple.csv"
    # Copy csv file with raw data
    copyfile(filename, "data/raw/original_11.csv")

    # Get meand and std from control group
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        swipes = {}
        for row in reader:
            if row[0] == 'control':
                if row[2] == "1":
                    # male
                    gender = 1
                elif row[2] == "8":
                    # female
                    gender = 0
                else:
                    gender = ""
                if row[1] == " " or gender == "":
                    continue
                age = int(round(float(row[1])/10, 0))
                if (age, gender) in swipes: 
                    swipes[(age, gender)].append(int(row[9]))
                else:
                    swipes[(age, gender)] = [int(row[9])]

    # New dataset
    new_file = "data/processed/dataset_11.csv"

    # Assume nudge_succes=1 when z_score>=1
    with open(new_file, mode='w') as outfile:
        writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            # writer.writerow(
            #     ["age", "gender", "nudge_domain", "nudge_type", "success", "z_score"])
            for row in reader:
                if row[0] == 'nudge':
                    if row[2] == "1":
                        # male
                        gender = 1
                    elif row[2] == "8":
                        # female
                        gender = 0
                    else:
                        gender = ""
                    age = int(round(float(row[1])/10, 0))
                    if (age, gender) in swipes:
                        mean = np.array(swipes[(age, gender)]).mean() 
                        std = np.array(swipes[(age, gender)]).std()

                        z_score = round((int(row[9]) - mean)/std, 2)
                        if float(row[9]) >= mean + std * THRESHOLD:
                            success = 1
                        else:
                            success = 0
                        writer.writerow([age, gender, NUDGE_DOMAIN, NUDGE_TYPE, success, z_score])


if __name__ == "__main__":
    convert()
