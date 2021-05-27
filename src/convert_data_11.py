#!/usr/bin/env python
"""Convert raw dataset 11 """
from shutil import copyfile

import csv
import numpy as np


NUDGE_DOMAIN = 3
NUDGE_TYPE = 3


def convert():
    """Convert raw csv"""

    # Raw dataset
    filename = "data/raw/011_andreas/Commuter experiment_simple.csv"
    # Copy csv file with raw data
    copyfile(filename, "data/temp/raw_11.csv")

    # Get meand and std from control group
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        swipes = []
        for row in reader:
            if row[0] == 'control':
                swipes.append(int(row[9]))
        swipes = np.array(swipes)
        mean = swipes.mean()
        std = swipes.std()

    # New dataset
    new_file = "data/processed/dataset_11.csv"

    # Assume nudge_succes=1 when z_score>=1
    with open(new_file, mode='w') as outfile:
        writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            writer.writerow(
                ["age", "gender", "nudge_domain", "nudge_type", "success", "z_score"])
            for row in reader:
                if row[0] == 'nudge':
                    z_score = round((int(row[9]) - mean)/std, 2)
                    if float(row[9]) >= mean + std:
                        success = 1
                    else:
                        success = 0
                    if row[2] == "1":
                        # male
                        gender = 1
                    elif row[2] == "8":
                        # female
                        gender = 0
                    else:
                        gender = ""
                    age = int(round(float(row[1]), 0))
                    writer.writerow([age, gender, NUDGE_DOMAIN, NUDGE_TYPE, success, z_score])


if __name__ == "__main__":
    convert()
