#!/usr/bin/env python
"""This datafile cannot be converted since there is missing info"""
import numpy as np
import h5py

NUDGE_DOMAIN = 3
NUDGE_TYPE = 2


def convert():
    """Read raw data"""
    data = h5py.File('data/raw/013_czajkowski/data_smieci_3_1.mat', 'r')

    # age = np.array(list(data['age'])).flatten()
    # gender =  np.array(list(data['male'])).flatten()
    print(data.keys())
    # INFO is T1-T8 treatment groups
    print(np.array(list(data['INFO'])).flatten())

    print(np.array(list(data['inc'])).flatten())
    print(set(np.array(data['inc'])[0]))

    # dataset = pd.DataFrame({
    #     "age": age,
    #     "gender": gender,
    #     "nudge_domain": nudge_domain,
    #     "nudge_type": nudge_type,
    # })


if __name__ == "__main__":
    convert()
