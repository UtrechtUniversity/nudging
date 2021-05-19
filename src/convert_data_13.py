#!/usr/bin/env python
import numpy as np
import h5py

data = h5py.File('data/raw/013_czajkowski/data_smieci_3_1.mat','r')
nudge_domain = 3
nudge_type = 2
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

