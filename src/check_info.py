#!/usr/bin/env python

import json
from pathlib import Path
from pprint import pprint
from collections import defaultdict
import numpy as np


def read_info(info_fp):
    try:
        with open(info_fp, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error finding 'info.json' in ", data_dir)
        return None
    return data


if __name__ == "__main__":
    all_data_dir = sorted(Path("data/raw").glob("[!.]*/"))
    data = defaultdict(lambda: [])
    for data_dir in all_data_dir:
        info_data = read_info(Path(data_dir, "info.json"))
        data["available"].append(info_data.get("data_availability"))
        data["usable"].append(info_data.get("data_usability", np.nan))
        data["n_participants"].append(info_data.get("number_of_participants", np.nan))
        data["personal_info"].append(info_data.get("personal_info", []))

    data = dict(data)
    data.update({k: np.array(v) for k, v in data.items()
                 if not isinstance(v[0], list)})
    n_data = len(data["usable"])
    n_available = np.nansum(data["available"])
    n_usable = int(np.nansum(data["usable"]))
    participant_list = data["n_participants"][np.where(data["usable"])[0]]
    total_participants = int(np.nansum(participant_list))
    print(f"# Datasets: {n_data}")
    print(f"# Available: {n_available}")
    print(f"# Usable: {n_usable}")
    print(f"# of participants: {total_participants}")
