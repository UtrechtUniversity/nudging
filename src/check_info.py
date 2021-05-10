"""Check info.json files and summarize"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np


def read_info(info_fp):
    """Read and return content of info.json """
    try:
        with open(info_fp, "r") as json_file:
            result = json.load(json_file)
    except FileNotFoundError:
        print("Error finding ", info_fp)
        return None
    return result


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
    n_usable = np.nansum(int(data["usable"]))
    participant_list = data["n_participants"][np.where(data["usable"])[0]]
    total_participants = np.nansum(int(participant_list))
    print(f"# Datasets: {n_data}")
    print(f"# Available: {n_available}")
    print(f"# Usable: {n_usable}")
    print(f"# of participants: {total_participants}")
