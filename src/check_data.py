#!/usr/bin/env python
"""Check data_info.json files and summarize"""
from collections import defaultdict
from datetime import date
import json
from pathlib import Path
# from pprint import pprint

import numpy as np
import pandas as pd


def create_data_dirs(filename):
    """ Read excel file with open dataset and create folders using 1st author name"""
    data = pd.read_excel(filename)

    valid_ids = np.where(np.logical_not(np.isnan(data["ID"])))[0]
    data_frame = data.iloc[valid_ids]

    for i in range(len(data_frame)):
        data_id = int(data_frame.iloc[i]["ID"])
        authors = data_frame.iloc[i]["Authors"]
        dir_name = f"{data_id:03d}" + "_" + \
            authors.split(",")[0].split(" &")[0].split(" et al")[0].lower()
        dir_name = dir_name.encode('ascii', 'ignore').decode()
        package_dir = Path("data/external", dir_name)
        download_properties = {
            "id": data_id,
            "download_date": str(date.today()),
            "data_downloaded": False,
            "title": data_frame.iloc[i]["Title"],
            "year": data_frame.iloc[i]["Year"],
            "authors": authors,
            "link": data_frame.iloc[i]["Open data"],
            "direct_link": False,
        }
        if not package_dir.exists():
            package_dir.mkdir(parents=True)

        download_fp = Path(package_dir, "data_info.json")
        if download_fp.exists():
            continue

        with open(download_fp, "w") as file_:
            json.dump(download_properties, file_, indent=4)
        print("--------------------------------")
        print(download_fp, download_fp.exists())


def read_info(info_fp):
    """Read and return content of data_info.json
    Args:
        info_fp (str): path to json file to check
    Returns:
        dict: contents of data_info.json
    """
    try:
        with open(info_fp, "r") as json_file:
            result = json.load(json_file)
            # pprint(result)
    except FileNotFoundError:
        print("Error finding ", info_fp)
        return None
    return result


def summarize_info():
    """Summarize and print info on datasets
    """
    all_data_dir = sorted(Path("data/external").glob("[!.]*/"))
    data = defaultdict(lambda: [])
    no_info = []
    for data_dir in all_data_dir:

        info_data = read_info(Path(data_dir, "data_info.json"))
        if not info_data:
            no_info.append(data_dir)
            continue
        data["downloaded"].append(info_data.get("data_downloaded"))
        data["available"].append(info_data.get("data_availability"))
        data["usable"].append(info_data.get("data_usability", np.nan))
        data["n_participants"].append(info_data.get("number_of_participants", np.nan))
        data["personal_info"].append(info_data.get("personal_info", []))

    data = dict(data)
    data.update({k: np.array(v) for k, v in data.items()
                 if not isinstance(v[0], list)})
    n_downloaded = np.nansum(data["downloaded"])
    n_available = np.nansum(data["available"])
    n_usable = np.nansum(data["usable"])
    participant_list = data["n_participants"][np.where(data["usable"])[0]]
    total_participants = np.nansum(participant_list)
    print(f"# Folders: {len(all_data_dir)}")
    print(f"# Downloaded: {n_downloaded}")
    print(f"# Available: {n_available}")
    print(f"# Usable: {int(n_usable)}")
    print(f"# of participants: {int(total_participants)}")


if __name__ == "__main__":

    # Create folders for all dataset entries in excel sheet
    create_data_dirs('data/Database search_OPEN DATA_Precision Nudging.xlsx')

    # print summary to screen
    summarize_info()
