#!/usr/bin/env python

from pathlib import Path
import json

import pandas as pd
import numpy as np
from datetime import date
from pprint import pprint

data = pd.read_excel('Database search_OPEN DATA_Precision Nudging.xlsx')

valid_ids = np.where(np.logical_not(np.isnan(data["ID"])))[0]
df = data.iloc[valid_ids]

for i in range(len(df)):
    data_id = int(df.iloc[i]["ID"])
    authors = df.iloc[i]["Authors"]
    dir_name = f"{data_id:03d}" + "_" + authors.split(",")[0].split(" &")[0].split(" et al")[0].lower()
    dir_name = dir_name.encode('ascii', 'ignore').decode()
    package_dir = Path("data/raw", dir_name)
    download_properties = {
        "id": data_id,
        "download_data": str(date.today()),
        "success": False,
        "title": df.iloc[i]["Title"],
        "year": df.iloc[i]["Year"],
        "authors": authors,
        "link": df.iloc[i]["Open data"],
        "direct_link": False,
    }
    if not package_dir.exists():
        package_dir.mkdir(parents=True)

    download_fp = Path(package_dir, "download.json")
    if download_fp.exists():
        continue

    with open(download_fp, "w") as f:
        json.dump(download_properties, f, indent=4)
#         continue
    print("--------------------------------")
    print(download_fp, download_fp.exists())
