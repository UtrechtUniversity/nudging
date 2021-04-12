#!/usr/bin/env python

import json
from pathlib import Path
from pprint import pprint

for data_dir in Path("packages").glob("*/"):
    print(f"\n---------- {str(data_dir)} ----------")
    try:
        with open(Path(data_dir, "download.json"), "r") as f:
            data = json.load(f)
            pprint(data)
    except FileNotFoundError:
        print("Error finding 'download.json' in ", data_dir)
