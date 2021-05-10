"""Check download.json files and summarize"""

import json
from pathlib import Path
from pprint import pprint

for data_dir in sorted(Path("data/raw").glob("[!.]*/")):
    try:
        download_fp = Path(data_dir, "download.json")
        with open(download_fp, "r") as f:
            data = json.load(f)
            if data["success"]:
                print(f"\n---------- {str(data_dir.name)} ----------")
                readme_fp = Path(data_dir, "README.md")
                with open(readme_fp, "r") as f:
                    print(f.read())
#                 print(data["success"], download_fp)
    except FileNotFoundError:
        print("Error finding 'download.json' in ", data_dir)
    try:
        with open(Path(data_dir, "info.json"), "r") as f:
            data = json.load(f)
            pprint(data)
    except FileNotFoundError:
        print("Error finding 'info.json' in ", data_dir)
