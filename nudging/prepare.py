""" Determine nudge success per subject using propensity score matching
"""
# pylint: disable=eval-used
# pylint: disable=unused-import
import glob
import pandas as pd

from nudging.reader import Balaban, Hotard, PennyCook1, Lieberoth, Simulated # noqa
from nudging.utils import clean_dirs


def combine(infiles, outfile):
    """Combine csv files
    Args:
        infiles (list): list of inputfiles
        outfile (str): name of outputfile
    """
    dataframes = []
    print("\nCombining files:")
    for fin in infiles:
        print(fin)
        dataframes.append(pd.read_csv(fin, encoding="iso-8859-1"))
    data = pd.concat(dataframes)

    # Replace missing values with Nans
    data.replace("", pd.NA, inplace=True)

    # Convert age to decades
    data.age = (data.age/10.).astype(int)
    data.to_csv(outfile, index=False)

    print(f"Combined dataset in {outfile}")


if __name__ == "__main__":

    datasets = {
        # "Simulated": "data/external/simulated/simulated.csv",
        # "Hotard": "data/external/004_hotard/NNYFeeWaiverReplicationData.dta",
        "PennyCook1": "data/external/002_pennycook/Pennycook et al._Study 1.csv",
        "Lieberoth": "data/external/011_lieberoth/Commuter experiment_simple.csv",
        # "Balaban": "data/external/008_balaban/anon1.dta"
    }

    # Cleanup old data
    outdirs = ["data/raw", "data/interim"]
    # Make sure output dirs exist and are empty
    clean_dirs(outdirs)

    # Read and convert each dataset
    for name, path in datasets.items():
        print(f"\ndataset {name}")
        dataset = eval(name + "('" + path + "')")
        # Write raw data to csv
        dataset.write_raw("data/raw/" + name + ".csv")

        # Write interim data to csv
        result = dataset.convert()
        dataset.write_interim(result, "data/interim/" + name + ".csv")

    # combine separate csv files to one
    files = glob.glob('data/interim/[!combined.csv]*')
    combine(files, "data/interim/combined.csv")
