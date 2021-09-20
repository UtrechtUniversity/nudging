""" Determine nudge success per subject using propensity score matching
"""
# pylint: disable=eval-used
# pylint: disable=unused-import
import glob

import pandas as pd

import propensity_score as ps
from reader import Hotard, PennyCook1, Lieberoth # noqa
from utils import clean_dirs


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
    dataset = pd.concat(dataframes)

    # Replace missing values with Nans
    dataset.replace("", pd.NA, inplace=True)

    # Convert age to decades
    dataset.age = (dataset.age/10).astype(int)
    dataset.to_csv(outfile, index=False)


if __name__ == "__main__":

    datasets = {
        # "Hotard": "data/external/004_hotard/NNYFeeWaiverReplicationData.dta",
        "PennyCook1": "data/external/002_pennycook/Pennycook et al._Study 1.csv",
        "Lieberoth": "data/external/011_lieberoth/Commuter experiment_simple.csv",
    }

    # Cleanup old data
    outdirs = ["data/raw", "data/interim"]
    # Make sure output dirs exist and are empty
    clean_dirs(outdirs)

    # Read and convert each dataset
    for name, path in datasets.items():

        print("")
        print("dataset ", name)
        data = eval(name + "('" + path + "')")
        # Write raw data to csv
        data.write_raw("data/raw/" + name + ".csv")

        # Convert data to standard format (with columns covariates, nudge, outcome)
        df = data.standard_df
        df.reset_index(drop=True, inplace=True)

        # Apply OLS regression and print info
        # print(smf.ols("outcome ~ nudge", data=df.apply(pd.to_numeric)).fit().summary().tables[1])

        # calculate propensity score
        df_ps = ps.get_pscore(df)

        # Check is treatment and control groups are well-balanced
        # ps.check_weights(df_ps)

        # Plots
        # ps.plot_confounding_evidence(df_ps)
        # ps.plot_overlap(df_ps)

        # Average Treatment Effect (ATE)
        ps.get_ate(df_ps)

        # propensity score weigthed ATE
        # ps.get_psw_ate(df_ps)

        # propensity score matched ATE with CausalModel
        # ps.get_psm_ate(df_ps)

        # Cacculate nudge success and write to csv file
        result = ps.match_ps(df_ps)
        result = data.get_success(result)
        data.write_interim(result, "data/interim/" + name + ".csv")

    # combine separate csv files to one
    files = glob.glob('data/interim/[!combined.csv]*')
    combine(files, "data/interim/combined.csv")
