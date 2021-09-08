""" Determine nudge success per subject using propensity score matching
"""
# pylint: disable=eval-used
# pylint: disable=unused-import
import glob
import os

import pandas as pd
from reader import Hotard, PennyCook1, Andreas # noqa
import propensity_score as ps


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
        "Andreas": "data/external/011_andreas/Commuter experiment_simple.csv",
    }

    # Cleanup old data
    files = glob.glob("data/raw/*")
    files = files + glob.glob("data/interim/*")
    files = files + glob.glob("data/processed/*")
    for f in files:
        os.remove(f)

    # Read and convert each dataset
    for name, path in datasets.items():

        print("")
        print("dataset ", name)
        data = eval(name + "('" + path + "')")
        # Write raw data to csv
        data.write_raw("data/raw/" + name + ".csv")

        # Convert data to standard format (with columns covariates, nudge, outcome)
        df = data.standard_df
        print(df)
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

        # Cacluate nudge success and write to csv file
        result = ps.match_ps(df_ps)
        data.write_interim(result, "data/interim/" + name + ".csv")

    # combine separate csv files to one
    files = glob.glob('data/interim/*.csv')
    combine(files, "data/processed/combined.csv")
