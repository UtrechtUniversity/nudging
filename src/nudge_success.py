import glob
import numpy as np
import pandas as pd
from reader.pennycook import PennyCook1
from reader.andreas import Andreas
import propensity_score as ps


def combine(infiles, outfile):
    """Combine csv files"""

    dataframes = []
    print("Combining files:\n")
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
        "PennyCook1": "data/external/002_pennycook/Pennycook et al._Study 1.csv",
        "Andreas": "data/external/011_andreas/Commuter experiment_simple.csv"
    }

    for name, path in datasets.items():

        print("")
        print("dataset ", name)
        all_data = eval(name + "('"+ path +"')") #Andreas("data/external/011_andreas/Commuter experiment_simple.csv")
        # Write raw data to csv
        rawfile = "data/raw/" + name + ".csv"
        all_data.write_raw(rawfile)
        df = all_data.df
        df.reset_index(drop=True, inplace=True)
        # Apply OLS regression and print info
        # print(smf.ols("outcome ~ nudge", data=df.apply(pd.to_numeric)).fit().summary().tables[1])

        # calculate propensity score
        df_ps = ps.get_pscore(df)
        # ps.check_weights(df_ps)    
        ps.plot_confounding_evidence(df_ps)
        ps.plot_overlap(df_ps)

        ps.get_ate(df_ps)    
        # ps.get_psw_ate(df_ps)
        # get_psm_ate(ps)

        # Cacluate nudge success and write to csv file
        result = ps.match_ps(df_ps)
        interimfile = "data/interim/" + name + ".csv"
        all_data._write_interim(result, interimfile)

    # combine separate csv files to one
    infiles = glob.glob('data/interim/*.csv')
    outfile = "data/processed/combined.csv"
    combine(infiles, outfile)
