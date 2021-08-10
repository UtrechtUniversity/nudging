import numpy as np
from reader.pennycook import PennyCook1
from reader.andreas import Andreas
import propensity_score as ps

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

        result = ps.match_ps(df_ps)

        interimfile = "data/interim/" + name + ".csv"
        all_data._write_interim(result, interimfile)
