import numpy as np
from reader.pennycook import PennyCook1
import propensity_score as ps

if __name__ == "__main__":

    if True:
        print("")
        print("Andreas: ")
        # Raw dataset
        filename = "data/external/011_andreas/Commuter experiment_simple.csv"
        df = ps.read(filename)
    else:
        print("")
        print("Pennycook: ")
        all_data = PennyCook1("data/external/002_pennycook")
        df1 = all_data._load("data/external/002_pennycook/Pennycook et al._Study 1.csv")
        # Write raw data to csv
        all_data._write_raw(df1)
        df = all_data._preprocess(df1)
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


    if True:
        result["nudge_type"] = "[3, 7, 8]"
        result["nudge_domain"] = 3
        result.to_csv("data/interim/011_andreas_success.csv", index=False)
    else:

        # write data with nudge succes to csv
        all_data._write_interim(result)

