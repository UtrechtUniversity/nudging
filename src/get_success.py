import numpy as np
from reader.pennycook import PennyCook1
from reader.andreas import Andreas
import propensity_score as ps

if __name__ == "__main__":


    datasets = [2, 11]
    if True:
        print("")
        print("Andreas: ")
        all_data = Andreas("data/external/011_andreas/Commuter experiment_simple.csv")
        # Write raw data to csv
        all_data.write_raw("data/raw/011_andres.csv")
        df = all_data.df
        df.reset_index(drop=True, inplace=True)        
    else:
        print("")
        print("Pennycook: ")
        all_data = PennyCook1("data/external/002_pennycook/Pennycook et al._Study 1.csv")
        # Write raw data to csv
        all_data.write_raw("data/raw/002_pennycook1.csv")
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


    if True:
        all_data._write_interim(result, "data/interim/011_andreas_success.csv")
    else:
        # write data with nudge succes to csv
        all_data._write_interim(result, "data/interim/002_pennycook1_success.csv")

