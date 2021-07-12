# Reference: https://matheusfacure.github.io/python-causality-handbook/11-Propensity-Score.html#

from shutil import copyfile

import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
from causalinference import CausalModel


def read(filename):
    """Read raw csv and convert to standard format
    Args:
        filename (str): name of file to convert
    Returns:
        pandas.DataFrame: containing age, gender, outcome, nudge
    """

    # Copy csv file with raw data
    copyfile(filename, "data/raw/original_11.csv")

    # Put data in DataFrame
    df = pd.DataFrame(columns=('age', 'gender', 'outcome', 'nudge'))
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        swipes = {}
        index = 0
        for i, row in enumerate(reader):
            if row[0] == 'control':
                nudge = 0
            elif row[0] == 'nudge':
                nudge = 1
            else:
                continue
            if row[2] == "1":
                # male
                gender = 1
            elif row[2] == "8":
                # female
                gender = 0
            else:
                gender = " "
            if row[1] == " " or gender == " ":
                continue
            age = int(round(float(row[1])/10, 0))
            outcome = int(row[9])
            df.loc[index] = [age, gender, outcome, nudge]
            index += 1

    df = df.apply(pd.to_numeric)

    return df


def get_pscore(df):
    """Calculate propensity score with logistic regression
    Args:
        df (pandas.DataFrame):
    
    Returns:
        pandas.DataFrame: dataframe with propensity score
    """
    T = 'nudge'
    Y = 'outcome'
    X = df.columns.drop([T, Y])  
    ps_model = LogisticRegression().fit(df[X].to_numpy().astype('int'), df[T].to_numpy().astype('int'))

    data_ps = df.assign(pscore=ps_model.predict_proba(df[X])[:, 1])

    return data_ps


def check_weights(data_ps):
    """Check if sum of propensity score weights match sample size
    Args:
        data_ps (pandas.DataFrame): dataframe with propesnity score
    """
    weight_t = 1/data_ps.query("nudge==1")["pscore"]
    weight_nt = 1/(1-data_ps.query("nudge==0")["pscore"])
    print("Original Sample Size", data_ps.shape[0])
    print("Treated Population Sample Size", sum(weight_t))
    print("Untreated Population Sample Size", sum(weight_nt))


def plot_confounding_evidence(data_ps):
    """Use the propensity score to find evidence of confounding
    Args:
        data_ps (pandas.DataFrame): dataframe with propensity score
    """
    sns.boxplot(x="age", y="pscore", data=data_ps)
    plt.title("Confounding Evidence")
    plt.show()
    sns.boxplot(x="gender", y="pscore", data=data_ps)
    plt.title("Confounding Evidence")
    plt.show()


def plot_overlap(data_ps):
    """ check that there is overlap between the treated and untreated population
    Args:
        data_ps (pandas.DataFrame): dataframe with propensity score    
    """
    sns.distplot(data_ps.query("nudge==0")["pscore"], kde=False, label="Non Nudged")
    sns.distplot(data_ps.query("nudge==1")["pscore"], kde=False, label="Nudged")
    plt.title("Positivity Check")
    plt.legend()
    plt.show()


def get_ate(data_ps):
    """Get ATE without bias correction
    Args:
        data_ps (pandas.DataFrame): dataframe with propensity score    
    """
    result = data_ps.groupby("nudge")["outcome"].mean()
    print("Not bias corrected:")
    print("Y0:", result[0])
    print("Y1:", result[1])
    print("ATE", result[1] - result[0])   


def get_psw_ate(data_ps):
    """Get propensity score weigthed ATE
    Args:
        data_ps (pandas.DataFrame): dataframe with propensity score
    """
    weight_t = 1/data_ps.query("nudge==1")["pscore"]
    weight_nt = 1/(1-data_ps.query("nudge==0")["pscore"])
    weight = ((data_ps["nudge"]-data_ps["pscore"]) /
            (data_ps["pscore"]*(1-data_ps["pscore"])))

    y1 = sum(data_ps.query("nudge==1")["outcome"]*weight_t) / len(data_ps)
    y0 = sum(data_ps.query("nudge==0")["outcome"]*weight_nt) / len(data_ps)

    ate = np.mean(weight * data_ps["outcome"])
    print("Propensity score weighted:")
    print("Y0:", y0)
    print("Y1:", y1)
    print("ATE", np.mean(weight * data_ps["outcome"]))   

def get_psm_ate(data_ps):
    """Get propensity score matched ATE using CausalModel
    Args:
        data_ps (pandas.DataFrame): dataframe with propensity score
    """
    cm = CausalModel(
        Y=data_ps["outcome"].values, 
        D=data_ps["nudge"].values, 
        X=data_ps[["pscore"]].values
    )

    cm.est_via_ols()
    cm.est_via_matching(matches=1, bias_adj=True)
    print(cm.estimates)

def perfom_matching(row, indexes, df_data):
    """Match participant from treatment group with particapant from control group
    Args:
        row (pandas.Series): row of dataframe to perform matching on
        indexes (numpy.ndarray): array of indexes of nearest neighbours
        df_data (pandas.DataFrame): dataframe with all participants
    Returns:
        integer: index of matched partcipant from control group
    """
    current_index = int(row['index']) # Obtain value from index-named column, not the actual DF index.
    prop_score_logit = row['pscore']
    for idx in indexes[current_index,:]:
        if (current_index != idx) and (row.nudge == 1) and (df_data.loc[idx].nudge == 0):
            return int(idx)

def obtain_match_details(row, all_data, attribute):
    """ Get details from matched participant of control group
    Args:
        row (pandas.Series): row of dataframe with particpant of treatment group
        all_data (pandas.dataframe): dataframe with all data (from treatment and control)
        attribute (str): name of column to get details for
    Returns:
        value of attribute of matched participant in control group
    """
    return all_data.loc[row.matched_element][attribute]         

def match_ps(data_ps):
    """Match participants in treatment group to control group by propensity score 
    Args:
        data_ps (pandas.DataFrame): dataframe with propensity score
    Returns:
        pandas.DataFrame: 
    """    

    knn = NearestNeighbors(n_neighbors=10 , p = 2)
    knn.fit(data_ps[['pscore']].to_numpy())

    # Common support distances and indexes
    distances, indexes = knn.kneighbors(
        data_ps[['pscore']].to_numpy(),
        n_neighbors=10
    )

    # print('For item 0, the 4 closest distances are (first item is self):')
    # for ds in distances[0,0:4]:
    #     print('Element distance: {:4f}'.format(ds))
    # print('...')
    # print('For item 0, the 4 closest indexes are (first item is self):')
    # for idx in indexes[0,0:4]:
    #     print('Element index: {}'.format(idx))
    # print('...')

    # Add index of match to dataframe
    data_ps['matched_element'] = data_ps.reset_index().apply(perfom_matching, axis=1, args=(indexes, data_ps))

    treated_with_match = ~data_ps.matched_element.isna()
    treated_matched_data = data_ps[treated_with_match][data_ps.columns]
    untreated_matched_data = pd.DataFrame(data=treated_matched_data.matched_element)

    attributes = ['age', 'gender', 'outcome', 'nudge', 'pscore']
    for attr in attributes:
        untreated_matched_data[attr] = untreated_matched_data.apply(obtain_match_details, axis=1, all_data=data_ps, attribute=attr)
        
    untreated_matched_data = untreated_matched_data.set_index('matched_element')

    all_matched_data = pd.concat([treated_matched_data, untreated_matched_data])

    # Get statistics of treatment and control group
    overview = all_matched_data[['outcome','nudge']].groupby(by = ['nudge']).aggregate([np.mean, np.var, np.std, 'count'])
    treated_outcome = overview['outcome']['mean'][1]
    treated_counterfactual_outcome = overview['outcome']['mean'][0]
    att = treated_outcome - treated_counterfactual_outcome
    print('ATE after propensity score matching: {:.4f}'.format(att))

    treated_outcome = treated_matched_data.outcome
    untreated_outcome = untreated_matched_data.outcome

    result = treated_matched_data
    result["control"] = untreated_outcome.values
    tmp = pd.DataFrame(data = {'treated_outcome' : treated_outcome.values, 'untreated_outcome' : untreated_outcome.values})
    result["success"] = (result["outcome"] > result["control"]).astype(int)

    result = result.drop(columns=["nudge", "pscore", "outcome", "control", "matched_element"])
    return result


if __name__ == "__main__":
    # Raw dataset
    filename = "data/external/011_andreas/Commuter experiment_simple.csv"
    df = read(filename)

    # Apply OLS regression and print info
    # print(smf.ols("outcome ~ nudge", data=df.apply(pd.to_numeric)).fit().summary().tables[1])

    # calculate propensity score
    ps = get_pscore(df)
    # check_weights(ps)    
    # plot_confounding_evidence(ps)
    # plot_overlap(ps)

    get_ate(ps)    
    get_psw_ate(ps)
    # get_psm_ate(ps)

    result = match_ps(ps)

    result["nudge_type"] = "[3, 7, 8]"
    result["nudge_domain"] = 3
    result.to_csv("data/processed/data_11.csv", index=False)

