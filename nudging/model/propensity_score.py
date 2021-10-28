""" Reference: https://matheusfacure.github.io/python-causality-handbook/11-Propensity-Score.html#
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from matplotlib import pyplot as plt
from causalinference import CausalModel

pd.options.mode.chained_assignment = None  # default='warn'


def get_pscore(data_frame, solver='liblinear'):
    """Calculate propensity score with logistic regression
    Args:
        data_frame (pandas.DataFrame): dataframe with input data

    Returns:
        pandas.DataFrame: dataframe with propensity score
    """
    treatment = 'nudge'
    outcome = 'outcome'
    predictors = data_frame.columns.drop([treatment, outcome])
    ps_model = LogisticRegression(solver=solver).fit(
        data_frame[predictors].to_numpy().astype('int'),
        data_frame[treatment].to_numpy().astype('int')
    )

    data_ps = data_frame.assign(pscore=ps_model.predict_proba(data_frame[predictors])[:, 1])

    return data_ps


def check_weights(data_ps):
    """Check if sum of propensity score weights match sample size
    Args:
        data_ps (pandas.DataFrame): dataframe with propensity score
    Return:
        tuple: sample size, treated size from weigths, untreated size froms weigths
    """
    weight_t = 1./data_ps.query("nudge==1")["pscore"]
    weight_nt = 1./(1.-data_ps.query("nudge==0")["pscore"])
    print(f"Original sample size {data_ps.shape[0]}")
    print("Original treatment sample size", data_ps.query("nudge==1").shape[0])
    print("Original control sample size", data_ps.query("nudge==0").shape[0])
    print(f"Weighted treatment sample size {round(sum(weight_t), 1)}")
    print(f"Weighted control sample size {round(sum(weight_nt), 1)}")

    return data_ps.shape[0], sum(weight_t), sum(weight_nt)


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
    Returns:
        float: average treatment effect
    """
    result = data_ps.groupby("nudge")["outcome"].mean()
    ate = result[1] - result[0]
    # print("Calculate Average Treatment Effect:")
    # print(f"Control: {round(result[0], 3)}")
    # print(f"Treatment: {round(result[1], 3)}")
    # print(f"ATE: {round(ate, 3)}")

    return ate


def get_psw_ate(data_ps):
    """Get propensity score weigthed ATE
    Args:
        data_ps (pandas.DataFrame): dataframe with propensity score
    """

    weight = ((data_ps["nudge"] - data_ps["pscore"]) / (data_ps["pscore"]*(1. - data_ps["pscore"])))
    ate = np.mean(weight * data_ps["outcome"])

    # weight_t = 1./data_ps.query("nudge==1")["pscore"]
    # weight_nt = 1./(1.-data_ps.query("nudge==0")["pscore"])
    # treatment = sum(data_ps.query("nudge==1")["outcome"]*weight_t) / len(data_ps)
    # control = sum(data_ps.query("nudge==0")["outcome"]*weight_nt) / len(data_ps)
    # ate = treatment - control

    # print(f"Propensity score weighted ATE: {round(ate, 3)}")

    return ate


def get_psm_ate(data_ps):
    """Get propensity score matched ATE using CausalModel
    Args:
        data_ps (pandas.DataFrame): dataframe with propensity score
    """
    cmodel = CausalModel(
        Y=data_ps["outcome"].values,
        D=data_ps["nudge"].values,
        X=data_ps[["pscore"]].values
    )

    cmodel.est_via_ols()
    cmodel.est_via_matching(matches=1, bias_adj=True)
    print(cmodel.estimates)


def match_ps(data_ps):
    """Match participants in treatment group to control group by propensity score
    Args:
        data_ps (pandas.DataFrame): dataframe with propensity score of all participants
    Returns:
        pandas.DataFrame: dataframe with nudged participants and matched control
    """
    df1 = data_ps.reset_index()[data_ps["nudge"]==1]
    df2 = data_ps.reset_index()[data_ps["nudge"]==0]
    result = pd.merge_asof(df1.sort_values('pscore'),
                   df2.sort_values('pscore'),
                   on='pscore', 
                   direction='nearest', 
                   suffixes=['', '_control'])


    columns = list(df1) + ['control'] 
    result = result.rename(columns={"outcome_control": "control"})
    result = result[columns].sort_values('index').reset_index(drop=True).drop(columns=['index', 'nudge', 'pscore'])

    return result
