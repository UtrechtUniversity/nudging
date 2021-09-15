"""Predict nudge effectiveness for subgroups"""
from itertools import product
from pathlib import Path

from joblib import load
import matplotlib.pyplot as plt

from utils import clean_dirs, read_data


def flatten(data):
    """ Flatten list of tuples or tuple of tuples
    Args:
        data (list or tuple): list/tuple of data to flatten
    Returns:
        list: flattened list
    """
    result = []
    for var in data:
        if isinstance(var, tuple):
            result = flatten(var)
        else:
            result.append(var)

    return result


def plot_probability(data, labels, outdir):
    """Plot nudge effectiveness (probability of success) as function of age for each nudge type
    Args:
       data (pandas.DataFrame): dataframe with nudge effectiveness
       labels (dict): dict with labels to choose what to plot
       outdir (str): folder name to save plots
    """
    width = 0.2
    _, axis = plt.subplots()
    colors = ["b", "r", "g", "c", "m", "y", "k"]
    types = data['nudge_type'].unique()
    position = 0
    condition = True
    for label in labels:
        condition = (data[label] == labels[label]) & condition

    for i, nudge in enumerate(types):
        dataset = data[(data['nudge_type'] == nudge) & condition]
        if dataset.empty:
            continue
        label = "nudge {}".format(nudge)
        dataset.plot(
            ax=axis, kind='bar', x='age', y='probability', label=label,
            color=colors[i], width=width, position=position)
        position += 1
    axis.set_xlabel('age [decades]')
    axis.set_ylabel('probability of nudge success')
    name = ""
    for key, value in labels.items():
        name = name + f"{key}_{value}_"
    name = name[:-1]
    plt.title(name)
    filename = Path(outdir, name + ".png")
    plt.savefig(filename)


def plot_data(data, predictors, outdir):
    """Make plots of nudge effectiveness for each subject subgroup
    Args:
        predictors (list): list of predictors
        data (pandas.DataFrame): dataframe with nudge effectivenedd for all subgroups
        outdir (str): folder name to save plots
    """
    plot_dict = {}
    for predictor in predictors:
        if predictor not in ["nudge_type", "age"]:
            plot_dict[predictor] = data[predictor].unique()

    for i, value in enumerate(plot_dict.values()):
        if i == 0:
            prod = value
        else:
            prod = product(prod, value)

    for term in prod:
        if isinstance(term, tuple):
            label = flatten(term)
        else:
            label = [term]
        labels = {key: label[i] for i, key in enumerate(plot_dict.keys())}
        plot_probability(data, labels, outdir)


if __name__ == "__main__":

    # Cleanup old data
    outdirs = ["data/processed", "plots"]
    # Make sure output dirs exist and are empty
    clean_dirs(outdirs)

    # Read combined dataset, note this is the same as used for training
    features = ["nudge_domain", "age", "gender", "nudge_type"]
    df_nonan = read_data("data/interim/combined.csv", features)
    subgroups = df_nonan[features].drop_duplicates().sort_values(by=features, ignore_index=True)

    # Load model
    model = load("models/nudging.joblib")

    # Calculate probabilties for all subgroups and write to file
    prob = subgroups.assign(
        probability=model.predict_proba(subgroups)[:, 1], success=model.predict(subgroups))
    prob.to_csv("data/processed/nudge_probabilty.csv", index=False)

    plot_data(prob, features, "plots")
