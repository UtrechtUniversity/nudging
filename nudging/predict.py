"""Predict nudge effectiveness for subgroups"""
from itertools import product
from pathlib import Path

from joblib import load
import matplotlib.pyplot as plt
import yaml

from nudging.utils import clean_dirs, read_data


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


def plot_probability(data, labels, outdir, xaxis, cat="nudge_type"):
    """Plot nudge effectiveness (probability of success) as function of x for each nudge type
    Args:
       data (pandas.DataFrame): dataframe with nudge effectiveness
       labels (dict): dict with labels to choose what to plot
       outdir (str): folder name to save plots
       xaxis (str): feature to use as x-axis
    """
    width = 0.2
    _, axis = plt.subplots()
    colors = ["b", "r", "g", "c", "m", "y", "k"]
    types = data[cat].unique()
    position = 0
    condition = True
    xticks = []
    for label in labels:
        condition = (data[label] == labels[label]) & condition

    for i, nudge in enumerate(types):
        dataset = data[(data[cat] == nudge) & condition]
        if dataset.empty:
            continue
        label = "{} {}".format(cat, nudge)
        dataset.plot(
            ax=axis, kind='bar', x=xaxis, y='probability', label=label,
            color=colors[i % 7], width=width, position=position)
        # get ticks of largest range
        if len(axis.get_xticks()) > len(xticks):
            xticks = axis.get_xticks()
        position += 1

    axis.set_xticks(xticks)
    axis.set_xticklabels(axis.get_xmajorticklabels())
    axis.set_xlabel(xaxis)
    axis.set_ylabel('nudge effectiveness')
    axis.set_ylim([0, 1])

    name = ""
    for key, value in labels.items():
        name = name + f"{key}_{value}_"
    if len(name) > 0:
        name = name[:-1]
    else:
        name = "nudge"
    filename = Path(outdir, name + ".png")
    plt.autoscale(axis='x')
    plt.savefig(filename)
    print(f"Plot generated: {filename}")


def plot_data(data, predictors, outdir, xaxis, cat="nudge_type"):
    """Make plots of nudge effectiveness for each subject subgroup
    Args:
        predictors (list): list of predictors
        data (pandas.DataFrame): dataframe with nudge effectivenedd for all subgroups
        outdir (str): folder name to save plots
        xaxis (str): feature to use as x-axis
    """
    plot_dict = {}
    prod = []
    labels = {}
    for predictor in predictors:
        if predictor not in [cat, xaxis]:
            plot_dict[predictor] = data[predictor].unique()
    if not plot_dict:
        plot_probability(data, labels, outdir, xaxis, cat)

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
        plot_probability(data, labels, outdir, xaxis, cat)


if __name__ == "__main__":

    # Cleanup old data
    outdirs = ["data/processed", "plots"]
    # Make sure output dirs exist and are empty
    clean_dirs(outdirs)

    # Read combined dataset, note this is the same as used for training
    config = yaml.safe_load(open("config.yaml"))
    features = config["features"]
    x_axis = config["plot"]["x"]
    df_nonan = read_data("data/interim/combined.csv", features)
    subgroups = df_nonan[features].drop_duplicates().sort_values(by=features, ignore_index=True)

    # Load model
    model = load("models/nudging.joblib")

    # Calculate probabilities for all subgroups and write to file
    prob = subgroups.assign(
        probability=model.predict_cate(subgroups))
    prob.to_csv("data/processed/nudge_probabilty.csv", index=False)
    print("Output written to data/processed/nudge_probabilty.csv")
    plot_data(prob, features, "plots", x_axis, "gender")
