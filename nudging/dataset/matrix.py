"""DataSet class for simlated matrix data"""
from copy import deepcopy
import json

from pandas import DataFrame
import yaml

from nudging.dataset.base import BaseDataSet


class MatrixData(BaseDataSet):
    """Class MatrixData"""
    def __init__(self, X, outcome, nudge, names=None):
        self.standard_df = DataFrame(X)
        if names is not None:
            self.standard_df.set_axis(names, axis=1, inplace=True)
        else:
            self.standard_df.set_axis(
                [str(x) for x in list(self.standard_df)], axis=1, inplace=True)
        self.standard_df["outcome"] = outcome
        self.standard_df["nudge"] = nudge
        super().__init__()

    def to_csv(self, csv_fp, config_fp, truth_fp=None):
        """Write to file in a format that can be easily read from file."""
        covariates = self.covariates
        try:
            # If there is a truth, add nudge type and nudge domain
            new_df = self.standard_df.copy()
            if "nudge_type" in self.truth:
                new_df["nudge_type"] = self.truth["nudge_type"]
                covariates.append("nudge_type")
            if "nudge_domain" in self.truth:
                new_df["nudge_domain"] = self.truth["nudge_domain"]
                covariates.append("nudge_domain")
        except AttributeError:
            new_df = self.standard_df

        # Write to CSV
        new_df.to_csv(csv_fp, index=False)
        feature_dict = {"features": covariates,
                        "plot": {"x": "age"}}

        # Create the configuration file
        with open(config_fp, "w") as file_:
            yaml.dump(feature_dict, file_)

        # Create the truth/json file.
        truth = deepcopy(self.truth)
        truth["cate"] = truth["cate"].tolist()
        if truth_fp is not None:
            with open(truth_fp, "w") as file_:
                json.dump(truth, file_)
