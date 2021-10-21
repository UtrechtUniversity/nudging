from nudging.reader.base import BaseDataSet
from pandas import DataFrame

import yaml
import json
from copy import deepcopy


class MatrixData(BaseDataSet):
    def __init__(self, X, outcome, nudge, names=None):
        self.df = DataFrame(X)
        if names is not None:
            self.df.set_axis(names, axis=1, inplace=True)
        else:
            self.df.set_axis([str(x) for x in list(self.df)], axis=1,
                             inplace=True)
        self.df["outcome"] = outcome
        self.df["nudge"] = nudge
        self.standard_df = self.df

    def _load(self, fp):
        pass

    def to_csv(self, csv_fp, config_fp, truth_fp=None):
        """Write to file in a format that can be easily read from file."""
        covariates = self.covariates
        try:
            # If there is a truth, add nudge type and nudge domain
            new_df = self.df.copy()
            if "nudge_type" in self.truth:
                new_df["nudge_type"] = self.truth["nudge_type"]
                covariates.append("nudge_type")
            if "nudge_domain" in self.truth:
                new_df["nudge_domain"] = self.truth["nudge_domain"]
                covariates.append("nudge_domain")
        except AttributeError:
            new_df = self.df

        # Write to CSV
        new_df.to_csv(csv_fp, index=False)
        feature_dict = {"features": covariates,
                        "plot": {"x": "age"}}

        # Create the configuration file
        with open(config_fp, "w") as f:
            yaml.dump(feature_dict, f)

        # Create the truth/json file.
        t = deepcopy(self.truth)
        t["cate"] = t["cate"].tolist()
        if truth_fp is not None:
            with open(truth_fp, "w") as f:
                json.dump(t, f)
