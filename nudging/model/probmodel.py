import numpy as np
import pandas as pd

from nudging.model.base import BaseModel
import nudging.model.propensity_score as ps

class ProbModel(BaseModel):
    """class for Probalistic model"""

    def _get_success(self, data_frame):
        """Convert outcome to nudge success"""
        result = data_frame.copy(deep=True)
        result["outcome"] = np.greater(
            data_frame["outcome"], data_frame["control"]).astype(int)
        return result.drop(columns=["control"])

    def preprocess(self, data_frame):
        """Convert raw dataset to standard-format csv file with nudge succes
        Args:
            name (str): name of reader
            path (str): path to original dataset
        Returns:
            pandas.DataFrame: dataframe with nudge success for subjects in treatment group
        """

        data_frame.reset_index(drop=True, inplace=True)

        # Apply OLS regression and print info
        # print(
        #    smf.ols("outcome ~ nudge",
        # data=data_frame.apply(pd.to_numeric)).fit().summary().tables[1])

        # calculate propensity score
        df_ps = ps.get_pscore(data_frame)

        # Check if treatment and control groups are well-balanced
        # ps.check_weights(df_ps)

        # Plots
        # ps.plot_confounding_evidence(df_ps)
        # ps.plot_overlap(df_ps)

        # Average Treatment Effect (ATE)
        ps.get_ate(df_ps)

        # propensity score weigthed ATE
        ps.get_psw_ate(df_ps)

        # propensity score matched ATE with CausalModel
        # ps.get_psm_ate(df_ps)

        # Calculate nudge success
        result = ps.match_ps(df_ps)
        result = self._get_success(result)
                
        # Convert age to decades if present
        if 'age' in result.columns:
            result["age"] = (result["age"]/10.).astype(int)

        return result


    def predict_cate(self, X):
        return self.model.predict_proba(X[self.predictors])[:, 1]

