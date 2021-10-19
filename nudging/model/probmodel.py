import numpy as np

from nudging.model.base import BaseModel
import nudging.propensity_score as ps

class ProbModel(BaseModel):
    """class for Probalistic model"""

    def _get_success(self, data_frame):
        """Convert outcome to nudge success"""
        data_frame["success"] = np.greater(
            data_frame["outcome"], data_frame["control"]).astype(int)
        result = data_frame.drop(columns=["outcome", "control"])
        return result

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

        return result
