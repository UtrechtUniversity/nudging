import numpy as np
from scipy.optimize import bisect
from scipy import stats
from nudge.simulation.utils import features_from_cmatrix


def find_smallest_alpha(M_zero, M_one):
    def f(alpha):
        return np.min(np.linalg.eigvalsh(M_zero*(1-alpha)+M_one*(alpha)))

    if f(1) > 0:
        return 1
    return bisect(f, 1e-6, 1)


def create_corr_matrix(n_features_uncorrelated=10, n_features_correlated=10,
                       eigen_power=3):
    n_tot_features = 2 + n_features_uncorrelated + n_features_correlated
    n_iid = n_features_uncorrelated
    eigen_vals = np.random.rand(n_tot_features)**eigen_power
    eigen_vals *= len(eigen_vals)/np.sum(eigen_vals)
    base_corr_matrix = stats.random_correlation.rvs(eigen_vals)
    M_zero = np.zeros_like(base_corr_matrix)
    M_zero[n_iid:, n_iid:] = base_corr_matrix[n_iid:, n_iid:]
    M_zero[:n_iid, :n_iid] = np.identity(n_iid)
    M_one = np.copy(base_corr_matrix)
    M_one[:n_iid, :n_iid] = np.identity(n_iid)
    alpha = find_smallest_alpha(M_zero, M_one) - 1e-10
    M = M_zero*(1-alpha)+M_one*alpha
    return M


def mixed_features(n_features_uncorrelated=10, n_features_correlated=10,
                   eigen_power=3, seed=None, **kwargs):
    np.random.seed(seed)
    corr_matrix = create_corr_matrix(
        n_features_uncorrelated, n_features_correlated,
        eigen_power)
    return features_from_cmatrix(corr_matrix, **kwargs)
