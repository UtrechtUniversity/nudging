import numpy as np
from scipy.optimize import bisect
from scipy import stats

from nudging.dataset.matrix import MatrixData


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


def features_from_cmatrix(
        corr_matrix, n_samples=500, nudge_avg=0.1,
        noise_frac=0.8, control_unique=0.5,
        control_precision=0.5):
    n_features = corr_matrix.shape[0]-2
    L = np.linalg.cholesky(corr_matrix)
    X = np.dot(L, np.random.randn(n_features+2, n_samples)).T
    nudge = np.zeros(n_samples, dtype=int)
    nudge[np.random.choice(n_samples, n_samples//2, replace=False)] = 1
    control_intrinsic = X[:, -1]
    nudge_intrinsic = X[:, -2]
    true_outcome_control = (control_intrinsic*control_unique
                            + (1-control_unique)*nudge_intrinsic)
    true_outcome_control *= control_precision
    true_outcome_nudge = nudge_intrinsic + nudge_avg
    cate = true_outcome_nudge - true_outcome_control
    outcome = (true_outcome_control*(1-nudge)
               + true_outcome_nudge*nudge)
    outcome += (noise_frac/(1-noise_frac))*np.random.randn(n_samples)
    X = X[:, :-2]
    return MatrixData(X, outcome, nudge), {"cate": cate}
