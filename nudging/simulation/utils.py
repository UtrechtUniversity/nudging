import numpy as np
from numpy import fabs
from scipy.optimize import bisect
from scipy import stats

from nudging.dataset.matrix import MatrixData


class Bounds():
    def __init__(self, value, int_val=False):
        self.int_val = int_val
        if isinstance(value, (list, tuple, np.ndarray)):
            if int_val:
                self.value = np.array(value, dtype=int)
            else:
                self.value = np.array(value)
        elif int_val:
            self.value = np.array([value, value+1], dtype=int)
        else:
            self.value = np.array([value, value], dtype=np.float)

    def rand(self):
        if self.int_val:
            return np.random.randint(*self.value)
        r = np.random.rand()
        return r*(self.value[1]-self.value[0]) + self.value[0]

    def max(self):
        return self.value[1]

    def min(self):
        return self.value[0]


# def convert_to_bounds(value, int_val=False):
#     if isinstance(value, (list, tuple, np.ndarray)):
#         return np.array(value), int_val
#     if int_val:
#         return np.array([value, value+1], dtype=int), int_val

#     return np.array([value, value], dtype=np.float), int_val
#     if not isinstance(new_value, np.ndarray):
#         new_value = np.array([new_value, new_value])
#     settings[name] = new_value


def mixed_features(n_features_uncorrelated=10, n_features_correlated=10,
                   eigen_power=3, **kwargs):
    corr_matrix = create_corr_matrix(
        n_features_uncorrelated, n_features_correlated,
        eigen_power)
    return features_from_cmatrix(corr_matrix, **kwargs)


def find_smallest_alpha(M_zero, M_one):
    def f(alpha):
        return np.min(np.linalg.eigvalsh(M_zero*(1-alpha)+M_one*(alpha)))

    if f(1) > 0 or fabs(f(1)) < 1e-12:
        return 1
    if f(1e-8) < 0:
        return 0

    try:
        return bisect(f, 1e-8, 1)
    except ValueError as e:
        print(f(0), f(1e-8), f(1))
        raise e


def create_corr_matrix(n_features_uncorrelated=10, n_features_correlated=10,
                       eigen_power=3, rng=None):
    n_tot_features = 2 + n_features_uncorrelated + n_features_correlated
    n_iid = n_features_uncorrelated
    if rng is None:
        eigen_vals = np.random.rand(n_tot_features)**eigen_power
    else:
        eigen_vals = rng.random(n_tot_features)**eigen_power
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


def _transform_outcome(outcome, a, powers=np.array([1, 0.5, 0.1])):
    ret_outcome = np.zeros_like(outcome)
    a *= powers
    for i in range(len(a)):
        ret_outcome += (a[i]-powers[i]/2)*outcome**(i+1)
    return ret_outcome


def features_from_cmatrix(
        corr_matrix, n_samples=500, nudge_avg=0.1,
        noise_frac=0.8, control_unique=0.5,
        control_precision=0.5, linear=True, **kwargs):
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
    if not linear:
        a = np.random.rand(3)
        true_outcome_control = _transform_outcome(true_outcome_control, a)
        true_outcome_nudge = _transform_outcome(true_outcome_nudge, a)
        for i_col in range(n_features):
            a = np.random.rand(3)
            X[:, i_col] = _transform_outcome(X[:, i_col], a)

    cate = true_outcome_nudge - true_outcome_control
    outcome = (true_outcome_control*(1-nudge)
               + true_outcome_nudge*nudge)
    outcome += (noise_frac/(1-noise_frac))*np.random.randn(n_samples)
    X = X[:, :-2]
    truth = {
        "cate": cate, "n_samples": n_samples, "nudge_avg": nudge_avg,
        "noise_frac": noise_frac, "control_unique": control_unique,
        "control_precision": control_precision, "linear": linear,
    }
    matrix = MatrixData.from_data((X, nudge, outcome), **kwargs, truth=truth)
    return matrix
