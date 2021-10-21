import numpy as np
from scipy import stats

# from protosc.feature_matrix import FeatureMatrix
from nudging.reader.matrix import MatrixData


def _slightly_correlated_data(n_features=20, n_samples=500, nudge_avg=0.1,
                              noise_frac=0.8, control_unique=0.5,
                              control_precision=0.5, eigen_power=1):
    eigen_vals = np.random.rand(n_features+2)**eigen_power
    eigen_vals *= len(eigen_vals)/np.sum(eigen_vals)
    corr_matrix = stats.random_correlation.rvs(eigen_vals)
    L = np.linalg.cholesky(corr_matrix)
    X = np.dot(L, np.random.randn(n_features+2, n_samples)).T
#     prob_y = 1/(1+np.exp(-sigma*X[:, -1]))
#     max_accuracy = np.mean(np.abs(prob_y-0.5)+0.5)
    nudge = np.zeros(n_samples, dtype=int)
    nudge[np.random.choice(n_samples, n_samples//2, replace=False)] = 1
    control_intrinsic = X[:, -1]
    nudge_intrinsic = X[:, -2]
    true_outcome_control = control_intrinsic*control_unique + (1-control_unique)*nudge_intrinsic
    true_outcome_control *= control_precision
    true_outcome_nudge = nudge_intrinsic + nudge_avg
    cate = true_outcome_nudge - true_outcome_control
    outcome = (1-noise_frac)*(true_outcome_control*(1-nudge) + true_outcome_nudge*nudge) + noise_frac*np.random.randn(n_samples)
    X = X[:, :-2]
    return MatrixData(X, outcome, nudge), {"cate": cate}


def get_random_data(
        n_features=None, n_samples=None, nudge_avg=None,
        noise_frac=None, control_unique=None, control_precision=None,
        eigen_power=None):
    """Create a feature matrix with correlations.

    All keyword arguments that remain at None, will have their values
    randomized.

    Arguments
    ---------
    n_features: int
        Number of features.
    n_samples: int
        Number of samples. [1, infty]
    nudge_avg: double
        Average treatment effect. Higher values mean that the
        treatment effect is higher. [0, infty]
    noise_frac: double
        Fraction of the signal that is gaussian noise. Higher values
        mean more noise. [0, 1]
    control_unique: double
        Measure of how different the response between control and nudge groups
        are after removing the ATE. Higher values means bigger difference
        between control and nudge responses. [0, 1]
    control_precision: double
        Strength of the dependency of the control group on the other features.
        Higher values have stronger dependency, with 1 being equal to the nudge
        group. [0, 1]
    eigen_power: double
        The eigenvalues are distributed according to minus this value power.
        [-3. 3]
    """
    if n_features is None:
        n_features = np.random.randint(10, 50)
    if n_samples is None:
        n_samples = np.random.randint(200, 2000)
    if nudge_avg is None:
        nudge_avg = 0.3*np.random.random()
    if noise_frac is None:
        noise_frac = 0.1+0.8*np.random.random()
    if control_unique is None:
        control_unique = np.random.random()
    if control_precision is None:
        control_precision = 0.2 + 0.8*np.random.random()
    if eigen_power is None:
        eigen_power = -4
        while eigen_power > 3 or eigen_power < -3:
            eigen_power = np.random.randn()

    X, truth = _slightly_correlated_data(
        n_features, n_samples, nudge_avg, noise_frac, control_unique,
        control_precision, eigen_power)

    truth.update({
        "n_features": n_features,
        "n_samples": n_samples,
        "nudge_avg": nudge_avg,
        "noise_frac": noise_frac,
        "control_unique": control_unique,
        "control_precision": control_precision,
        "eigen_power": eigen_power,
    })
    return X, truth
