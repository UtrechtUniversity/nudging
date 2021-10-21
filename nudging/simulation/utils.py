import numpy as np
from nudging.reader.matrix import MatrixData


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
