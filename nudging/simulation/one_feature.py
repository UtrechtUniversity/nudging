import numpy as np


def generate_data(f_nudge, f_control, n_data, noise=1):
    x = np.linspace(0, 1, n_data, endpoint=True)
    y_nudge = f_nudge(x)
    y_control = f_control(x)
    nudge_idx = np.random.choice(n_data, size=n_data//2, replace=False)
    control_idx = np.delete(np.arange(n_data), nudge_idx)
    X_nudge = np.zeros(n_data, dtype=int)
    X_nudge[nudge_idx] = 1
    y = np.zeros(n_data)
    y[nudge_idx] = y_nudge[nudge_idx]
    y[control_idx] = y_control[control_idx]
    return {
        "nudge": X_nudge,
        "outcome": y,
        "y_model_nudge": y_nudge+noise*np.random.randn(n_data),
        "y_model_control": y_control + noise*np.random.randn(n_data)
    }
