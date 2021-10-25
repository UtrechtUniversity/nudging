import numpy as np

from nudging.simulation.utils import create_corr_matrix, features_from_cmatrix


def rescale(var, min_value, max_value):
    """ Rescale data to given range
    Args:
        var (series): data to be rescaled
        min_value (int): min value of new range
        max_value (int): max value of new range
    Returns:
        series: rescaled data
    """
    result = (((var - var.min())/(var.max() - var.min() + 1e-12)) *
              (max_value - min_value) + min_value)
    return result.astype(int)


def generate_multi_dataset(
        n_dataset=10,
        n_features_correlated=8, n_features_uncorrelated=2, eigen_power=1.5,
        n_nudge_domain=3, n_nudge_type=3, nudge_domain_weight=0.1,
        nudge_type_weight=0.2, dataset_weight=0.2):

    # Get the weights for the correlation matrices

    if n_nudge_domain <= 1:
        nudge_domain_weight = 0
    if n_nudge_type <= 1:
        nudge_type_weight = 0

    w = (nudge_domain_weight + nudge_type_weight + dataset_weight)
    assert w < 1 + 1e-7, "variations have to add up to less than 1"
    assert w > 1e-7, (
            "variations have to have at least some weight, otherwise"
            "all datasets are the same.")
    base_weight = max(0, 1-w)

    # Set the correlation arguments for all cor matrices.
    corr_kwargs = {
        "n_features_correlated": n_features_correlated,
        "n_features_uncorrelated": n_features_uncorrelated,
        "eigen_power": eigen_power,
    }

    # Each domain/nudge type has a specific correlation matrix
    # To get the final one, add them together while weighting them.
    base_corr_matrix = create_corr_matrix(**corr_kwargs)
    nudge_domain_matrices = [create_corr_matrix(**corr_kwargs)
                             for _ in range(n_nudge_domain)]
    nudge_type_matrices = [create_corr_matrix(**corr_kwargs)
                           for _ in range(n_nudge_type)]

    # Initial space for the paramaters. The nudge type and domain
    # narrow this range consecutively.
    initial_settings = {
        "nudge_avg": np.array([0, 0.3]),
        "noise_frac": np.array([0.1, 0.9]),
        "control_unique": np.array([0, 1.0]),
        "control_precision": np.array([0.2, 1.0]),
    }

    # Each domain/type has their own bias for each of the parameters.
    random_cor_vars = list(initial_settings)
    domain_var_pos = [{random_cor_vars[i]: np.random.random()
                       for i in range(len(random_cor_vars))}
                      for _ in range(n_nudge_domain)]
    type_var_pos = [{random_cor_vars[i]: np.random.random()
                     for i in range(len(random_cor_vars))}
                    for _ in range(n_nudge_type)]

    # Compute the weights when going from one level to the next
    # i.e. all -> specific nudge domain -> specific nudge type.
    variations = np.array([nudge_domain_weight, nudge_type_weight,
                           dataset_weight]).astype(np.float)
    variations /= np.sum(variations)
    var_left = 1-np.cumsum(variations)
    update_weights = [var_left[0], var_left[1]/var_left[0]]

    # Generate the feature matrices.
    all_matrices = []
    for _ in range(n_dataset):
        # Set the nudge domain/type
        nudge_domain = np.random.randint(n_nudge_domain)
        nudge_type = np.random.randint(n_nudge_type)

        # Generate the correlation matrix
        corr_matrix = base_corr_matrix*base_weight
        corr_matrix += nudge_domain_matrices[nudge_domain]*nudge_domain_weight
        corr_matrix += nudge_type_matrices[nudge_type]*nudge_type_weight
        corr_matrix += create_corr_matrix(**corr_kwargs)*dataset_weight

        # Get the parameter settings for the dataset.
        settings = update_settings(
            initial_settings, domain_var_pos[nudge_domain], update_weights[0])
        settings = update_settings(
            settings, type_var_pos[nudge_type], update_weights[1])
        feature_kwargs = kwargs_from_settings(settings)

        # Number of samples is independent on the nudge type/domain.
        feature_kwargs["n_samples"] = np.random.randint(300, 5000)
        X, truth = features_from_cmatrix(corr_matrix, **feature_kwargs)
        truth.update(feature_kwargs)
        truth.update(corr_kwargs)
        X.truth = truth
        X.truth["nudge_domain"] = nudge_domain
        X.truth["nudge_type"] = nudge_type
        all_matrices.append(X)
        if n_features_uncorrelated >= 2:
            X.df["0"] = rescale(X.df["0"].values, 18, 80)
            X.df["1"] = rescale(X.df["1"].values, 0, 2)
            rename_dict = {"0": "age", "1": "gender"}
            X.df.rename(columns=rename_dict, inplace=True)
    return all_matrices


def update_settings(settings, transformation, weight):
    new_settings = {}
    for var, bounds in settings.items():
        interval = bounds[1]-bounds[0]
        new_interval = interval*weight
        new_start = bounds[0] + transformation[var]*(interval-new_interval)
        new_bounds = np.array([new_start, new_start+new_interval])
        new_settings[var] = new_bounds
    return new_settings


def kwargs_from_settings(settings):
    kwargs = {}
    for var, bounds in settings.items():
        kwargs[var] = np.random.random()*(bounds[1]-bounds[0])+bounds[0]
    return kwargs
