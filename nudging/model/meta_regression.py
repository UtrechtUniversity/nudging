import numpy as np
from scipy import stats
from hyperopt import hp, fmin, tpe


class MetaRegressionModel():
    def __init__(self, model):
        """A regression/meta model to train a model from multiple datasets

        It works by training a classifier for each of the datasets. For
        a new dataset, the expected cate is a weighted average of the
        trained classifiers, with different weights for each of the
        classifiers. The weights are determined in an optimization routine for
        each of the combinations of nudge domain and types, where classifiers
        with the same domain and types are weighted more strongly (generally).

        Arguments
        ---------
        model: BaseModel
            ML model that can predict the cate.
        """
        self._model = model

    def train(self, multi_dataset):
        self._classifiers = [self._model.clone() for _ in multi_dataset]
        for i, cl in enumerate(self._classifiers):
            cl.train(multi_dataset[i].standard_df)
        self._domains = np.array([d.truth["nudge_domain"]
                                  for d in multi_dataset])
        self._types = np.array([d.truth["nudge_type"] for d in multi_dataset])
        unq_domains = np.unique(self._domains)

        # Determine the model parameters for each of the domains/types.
        self._model_parameters = {}
        for domain_id in unq_domains:
            self._model_parameters[domain_id] = {}
            unq_types = np.unique(self._types[self._domains == domain_id])
            for type_id in unq_types:
                self._model_parameters[domain_id][type_id] = self._train_domain_type(
                    multi_dataset, domain_id, type_id)

    def _train_domain_type(self, multi_dataset, domain_id, type_id):
        """Compute the model parameters for one of the domain/type combinations
        """
        n_data = len(self._classifiers)
        same_type = (self._types == type_id)
        same_domain = (self._domains == domain_id)
        same_domain_type = np.logical_and(same_type, same_domain)
        test_results = {}
        for test_id in np.where(same_domain_type)[0]:
            not_test_id = np.ones(n_data, dtype=np.bool_)
            not_test_id[test_id] = False
            same_type_idx = np.where(np.logical_and(same_type, not_test_id))[0]
            same_domain_idx = np.where(np.logical_and(same_domain, not_test_id))[0]
            same_domain_type_idx = np.where(np.logical_and(same_domain_type, not_test_id))[0]
            all_idx = np.delete(np.arange(n_data), [test_id])
            test_results[test_id] = compute_test_cate(
                self._classifiers, multi_dataset, all_idx, same_type_idx,
                same_domain_idx, same_domain_type_idx, test_id)
        return optimize_results(test_results)

    def predict_cate(self, test_data):
        test_type = test_data.truth["nudge_type"]
        test_domain = test_data.truth["nudge_domain"]
        same_type = self._types == test_type
        same_domain = self._domains == test_domain
        same_domain_type = np.logical_and(same_type, same_domain)
        cates = np.array([cl.predict_cate(test_data.standard_df)
                          for cl in self._classifiers])
        try:
            coefs = self._model_parameters[test_domain][test_type]
        except KeyError:
            coefs = {
                "a_all": 0.5,
                "a_domain": 0.5,
                "a_type": 0.5,
            }
        cate_est = np.zeros(len(test_data.standard_df))
        cate_est += coefs["a_all"]*np.sum(cates, axis=0)
        cate_est += coefs["a_type"]*np.sum(cates[same_type], axis=0)
        cate_est += coefs["a_domain"]*np.sum(cates[same_domain], axis=0)
        cate_est += np.sum(cates[same_domain_type], axis=0)
        return cate_est


def compute_test_cate(classifiers, multi_dataset, all_idx, same_type_idx,
                      same_domain_idx, same_domain_type_idx, test_id):
    results = {"other": [0, 0, 0, 0], "self": 0}
    test_data = multi_dataset[test_id]
    n_samples = len(test_data.standard_df)

    def get_sum_cate(idx):
        sum_cate = np.zeros(n_samples)
        for cur_id in idx:
            sum_cate += classifiers[cur_id].predict_cate(test_data.standard_df)
        if len(idx) > 0:
            return sum_cate*(len(idx)+1)/len(idx)
        return sum_cate
    results["other"][0] = get_sum_cate(all_idx)
    results["other"][1] = get_sum_cate(same_type_idx)
    results["other"][2] = get_sum_cate(same_domain_idx)
    results["other"][3] = get_sum_cate(same_domain_type_idx)
    results["self"] = classifiers[test_id].predict_cate(test_data.standard_df)
    return results


def optimize_results(results):
    space = [
        hp.uniform('a_all', 0, 1),
        hp.uniform('a_type', 0, 1),
        hp.uniform('a_domain', 0, 1),
    ]

    def optimizer(x):
        corr = 0
        for res in results.values():
            mixed_cate = res["other"][3]
            for i in range(0, len(x)):
                mixed_cate += x[i]*res["other"][i]
            if np.all(mixed_cate == mixed_cate[0]):
                print(mixed_cate[0], x)
                print(res)
            corr += -stats.spearmanr(mixed_cate, res["self"]).correlation
        return corr / len(results)

    best = fmin(optimizer, space=space, algo=tpe.suggest, max_evals=100,
                show_progressbar=False)
    return best
