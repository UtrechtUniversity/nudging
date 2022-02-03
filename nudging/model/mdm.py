import numpy as np

from nudging.model.base import BaseModel
from numpy.linalg.linalg import LinAlgError
from sklearn.base import clone


class MultiMDMModel(BaseModel):
    def __init__(self, *args, n_model=5, **kwargs):
        self.n_model = n_model
        super().__init__(*args, **kwargs)
        self.models = [MDMModel(clone(self.model), predictors=self.predictors)
                       for _ in range(n_model)]

    def train(self, data):
        for m in self.models:
            m.train(data)

    def predict_cate(self, data):
        return np.mean([m.predict_cate(data) for m in self.models], axis=0)


class MDMModel(BaseModel):
    def train(self, data):
        X, nudge, outcome = self._X_nudge_outcome(data)
        cov_matrix = np.cov(X, rowvar=False)
        zero_diag = np.where(np.isclose(np.diag(cov_matrix), 0))[0]
        cov_matrix[zero_diag, zero_diag] = 1
        cov_inv = np.linalg.inv(cov_matrix)

        zero_idx = np.where(nudge == 0)[0]
        one_idx = np.delete(np.arange(len(nudge)), zero_idx)
        matches = random_match(X, zero_idx, one_idx, cov_inv)
        distances = [m[2] for m in matches]
        std_dist = np.std(distances)
        matches = [m for i, m in enumerate(matches) if distances[i] < 3*std_dist]
        new_X = []
        new_y = []
        for m in matches:
            new_X.extend([X[m[0]], X[m[1]]])
            new_y.extend(2*[outcome[m[1]]-outcome[m[0]]])
        new_X, new_y = np.array(new_X), np.array(new_y)
        self.model.fit(new_X, new_y)

    def predict_cate(self, data):
        X, _ = self._X_nudge(data)
        return self.model.predict(X)
#         return new_X, new_y
#         return matches


def random_match(X, zero_idx, one_idx, cov_inv):
    matches = []
    available_idx = np.ones(len(one_idx), dtype=np.bool_)
    permutation = np.random.permutation(zero_idx)
    for i_perm in range(len(permutation)):
        cur_id = permutation[i_perm]
        X_comp = X[one_idx[available_idx]]
        if X_comp.shape[0] == 0:
            break
        distances = (np.dot(X_comp-X[cur_id, :], cov_inv)*(X_comp-X[cur_id, :])).sum(axis=1)
        i_min = np.argmin(distances)
        min_id = one_idx[np.where(available_idx)[0][i_min]]
        matches.append((cur_id, min_id, distances[i_min]))
    return matches