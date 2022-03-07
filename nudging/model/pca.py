from sklearn.decomposition import PCA
import numpy as np
from nudging.model.matching import MatchingModel


class PCAModel(MatchingModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pca_model = PCA()

    def train(self, data):
        X, nudge, outcome = self._X_nudge_outcome(data)
        X_pca = self.pca_model.fit_transform(X)
        self.pca_fac = np.sqrt(self.pca_model.explained_variance_ratio_)
        matches = multimatch(X_pca*self.pca_fac, nudge)
        distances = [m[2] for m in matches]
        std_dist = np.std(distances)
        matches = [m for i, m in enumerate(matches)
                   if distances[i] <= np.mean(distances) + std_dist]

        new_X = []
        new_y = []
        for m in matches:
            new_X.extend([X_pca[m[0]], X_pca[m[1]]])
            new_y.extend(2*[outcome[m[1]]-outcome[m[0]]])
        new_X, new_y = np.array(new_X), np.array(new_y)
        if len(new_X.shape) == 1:
            print(distances)
            print(nudge)
            print(new_X.shape)
            print(matches)
        self.model.fit(new_X, new_y)
        if self.train_tlearner:
            all_idx = []
            for m in matches:
                all_idx.extend([m[0], m[1]])
            all_idx = np.array(all_idx, dtype=int)
            self.tlearner._fit(X_pca[all_idx], nudge[all_idx],
                               outcome[all_idx])
        return matches

    def _X_nudge_transform(self, data):
        X, nudge = self._X_nudge(data)
        X_pca = self.pca_model.transform(X)
        return X_pca, nudge

    def predict_cate(self, data):
        X, _ = self._X_nudge_transform(data)
        return self.model.predict(X)


def greedy_match(X, nudge):
    zero_idx = np.where(nudge == 0)[0]
    one_idx = np.delete(np.arange(len(nudge)), zero_idx)

    matches = []
    available_idx = np.ones(len(one_idx), dtype=np.bool_)
    permutation = np.random.permutation(zero_idx)
    for i_perm in range(len(permutation)):
        cur_id = permutation[i_perm]
        X_comp = X[one_idx[available_idx]]
        if X_comp.shape[0] == 0:
            break
        distances = ((X_comp-X[cur_id, :])**2).sum(axis=1)
        i_min = np.argmin(distances)
        min_id = one_idx[np.where(available_idx)[0][i_min]]
        matches.append((cur_id, min_id, distances[i_min]))
    return matches


def multimatch(X, nudge):
    zero_idx = np.where(nudge == 0)[0]
    one_idx = np.delete(np.arange(len(nudge)), zero_idx)
    matches = []
    permutation = np.random.permutation(zero_idx)
    for i_perm in range(len(permutation)):
        cur_id = permutation[i_perm]
        X_comp = X[one_idx]
        distances = ((X_comp-X[cur_id, :])**2).sum(axis=1)
        i_min = np.argmin(distances)
        min_id = one_idx[i_min]
        matches.append((cur_id, min_id, distances[i_min]))
    return matches
