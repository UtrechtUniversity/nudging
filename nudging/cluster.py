import numpy as np
from sklearn.cluster import KMeans
from nudging.model import BiRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import normalize
from nudging.cate import get_cate
from nudging.evaluate_outcome import safe_spearmanr
from collections import defaultdict
import warnings


class KMeansPartitioner():
    """Partitioner that uses k-means to partition patients"""
    name = "kmeans"

    def __init__(self, dataset, n_clusters=2, eps=0.1):
        """Initialize KMeans partitioner

        Arguments
        ---------
        dataset: BaseDataset
            Dataset to partition.
        n_clusters: int
            Number or clusters to create.
        eps: float
            Tendency to create more even clusters.
        """
        # We need this model to extract the feature matrix
        model = BiRegressor(BayesianRidge())
        self.X, _nudge, _outcome = model._X_nudge_outcome(dataset.standard_df)

        # Normalize the feature matrix
        self.X = normalize(self.X, axis=0, norm="l1")*len(self.X)
        self.kmodel = KMeans(n_clusters=n_clusters)

        # Surpress warning if there are not enough unique points.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.clusters = self.kmodel.fit_predict(self.X)
        self.cluster_sizes = np.unique(self.clusters, return_counts=True)[1]
        self.n_clusters = len(self.cluster_sizes)
        self.cluster_sorted = np.argsort(-self.cluster_sizes)
        self.centers = self.kmodel.cluster_centers_
        self.eps = eps
        self.recompute()

    def recompute(self, i_clust=0):
        """Recompute the cluster centers/sizes and costs."""
        self.cluster_sizes = np.unique(self.clusters, return_counts=True)[1]

        all_costs = []
        for i_other in range(i_clust+1, self.n_clusters):
            all_costs.append(self.compute_cost(i_clust, i_other))
        if len(all_costs) == 0:
            return
        self.all_costs = np.array(all_costs).T
        self.compute_centers(i_clust)

    def compute_cost(self, i_src, i_dst):
        """Compute the cost for moving from one cluster to the other

        Arguments
        ---------
        i_src: int
            Source cluster to move from
        i_dst: int
            Destination cluster to move to

        Returns
        -------
        double: cost
            Total costs of all clusters, which includes inter-cluster distances
            and a cost depending on the sizes of the two clustsers.
        """
        src_clust = self.cluster_sorted[i_src]
        dst_clust = self.cluster_sorted[i_dst]

        # Compute the distance 
        clust_mask = self.clusters == src_clust
        src_center = self.centers[src_clust]
        dst_center = self.centers[dst_clust]
        src_cost = np.linalg.norm(self.X[clust_mask] - src_center, axis=1)**2
        dst_cost = np.linalg.norm(self.X[clust_mask] - dst_center, axis=1)**2

        clust_cost = self.cluster_sizes[dst_clust] - self.cluster_sizes[src_clust] + 1
        radius_cost = dst_cost - src_cost
        clust_cost = self.eps*clust_cost
        return radius_cost + clust_cost

    def swap_items(self, i_clust):
        min_costs = np.min(self.all_costs, axis=1)
        i_min_costs_local = np.argmin(self.all_costs, axis=1)+1+i_clust
        i_min_costs = self.cluster_sorted[i_min_costs_local]
        swap_mask = np.sort(min_costs) + 2*self.eps*np.arange(len(min_costs)) < 0
        swap_idx_local = np.argsort(min_costs)[swap_mask]
        swap_idx = (np.where(self.clusters == self.cluster_sorted[i_clust]
                             )[0])[swap_idx_local]
        self.clusters[swap_idx] = i_min_costs[swap_idx_local]

    def compute_centers(self, i_clust=0):
        for i in range(i_clust, self.n_clusters):
            clust = self.cluster_sorted[i]
            clust_mask = self.clusters == clust
            self.centers[clust] = np.mean(self.X[clust_mask], axis=0)

    def compute_energy(self):
        self.recompute()
        energy = 0
        for i in range(self.n_clusters):
            clust = self.cluster_sorted[i]
            clust_mask = self.clusters == clust
            energy += np.sum(np.linalg.norm(
                self.X[clust_mask]-self.centers[clust], axis=1)**2)
            energy += self.eps*np.sum(clust_mask)**2
        return energy

    def optimize(self, max_try=100, eps=None):
        if eps is not None:
            self.eps = eps
        last_energy = -1
        for _ in range(max_try):
            for i_clust in range(self.n_clusters-1):
                self.recompute(i_clust)
                self.swap_items(i_clust)
            new_energy = self.compute_energy()
            if last_energy == new_energy:
                break
            last_energy = new_energy
        return self.clusters

    @classmethod
    def generate(cls, dataset):
        for n_clusters in range(2, 6):
            for eps in np.linspace(0.3, 0.7, 3):
                partitioner = cls(dataset, n_clusters, eps)
                clusters = partitioner.optimize()
                for i in range(n_clusters):
                    yield clusters == i


class KSplitPartitioner():
    name = "ksplit"

    @classmethod
    def generate(cls, dataset):
        model = BiRegressor(BayesianRidge())
        X, _, _ = model._X_nudge_outcome(dataset.standard_df)
        for k_split in range(2, 6):
            for i_col in range(X.shape[1]):
                split_vals = X[:, i_col]
                sorted_idx = np.lexsort((np.random.randn(len(split_vals)), split_vals))
                idx_list = np.array_split(sorted_idx, k_split)
                for sub_idx in idx_list:
                    mask = np.zeros(len(split_vals), dtype=bool)
                    mask[sub_idx] = True
                    yield mask


class RandomPartitioner():
    name = "random"

    @classmethod
    def generate(cls, dataset):
        for k_split in range(2, 6):
            for _ in range(10):
                yield np.random.randn(len(dataset)) < 1/k_split


def compute_partition_correlation(model, dataset, *all_partitioner):
    cate_results = get_cate(model, dataset, k=5)
    true_avg_cate = defaultdict(lambda: [])
    pred_avg_cate = defaultdict(lambda: [])
    ind_cate_corr = []
    for cate, idx in cate_results:
        sub_dataset = dataset.split(idx)
        _, nudge, outcomes = model._X_nudge_outcome(sub_dataset.standard_df)
        treatment = nudge == 1
        control = nudge == 0
        for partitioner in all_partitioner:
            for split_mask in partitioner.generate(sub_dataset):
                mask_treat = np.logical_and(split_mask, treatment)
                mask_control = np.logical_and(split_mask, control)
                if np.sum(mask_treat) == 0 or np.sum(mask_control) == 0:
                    continue
                pred_avg_cate[partitioner.name].append(
                    np.mean(outcomes[mask_treat])-np.mean(outcomes[mask_control]))
                true_avg_cate[partitioner.name].append(np.mean(cate[split_mask]))
                if "cate" in sub_dataset.truth:
                    ind_cate_corr.append(safe_spearmanr(cate, sub_dataset.cate))

    all_true = []
    all_pred = []
    for name in true_avg_cate:
        all_true.extend(true_avg_cate[name])
        all_pred.extend(pred_avg_cate[name])
    true_avg_cate["all"] = all_true
    pred_avg_cate["all"] = all_pred
    sub_result = {x: safe_spearmanr(true_avg_cate[x], pred_avg_cate[x])
                  for x in true_avg_cate}
    if "cate" in dataset.truth:
        sub_result["individual"] = np.mean(ind_cate_corr)
    return sub_result
