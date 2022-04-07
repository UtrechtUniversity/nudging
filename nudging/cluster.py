import numpy as np
from sklearn.cluster import KMeans
from nudging.model import BiRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import normalize
from nudging.cate import get_cate
from scipy.stats.stats import spearmanr
from nudging.evaluate_outcome import safe_spearmanr
from collections import defaultdict


class KMeansPartitioner():
    name = "kmeans"

    def __init__(self, dataset, n_clusters=2, eps=0.1):
        model = BiRegressor(BayesianRidge())
        self.X, _nudge, _outcome = model._X_nudge_outcome(dataset.standard_df)
        self.X = normalize(self.X, axis=0, norm="l1")*len(self.X)
        self.kmodel = KMeans(n_clusters=n_clusters)
        self.clusters = self.kmodel.fit_predict(self.X)
        self.cluster_sizes = np.unique(self.clusters, return_counts=True)[1]
        self.n_clusters = len(self.cluster_sizes)
        self.cluster_sorted = np.argsort(-self.cluster_sizes)
        self.centers = self.kmodel.cluster_centers_
        self.eps = eps
        self.recompute()

    def recompute(self, i_clust=0):
        self.cluster_sizes = np.unique(self.clusters, return_counts=True)[1]

        all_costs = []
        for i_other in range(i_clust+1, self.n_clusters):
            all_costs.append(self.compute_cost(i_clust, i_other))
        if len(all_costs) == 0:
            return
        self.all_costs = np.array(all_costs).T
        self.compute_centers(i_clust)

    def compute_cost(self, i_src, i_dst):
        src_clust = self.cluster_sorted[i_src]
        dst_clust = self.cluster_sorted[i_dst]

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
        i_min_costs = self.cluster_sorted[np.argmin(self.all_costs, axis=1)+1+i_clust]
        swap_mask = np.sort(min_costs) + 2*self.eps*np.arange(len(min_costs)) < 0
        swap_idx_local = np.argsort(min_costs)[swap_mask]
        swap_idx = (np.where(self.clusters == self.cluster_sorted[i_clust])[0])[swap_idx_local]
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
            energy += np.sum(np.linalg.norm(self.X[clust_mask]-self.centers[clust], axis=1)**2)
            energy += self.eps*np.sum(clust_mask)**2
        return energy

    def optimize(self, max_try=100, eps=None):
        if eps is not None:
            self.eps = eps
#         print("before", self.compute_energy(), self.cluster_sizes)
        last_energy = -1
        for n_try in range(max_try):
            for i_clust in range(self.n_clusters-1):
                self.recompute(i_clust)
                self.swap_items(i_clust)
            new_energy = self.compute_energy()
            if last_energy == new_energy:
                break
            last_energy = new_energy
#         print("after", self.compute_energy(), self.cluster_sizes, n_try)
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
                flag = False
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
# #                     mask_treat = np.logical_and(mask, treatment)
# #                     mask_control = np.logical_and(mask, control)
# 
#         cols = list(dataset.standard_df)
#         cols.remove("outcome")
#         cols.remove("nudge")
#         pred_correct = []
#         sub_true = []
#         sub_pred = []
#         for split_var in cols:
#             split_vals = dataset.standard_df[split_var].iloc[idx].values
#             outcomes = dataset.standard_df["outcome"].iloc[idx].values
#             treatment = dataset.standard_df["nudge"].iloc[idx].values == 1
#             control = np.logical_not(treatment)
#             n_try = 0
#             while n_try < 100:
#                 cur_sub_true = []
#                 cur_sub_pred = []
# 
#                 sorted_idx = np.lexsort((np.random.randn(len(split_vals)), split_vals))
#                 idx_list = np.array_split(sorted_idx, k_split)
#                 flag = False
#                 for sub_idx in idx_list:
#                     mask = np.zeros(len(split_vals), dtype=bool)
#                     mask[sub_idx] = True
#                     mask_treat = np.logical_and(mask, treatment)
#                     mask_control = np.logical_and(mask, control)
#                     if np.sum(mask_treat) == 0 or np.sum(mask_control) == 0:
#                         flag = True
#                         break
#                     cur_sub_true.append(np.mean(outcomes[mask_treat])-np.mean(outcomes[mask_control]))
#                     cur_sub_pred.append(np.mean(cate[sub_idx]))
#                 if not flag:
#                     break
#                 n_try += 1
#             if n_try == 100:
#                 print(split_var, np.sum(mask_treat), np.sum(mask_control), len(split_vals), np.sum(treatment), np.sum(control))
#                 raise ValueError("Could not find good split after 100 tries")
#             sub_true.extend(cur_sub_true)
#             sub_pred.extend(cur_sub_pred)
#             #print(cur_sub_pred, cur_sub_true)
#             corr = spearmanr(cur_sub_pred, cur_sub_true).correlation
#             if np.isnan(corr):
#                 corr = 0
#             pred_correct.append(corr)
#             #idx_A = np.argsort(split_vals)[:len(split_vals)//2]
#             #mask_A = np.zeros(len(split_vals), dtype=bool)
#             #mask_A[idx_A] = True
#             #mask_B = np.logical_not(mask_A)
#             #q = np.quantile(split_vals, 0.5)
#             #split_1 = split_vals > q
#             #split_2 = split_vals >= q
#             #if abs(np.sum(split_1)-len(split_vals)/2) <= abs(np.sum(split_2)-len(split_vals)/2):
#             #    idx_A = split_1
#             #else:
#             #    idx_A = split_2
#             #idx_B = np.logical_not(idx_A)
#             #idx_A_treat = np.logical_and(mask_A, treatment)
#             #idx_A_control = np.logical_and(mask_A, control)
#             #idx_B_treat = np.logical_and(mask_B, treatment)
#             #idx_B_control = np.logical_and(mask_B, control)
#             #sub_cate_A = np.mean(outcomes[idx_A_treat]) - np.mean(outcomes[idx_A_control])
#             #sub_cate_B = np.mean(outcomes[idx_B_treat]) - np.mean(outcomes[idx_B_control])
#             #sub_true.extend([sub_cate_A, sub_cate_B])
#             #sub_pred.extend([np.mean(cate[mask_A]), np.mean(cate[mask_B])])
#             #pred_correct.append(
#             #    np.sign(sub_cate_A-sub_cate_B) == np.sign(np.mean(cate[mask_A])-np.mean(cate[mask_B])))
#     return spearmanr(sub_true, sub_pred).correlation, np.mean(pred_correct)




# def same_size_cluster(dataset, n_cluster=2, eps=0.1):
#     X, nudge, outcome = model._X_nudge_outcome(dataset.standard_df)
#     kmodel = KMeans(n_clusters=n_cluster)
#     clusters = Clusters(kmodel.fit_predict(X))
#     
#     def recompute_clusters():
#         cluster_sizes = np.unique(clusters, return_counts=True)[1]
#         cluster_sorted = np.argsort(-cluster_sizes)
# 
#         all_costs = []
#         for i_other in range(i_clust+1, n_cluster):
#             all_costs.append(compute_cost(i_clust, i_other))
#         if len(all_costs) == 0:
#             return
#         return np.array(all_costs).T
# 
#     def compute_cost(i_clust, i_other):
#         clust_mask = clusters == cluster_sorted[i_clust]
#         center = kmodel.cluster_centers_[cluster_sorted[i_other]]
#         dist_cost = np.linalg.norm(X[clust_mask] - center, axis=1)
#         clust_cost = cluster_sizes[cluster_sorted[i_other]] - cluster_sizes[cluster_sorted[i_clust]] + 1
#         print("clust", clust_cost)
#         return dist_cost + eps*clust_cost
# 
#     for i_clust in range(n_cluster):
#         clusters.recompute_clusters(i_clust)
#         clusters.swap_items(i_clust)
#         all_costs = clusters.all_costs
#         min_costs = np.min(all_costs, axis=1)
#         i_min_costs = np.argmin(all_costs, axis=1)
#         swap_mask = np.sort(min_costs) + 2*eps*np.arange(len(min_costs)) < 0
#         swap_idx_local = np.argsort(min_costs)[swap_mask]
#         swap_idx = (np.where(clusters == cluster_sorted[i_clust])[0])[swap_idx_local]
#         self.clusters.clusters[swap_idx] = i_other    
#         all_costs = recompute_clusters()
# 
#     return clusters
#     zero_clust = clusters == 0
#     one_clust = clusters == 1
#     d_zero_one = X[zero_clust] - kmodel.cluster_centers_[1]
#     d_zero_zero = X[zero_clust] - kmodel.cluster_centers_[0]
#     cost = np.linalg.norm(d_zero_one, axis=1)-np.linalg.norm(d_zero_zero, axis=1)
    