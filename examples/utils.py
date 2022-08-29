from nudging.simulation import generate_datasets
from nudging.model import BiRegressor
from nudging.model import ProbModel, MDMModel, PCAModel
import numpy as np
from nudging.partition import KSplitPartitioner, compute_partition_correlation, KMeansPartitioner
import pickle as pkl
from pathlib import Path
from tqdm import tqdm
from nudging.model.monoregressor import MonoRegressor
from nudging.model.xregressor import XRegressor
from nudging.evaluate_outcome import evaluate_performance

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression, ElasticNet
from sklearn.linear_model import ARDRegression, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from nudging.parallel import execute_parallel
from multiprocessing import Pool


def get_datasets(n=500, base_cache_dir=None):
    if base_cache_dir is None:
        base_cache_dir = Path(Path.home(), "cache", "nudging")
    cache_dir = base_cache_dir
    cache_dir.mkdir(exist_ok=True)
    dataset_fp = Path(cache_dir, f"datasets{n}.pkl")
    if dataset_fp.is_file():
        with open(dataset_fp, "rb") as f:
            sim_datasets = pkl.load(f)
    else:
        np.random.seed(1298734)
        sim_datasets = generate_datasets(n)
        with open(dataset_fp, "wb") as f:
            pkl.dump(sim_datasets, f)
    return sim_datasets


def get_correlations(datasets, base_cache_dir=None):
    if base_cache_dir is None:
        base_cache_dir = Path(Path.home(), "cache", "nudging")

    n = len(datasets)
    cache_dir = Path(base_cache_dir, "partition")
    cache_dir.mkdir(exist_ok=True)
    correlation_fp = Path(cache_dir, f"correlation{n}.pkl")
    if correlation_fp.is_file():
        with open(correlation_fp, "rb") as f:
            return pkl.load(f)

    np.random.seed(28374582)
    models = get_models("default")
    all_results = {x: {} for x in models}
    with tqdm(total=len(datasets)*len(models)) as pbar:
        for model_name, model in models.items():
            for dataset in datasets:
                res = compute_partition_correlation(model, dataset, KSplitPartitioner, KMeansPartitioner)
                for part_name in res:
                    try:
                        all_results[model_name][part_name].append(res[part_name])
                    except KeyError:
                        all_results[model_name][part_name] = [res[part_name]]
                pbar.update(1)

    with open(correlation_fp, "wb") as f:
        pkl.dump(all_results, f)
    return all_results


def get_models(key="default"):
    if key == "default":
        return {
            "mdm": MDMModel(ARDRegression()),
            "prob_log": ProbModel(LogisticRegression()),
            "prob_bay": ProbModel(ARDRegression()),
            "t-learner": BiRegressor(ARDRegression()),
            "pca": PCAModel(ARDRegression()),
        }

    meta_conversion = {
        "s-learner": MonoRegressor,
        "t-learner": BiRegressor,
        "x-learner": XRegressor,
    }

    regressors = {
        "gauss_process": GaussianProcessRegressor,
        "ridge": Ridge,
        "linear": LinearRegression,
        "elasticnet": ElasticNet,
        "ard": ARDRegression,
        "bayesian_ridge": BayesianRidge,
        "knn": KNeighborsRegressor,
        "mlp": MLPRegressor,
        "svm": SVR,
        "decision_tree": DecisionTreeRegressor,
        "extra_tree": ExtraTreeRegressor,
    }

    if key in meta_conversion:
        return {key + " " + regressor_name: meta_conversion[key](regressor_class())
                for regressor_name, regressor_class in regressors.items()}

    if key in regressors:
        return {meta_name + " " + key: meta_class(regressors[key]())
                for meta_name, meta_class in meta_conversion.items()}

    if key == "meta":
        models = {}
        for meta_name, meta_class in meta_conversion.items():
            for regressor_name, regressor_class in regressors.items():
                models[meta_name + " " + regressor_name] = meta_class(regressor_class())
        return models

    raise ValueError(f"Unknown model name '{key}'")

@ignore_warnings(category=ConvergenceWarning)
# def compute_learners(datasets, model, i_data):#, regressor_name, learner_type):
def compute_learners(kwargs):
    
#     data = datasets[i_data]
    model, data = kwargs["model"], kwargs["data"]
    np.random.seed(23847234)
    cate_perf, out_perf = evaluate_performance(model, data)
    return {
#         "regressor_name": regressor_name,
#         "learner_type": learner_type,
        "cate_perf": cate_perf,
        "out_perf": out_perf,
    }


def run_model(model, datasets, pbar):
    jobs =  [{"model": model, "i_data": i}
             for i, _data in enumerate(datasets)]
    return execute_parallel(jobs, compute_learners, args=[datasets], progress_bar=pbar, n_jobs=1)



def run_model_2(model, datasets, pbar):
    jobs =  [{"model": model, "data": data}
             for i, data in enumerate(datasets)]
    pool = Pool(processes=10)
    results = pool.map(compute_learners, jobs)
    pbar.update(len(jobs))
    return results

# def aggregate_results(results):
#     if isinstance(results, dict):
#         return results
#     agg_results = {}
#     for model_name in regressors:
#         agg_results[model_name] = {}
#         for learner_type in learner_dict:
#             cate_perf = np.array([x["cate_perf"] for x in results 
#                                   if x["model_name"] == model_name and x["learner_type"] == learner_type])
#             out_perf = np.array([x["out_perf"] for x in results 
#                                  if x["model_name"] == model_name and x["learner_type"] == learner_type])
#             agg_results[model_name][learner_type] = {"cate_perf": cate_perf.flatten(), "out_perf": out_perf.flatten()}
#     return agg_results


def get_cate_outcomes(datasets, models="meta", base_cache_dir=None):
    if base_cache_dir is None:
        base_cache_dir = Path(Path.home(), "cache", "nudging")

    n = len(datasets)
    cache_dir = Path(base_cache_dir, "cate", str(n))
    cache_dir.mkdir(exist_ok=True, parents=True)

    if isinstance(models, str):
        models = get_models(models)
    correlation_fp = Path(cache_dir, f"correlation{n}.pkl")
    if correlation_fp.is_file():
        with open(correlation_fp, "rb") as f:
            return pkl.load(f)

    all_results = {}
    with tqdm(total=len(datasets)*len(models)) as pbar:
        for model_name, model in models.items():
            cate_fp = Path(cache_dir, model_name + ".pkl")
            if cate_fp.is_file():
                with open(cate_fp, "rb") as f:
                    all_results[model_name] = pkl.load(f)
#                     for name in results:
#                         results[name].extend(new_results[name])
                    pbar.update(len(datasets))
                    continue
            cur_results = run_model_2(model, datasets, pbar)
            new_results = {
                name: np.array([np.mean(x[name]) for x in cur_results])
                for name in ["cate_perf", "out_perf"]
            }
#             for name in results:
#                 results[name].extend(new_results[name])
            all_results[model_name] = new_results
            with open(cate_fp, "wb") as f:
                pkl.dump(new_results, f)
            #pbar.update(len(datasets))
    return all_results


def parameter_descriptions():
    return {
        "noise_frac": "Proportion noise",
        "n_samples": "Number of people",
        "control_precision": "Control group response heterogeneity",
        "control_unique": "Control/treatment group response similarity",
        "n_features": "Number of features",
        "avg_correlation": "Average correlation between features/outcome",
        "balance": "Proportion treatment group",
        "n_rescale": "Number of categorical variables",
    }
