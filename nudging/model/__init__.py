from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression, SGDRegressor
from sklearn.linear_model import ARDRegression, BayesianRidge, ElasticNet
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from .probmodel import ProbModel
from .biregressor import BiRegressor
from .monoregressor import MonoRegressor
from .xregressor import XRegressor
from .mdm import MDMModel, MultiMDMModel
from .pca import PCAModel


__all__ = ['ProbModel', 'BiRegressor', 'MonoRegressor', 'XRegressor',
           'MDMModel', 'MultMDMModel', 'PCAModel']

regressors = {
    "gauss_process": GaussianProcessRegressor,
    "ridge": Ridge,
    "linear": LinearRegression,
    "logistic": LogisticRegression,
    "nb": GaussianNB,
    "sgd": SGDRegressor,
    "elasticnet": ElasticNet,
    "ard": ARDRegression,
    "bayesian_ridge": BayesianRidge,
    "knn": KNeighborsRegressor,
    "mlp": MLPRegressor,
    "svm": SVR,
    "decision_tree": DecisionTreeRegressor,
    "extra_tree": ExtraTreeRegressor,
}

learners = {
    "s-learner": MonoRegressor,
    "t-learner": BiRegressor,
    "x-learner": XRegressor,
    "probabilistic": ProbModel
}