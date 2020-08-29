import os
from scipy.stats import truncnorm
from functools import partial
import numpy as np
import pandas as pd
import functools

CLASS_DBS_PATH = "./datasets/"
META_DBS_PATH = "./ClassificationAllMetaFeatures.csv"

RANDOM_SEED = 1332
MAX_BATCH_SIZE = 100  # Max batch size for optimization in RBoost

LOAD_ALL = True

USE_RBOOST = True

OUR_MODEL = "RBoost" if USE_RBOOST else "ELPBoost"
COMP_MODEL = "LGBoost"
NOT_PREFIX = "not_label_"

STAT_CHOSEN_METRIC = 'accuracy'

WEIGHTED_METRICS = True
WORKING_DIR = "./output"

EVAL_FOLDS = 10
HPT_FOLDS = 3
RANDOM_CV_ITER = 10

RESULTS_CSV_PATH = os.path.join(WORKING_DIR, "results.csv")
RESULT_CSV_PATH = lambda db_name: os.path.join(WORKING_DIR, "%s_results.csv" % db_name)

MODELS_LIST = [OUR_MODEL, COMP_MODEL]

P_THRESH = 0.05

figure_num = 1
PLOTS_FORMAT = 'png'

PLOTS_DIR = os.path.join(WORKING_DIR, "plots/")
MODELS_DIR = os.path.join(WORKING_DIR, "models")

IMPORTANCE_TYPES = ['weight', 'gain', 'cover']

IMPORTANCE_DIR = 'importance'
SHAP_DIR = 'shap'

PLOT_TOP_FEATURES = 10

params_ranges = {
    "model__n_estimators": (5, 10000),
    "model__learning_rate": (1e-10, 1),
    "model__gamma": (0, 2),
    "model__max_depth": (0, 100),
    "model__lambda": (0, 100),
    "model__reg_lambda": (0, 100),
    "model__alpha": (0, 100),
    "model__reg_alpha": (0, 100),
    "model__colsample_bytree": (0, 5),
    "model__num_leaves": (0, 10000),
    "model__subsample": (0, 1)
}


def get_truncated_normal(mean=0., sd=1., low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def get_integer_truncated_normal(*args, **kwargs):
    res = get_truncated_normal(*args, **kwargs)

    def integer_wrap(*args, f=None, **kwargs):
        return int(np.round(f(*args, **kwargs)))

    res.rvs = partial(integer_wrap, f=res.rvs)
    return res


explored_n_estimators = get_integer_truncated_normal(mean=225, sd=50, low=params_ranges['model__n_estimators'][0],
                                                     upp=params_ranges['model__n_estimators'][1])

explored_learning_rate = get_truncated_normal(mean=0.12, sd=0.1, low=params_ranges['model__learning_rate'][0],
                                              upp=params_ranges['model__learning_rate'][1])
explored_max_depth = get_integer_truncated_normal(mean=10, sd=5, low=params_ranges['model__max_depth'][0],
                                                  upp=params_ranges['model__max_depth'][1])

explored_num_leaves = get_integer_truncated_normal(mean=28, sd=10, low=params_ranges['model__num_leaves'][0],
                                                   upp=params_ranges['model__num_leaves'][1])

explored_lambda = get_truncated_normal(mean=22, sd=15, low=params_ranges['model__lambda'][0],
                                       upp=params_ranges['model__lambda'][1])


comp_model_params = {
    "estimator__model__n_estimators": explored_n_estimators,
    "estimator__model__learning_rate": explored_learning_rate,
    "estimator__model__max_depth": explored_max_depth,
    "estimator__model__reg_lambda": explored_lambda,
    "estimator__model__num_leaves": explored_num_leaves,
    "estimator__model__objective": ['binary'],
}

y_values_encoding = {'tested_negative': -1, 'schizophrenic': 1, 'Lewis': -1, 'sn': -1, 'cn': 1, 'excellent': 2,
                     'not_suitable': 0, 'Tile': -1, 'republican': -1, 'democrat': 1, 'ok': 1, 'dead': -1,
                     'non-schizophrenic': -1, 'Holyfield': 1, 'bad': -1, 'tested_positive': 1, 'P': 1, 'good': 1,
                     'alive': 1, 'N': -1, 'Insulation': 1}
X_nan_values = {'labor.csv': ['?'], 'braziltourism.csv': ['?']}
X_func_replacement = {'braziltourism.csv': functools.partial(pd.to_numeric, errors='ignore')}
