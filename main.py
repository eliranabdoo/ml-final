# %% [code] {"execution":{"iopub.execute_input":"2020-08-26T15:31:59.626531Z","iopub.status.busy":"2020-08-26T15:31:59.625771Z","iopub.status.idle":"2020-08-26T15:31:59.631466Z","shell.execute_reply":"2020-08-26T15:31:59.630686Z"},"papermill":{"duration":0.024011,"end_time":"2020-08-26T15:31:59.631606","exception":false,"start_time":"2020-08-26T15:31:59.607595","status":"completed"},"tags":[]}
# https://users.soe.ucsc.edu/~manfred/pubs/C83.pdf  EPLBoost
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3173974/  RBoost

# %% [markdown] {"papermill":{"duration":0.010568,"end_time":"2020-08-26T15:31:59.653755","exception":false,"start_time":"2020-08-26T15:31:59.643187","status":"completed"},"tags":[]}
# # Imports

# %% [code] {"execution":{"iopub.execute_input":"2020-08-26T15:31:59.690455Z","iopub.status.busy":"2020-08-26T15:31:59.689503Z","iopub.status.idle":"2020-08-26T15:32:49.385992Z","shell.execute_reply":"2020-08-26T15:32:49.385126Z"},"papermill":{"duration":49.721794,"end_time":"2020-08-26T15:32:49.386130","exception":false,"start_time":"2020-08-26T15:31:59.664336","status":"completed"},"tags":[]}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
from numpy import linalg
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import os
import sys

#os.system("python -m pip install xgboost==1.0.0")
import xgboost as xgb

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import precision_score, roc_auc_score, roc_curve, average_precision_score, balanced_accuracy_score, \
    accuracy_score, make_scorer, auc
from sklearn.model_selection import RandomizedSearchCV, LeaveOneOut
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.multiclass import OneVsRestClassifier
from datetime import datetime

import matplotlib.pyplot as plt

import time

import functools
from functools import partial

#os.system("python -m pip install cvxopt")

from cvxopt import modeling, matrix, solvers
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer

from scipy.stats import friedmanchisquare
from scipy.stats import truncnorm

#os.system("python -m pip install scikit_posthocs")
from scikit_posthocs import posthoc_nemenyi_friedman

import dill

import csv

import math

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# %% [markdown] {"papermill":{"duration":0.010151,"end_time":"2020-08-26T15:32:49.407244","exception":false,"start_time":"2020-08-26T15:32:49.397093","status":"completed"},"tags":[]}
# # Constants

# %% [code] {"execution":{"iopub.execute_input":"2020-08-26T15:32:49.449736Z","iopub.status.busy":"2020-08-26T15:32:49.444455Z","iopub.status.idle":"2020-08-26T15:32:49.466488Z","shell.execute_reply":"2020-08-26T15:32:49.467123Z"},"papermill":{"duration":0.047869,"end_time":"2020-08-26T15:32:49.467290","exception":false,"start_time":"2020-08-26T15:32:49.419421","status":"completed"},"tags":[]}
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


# %% [markdown] {"papermill":{"duration":0.010247,"end_time":"2020-08-26T15:32:49.488170","exception":false,"start_time":"2020-08-26T15:32:49.477923","status":"completed"},"tags":[]}
# # ELPBoost

# %% [code] {"execution":{"iopub.execute_input":"2020-08-26T15:32:49.520076Z","iopub.status.busy":"2020-08-26T15:32:49.519116Z","iopub.status.idle":"2020-08-26T15:32:49.598663Z","shell.execute_reply":"2020-08-26T15:32:49.599232Z"},"papermill":{"duration":0.100657,"end_time":"2020-08-26T15:32:49.599405","exception":false,"start_time":"2020-08-26T15:32:49.498748","status":"completed"},"tags":[]}
# loosly based on https://github.com/YuxinSun/LPBoost-Using-String-and-Fisher-Features

class ELPBoost(BaseEstimator):
    """
    Entropy Regularized Boost
    -------
    weak_learners: A list of weak learners where:
                   a weak learner h gets as a parameter an entry x: array_like, shape (1, n_features)
                   and return the predicted class (1, -1)
    kappa: float greater or equal to 0, exclusive. optional
        Constant factor which penalizes the slack variables
    threshold: float, optional
        Threshold of feature selection. Features with weights below threshold would be discarded.
    T: int, optional
        Maximum iteration of LPBoost
    reg: float, optinal
        The regularization parameter. Corresponds to 1/eta in the paper.
    Attributes
    -------
    converged: bool
        True when convergence reached in fit() and fit_transform().
    u: array_like, shape (n_samples, )
        Misclassification cost
    w: array_like, shape (n_selected_features, )
        Weights of selected features, such features are selected because corresponding weights are lower than threshold.
    beta: float
        beta in LPBoost
    idx: list of integers
        Indices of selected features
    """
    POSITIVE_CLASS = 1
    NEGATIVE_CLASS = 0

    def __init__(self, kappa=1, threshold=10 ** -3, T=1000, reg=0.01, verbosa=False, silent=False):
        self.kappa = kappa
        self.threshold = threshold
        self.T = T
        self.reg = reg
        self.verbose = verbose
        self.converged = False
        self.H = None
        self.w = None
        self.slacks = None
        self.d_0 = None
        self.u = None
        self.silent = silent
        self.H_weak = []

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)  # only difference

    def predict_proba(self, X):
        """
        Predict probabilities for positive and negative labels for each sample in X.
        Left and Right correspond to Negative and Positive.
        Scores in [-1, 1] are linearly transformed to probabilities in [0, 1]
        """
        res = (np.dot(self.w.T,
                      np.squeeze([np.apply_along_axis(h, 1, X) for h in self.H])) + 1.0) / 2.0  # score is in [-1, 1]
        res = np.vstack([1 - res, res]).T
        return np.array(res)

    def predict(self, X):
        """
        Predict binary labels for each sample in X, using a 0.5 threshold.
        Outputs a POSITIVE_CLASS/NEGATIVE_CLASS labels vector.
        """
        res = self.predict_proba(X)
        positive_mask = res >= 0.5
        negative_mask = res < 0.5
        res[positive_mask] = self.POSITIVE_CLASS
        res[negative_mask] = self.NEGATIVE_CLASS
        return res

    def _update_samples_weight(self):
        """
        Solve the soft margin dual problem and return the optimal samples distribution of equation (4) in the paper.
        Also updates w and slacks.
        """
        m, n = 0, self.u.shape[0]
        T = self.u.shape[1]
        N = n + T
        d_0 = matrix(self.d_0.reshape(n, 1))

        # Linear Inequallity Constraints,  Gx <= h
        G = matrix(-1 * np.eye(N))
        h = matrix(np.zeros(shape=(N, 1)))

        # Linear Equality Constraints, Ax = b
        A = matrix(np.concatenate((np.ones(shape=(T, 1)), np.zeros(shape=(n, 1))), axis=0).T)
        b = matrix(1.0)

        def F(x=None, z=None):
            if x is None: return 0, matrix(0.5, (N, 1))
            w = x[:T, :]
            phi = x[T:, :]
            reg_inv = 1 / self.reg

            weighted_u = np.dot(self.u, w)  # n x 1
            scores = -1 * reg_inv * (weighted_u + phi)  # n x 1

            # Numeric correction
            scores -= max(scores)

            # Auxilliaries
            weighted_scores_exp = np.multiply(d_0, np.exp(scores))
            sum_weighted_scores_exp = np.sum(weighted_scores_exp)
            sum_weighted_scores_exp_square = sum_weighted_scores_exp ** 2
            squared_weighted_scores_exp = np.square(weighted_scores_exp)
            weighted_scores_exp_mults = np.dot(weighted_scores_exp, weighted_scores_exp.T)
            uw_mult = np.multiply(self.u, weighted_scores_exp)
            uw_mult_sum = np.sum(np.multiply(self.u, weighted_scores_exp), axis=0)

            f = self.reg * np.log(sum_weighted_scores_exp) + self.kappa * np.sum(phi)  # f(x)

            dfdw = -1 * uw_mult_sum.T / sum_weighted_scores_exp
            dfdphi = (-1 * weighted_scores_exp / sum_weighted_scores_exp) + self.kappa
            Df = np.concatenate((dfdw, dfdphi), axis=0)  # Gradient

            mf = matrix(f)
            mDf = matrix(Df.T)
            if z is None:
                return mf, mDf
            # Assumes d_0 is uniform
            H = np.zeros(shape=(N, N))  # Hessian
            dfdwiwi = np.zeros(shape=(T, 1))
            dfdphiiphij = -1 * reg_inv * (np.tril(weighted_scores_exp_mults)) / sum_weighted_scores_exp_square
            dfdphiiphii = reg_inv * (np.multiply(weighted_scores_exp,
                                                 sum_weighted_scores_exp - weighted_scores_exp) / sum_weighted_scores_exp_square)
            # dfdwiwj, dfwiphij are zeros
            dfdphiiwj = reg_inv * ((
                                           uw_mult * sum_weighted_scores_exp - weighted_scores_exp * uw_mult_sum) / sum_weighted_scores_exp_square)

            H[T:, T:] = dfdphiiphij
            H[T:, :T] = dfdphiiwj
            H_diagonal = np.concatenate((dfdwiwi, dfdphiiphii), axis=0)
            np.fill_diagonal(H, H_diagonal)

            mH = matrix(z[0] * H)
            return mf, mDf, mH

        prev_w = self.w
        prev_slacks = self.slacks
        try:
            wphi = solvers.cp(F, G=G, h=h, A=A, b=b)['x']
            self.w = wphi[:T, :]
            self.slacks = wphi[T:, :]
        except Exception as e:
            self.slacks = prev_slacks
            self.w = prev_w
            try:
                self.w = np.concatenate((self.w, [[1 / (len(self.w) + 1)]]), axis=0)
            except:
                self.w = np.concatenate((self.w, [1 / (len(self.w) + 1)]), axis=0)
            self.w /= np.sum(self.w)
        scores = ((-1 / self.reg) * np.squeeze(np.asarray(np.dot(self.u, self.w) + self.slacks))) + np.log(
            self.d_0)  # Update according to Equation (6)
        return self.softmax(scores)

    def _fitString(self, X, y):
        """
        Perform ELPBoost model fitting on X,y pair.
        """
        n_samples = y.size

        self.d_0 = np.ones(n_samples) / n_samples
        d = self.d_0.copy()

        y_t = y[:, np.newaxis]

        # A matrix (n_samples * n_weak_learners) where the i,j entry is h_j(x_i)
        # (the prediction of the j_th weak learner on that enrty)
        P = np.matrix(np.squeeze([np.apply_along_axis(h, 1, X) for h in self.H_weak])).T
        # A matrix (n_samples * n_weak_learners) where the i,j entry is y_i*h_j(x_i)
        # (the true label of the i_th data entry * the prediction of the j_th weak learner on that enrty)
        M = np.multiply(P, y_t)

        # Initializing the weak learners vector with 2 random one to prevent always
        # choosing the same weak learner
        random.seed(RANDOM_SEED)
        init_wl_idx = random.sample(range(len(self.H_weak)), 2)
        if self.verbose:
            print("Initialized H with the weak learners %s" % init_wl_idx)
        self.H = [self.H_weak[i] for i in init_wl_idx]
        self.u = np.matrix(M[:, init_wl_idx])
        self.w = np.random.uniform(size=len(self.H))
        chosen_wls = init_wl_idx

        for t in range(3, self.T + 1):
            if not self.silent:
                print('=== Iteration %d ===' % (t - 2))
            # A vector (n_weak_learners) where an entry j is the weighted (by d) sum of correct and incorrect
            # predictions of the corresponding j-th weak learner
            h_score = np.squeeze(np.dot(d, M).T).T

            # Add the current best weak learner
            best_weak_idx = np.argmax(h_score)
            chosen_wls.append(best_weak_idx)
            self.H.append(self.H_weak[best_weak_idx])
            # Add a cloumn of the t_th added weak learner correctness to u (n_samples * t)
            self.u = np.hstack((self.u, M[:, best_weak_idx]))

            # update the samples weight using the Riemannian distance
            solvers.options['show_progress'] = self.verbose
            d = self._update_samples_weight()
            # print("Current D is %s" % str(d))

            if self.verbose:
                print('chose weak learner %d, with score: %10.6f.' % (best_weak_idx, h_score[best_weak_idx]))
                print('calculated w min %10.6f w max %10.6f' % (np.min(self.w), np.max(self.w)))
                print('calculated d min %10.6f d max %10.6f' % (np.min(d), np.max(d)))

        # self.w = self.w[np.where(self.w >= self.threshold)]
        if self.verbose:
            print("Final w is %s" % self.w)
            print("Chose the weak learners %s" % chosen_wls)

    @staticmethod
    def generate_weak_learner(X, y, feature):
        sampled_X = X[feature]
        feature_idx = list(X.columns).index(feature)
        sorted_values = sampled_X.sort_values()
        consecutive_means = sorted_values.rolling(2, min_periods=1).mean()
        thresholds = list(set(consecutive_means.values))
        thresholds.append(sorted_values.max())

        max_acc = -np.inf
        best_threshold = thresholds[0]
        best_threshold_idx = 0

        for idx, thresh in enumerate(thresholds):
            y_pred = (sampled_X >= thresh).astype(int).replace(0, -1)
            curr_acc = (y.values * y_pred.values.T).sum()
            if curr_acc > max_acc:
                best_threshold, best_threshold_idx = thresh, idx
                max_acc = curr_acc
        return lambda x: 1 if x[feature_idx] >= best_threshold else -1

    @staticmethod
    def generate_H(X, y, size_H=None):
        if type(X) is not pd.DataFrame:
            X = pd.DataFrame(data=X)
        if type(y) is not pd.Series:
            y = pd.DataFrame(data=y)
        if size_H is None:
            size_H = len(X.columns)
        size_H = min(size_H, len(X.columns))
        sampled_X = X.sample(n=size_H, axis=1, replace=False)
        return [ELPBoost.generate_weak_learner(X, y, feature) for feature in sampled_X.columns]

    def fit(self, X, y):
        """ Predicts for pair X,y where y is in {0, 1}
        """
        positive_mask = y == self.POSITIVE_CLASS
        negative_mask = y == self.NEGATIVE_CLASS
        y[positive_mask] = 1
        y[negative_mask] = -1
        self.H_weak = ELPBoost.generate_H(X, y)
        return self._fitString(X, y)


# %% [markdown] {"papermill":{"duration":0.010134,"end_time":"2020-08-26T15:32:49.620407","exception":false,"start_time":"2020-08-26T15:32:49.610273","status":"completed"},"tags":[]}
# # RBoost

# %% [code] {"execution":{"iopub.execute_input":"2020-08-26T15:32:49.688586Z","iopub.status.busy":"2020-08-26T15:32:49.651695Z","iopub.status.idle":"2020-08-26T15:32:49.717260Z","shell.execute_reply":"2020-08-26T15:32:49.716479Z"},"papermill":{"duration":0.086264,"end_time":"2020-08-26T15:32:49.717385","exception":false,"start_time":"2020-08-26T15:32:49.631121","status":"completed"},"tags":[]}
# loosly based on https://github.com/YuxinSun/LPBoost-Using-String-and-Fisher-Features

class RBoost(BaseEstimator):
    """
    Linear programming boosting (LPBoost) model
    Representation of a LPBoost model.
    This class allows for selecting weak learners (features) from an explicit hypothesis space.
    Parameters
    -------
    weak_learners: A list of weak learners where:
                   a weak learner h gets as a parameter an entry x: array_like, shape (1, n_features)
                   and return the predicted class (1, -1)
    kappa: float greater or equal to 0, exclusive. optional
        Constant factor which penalizes the slack variables
    threshold: float, optional
        Threshold of feature selection. Features with weights below threshold would be discarded.
    n_iter: int, optional
        Maximum iteration of LPBoost
    verbose: int, default 0
        Enable verbose output. If greater than 0 then it prints the iterations in fit() and fit_transform().
    Attributes
    -------
    converged: bool
        True when convergence reached in fit() and fit_transform().
    u: array_like, shape (n_samples, )
        Misclassification cost
    w: array_like, shape (n_selected_features, )
        Weights of selected features, such features are selected because corresponding weights are lower than threshold.
    beta: float
        beta in LPBoost
    idx: list of integers
        Indices of selected features
    """
    POSITIVE_CLASS = 1
    NEGATIVE_CLASS = 0

    def __init__(self, kappa=1, threshold=10 ** -3, T=1000, batch_size_ratio=1, max_batch_size=MAX_BATCH_SIZE,
                 verbose=False, reg=0.01, silent=False):
        self.kappa = kappa
        self.threshold = threshold
        self.T = T
        self.batch_size_ratio = batch_size_ratio
        self.max_batch_size = max_batch_size
        self.reg = reg
        self.verbose = verbose
        self.converged = False
        self.H = None
        self.w = None
        self.d_0 = None
        self.u = None
        self.silent = silent
        self.H_weak = []

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)  # only difference

    def predict_proba(self, X):
        res = (np.dot(self.w.T,
                      np.squeeze([np.apply_along_axis(h, 1, X) for h in self.H])) + 1.0) / 2.0  # score is in [-1, 1]
        res = np.array(np.vstack([1 - res, res]).T)
        return res

    def predict(self, X):
        """
        Predict labels given a data matrix by LPBoost classifier: sign(data_transformed * a)
        Parameters
        -------
        :param X: array_like, shape (n_samples, n_selected_features)
            Data matrix to be predicted
        Returns
        -------
        :return: array_like, shape (n_samples, )
            Predicted labels
        """
        # return a prediction score vector (len(X))

        # TODO CHECK WHY UNUSED
        res = self.predict_proba(X)
        positive_mask = res >= 0.5
        negative_mask = res < 0.5
        res[positive_mask] = self.POSITIVE_CLASS
        res[negative_mask] = self.NEGATIVE_CLASS
        return res

    def _update_samples_weight(self, curr_d):
        # A vector (n_samples) where the i_th entry is the weighted (by w) sum of i_th row in u, squared
        uw = np.dot(self.u, self.w)
        uw_square = np.square(uw)

        d = np.squeeze(np.asarray(self.d_0 / uw_square.T))
        d = self.softmax(d)
        dot_prod = np.dot(np.sqrt(self.d_0), np.sqrt(d))

        reim_delta = np.arccos(1 if np.isclose([dot_prod], [1]) else dot_prod)
        reg_factor = math.exp(-1 * self.reg * reim_delta)

        # reg_factor = (self.reg ** 2) * 0.25 * (1/1-np.dot(self.d_0, d)**2)
        d *= reg_factor

        # return np.squeeze(np.asarray(d/np.sum(d)))
        return self.softmax(d)

    def _update_learners_weights(self, t, samples_dist):
        """ Linear programming optimisation for LPBoost
        Parameters
        -------
        :param z: array_like, shape (n_iterations, n_samples)
            transposed hypothesis space in current iteration
        :param y: array_like, shape (n_samples, )
            desired labels for classification
        :param D: float
            optimisation parameter, practically D = 1/(n_samples, nu)
        Return
        -------
        :return d: array_like, shape (n_samples, )
            misclassification cost
        :return beta: float
            beta in LPBoost
        :return c4.multiplier.value: array_like, shape (n_features, )
            weights of weak learners
        """
        n = len(samples_dist)
        batch_size = min(self.max_batch_size, int(n * self.batch_size_ratio))
        batch_indices = np.random.choice(a=list(range(n)), replace=False, p=samples_dist, size=batch_size)

        # Weak learners weights - what we need to find
        w = modeling.variable(t, 'w')

        # Slack variables
        # zetas = {int(i): modeling.variable(1, 'zeta_%d' % int(i)) for i in batch_indices}
        zetas = modeling.variable(batch_size, 'zetas')

        # Margin
        rho = modeling.variable(1, 'rho')

        # Constraints
        c1 = (w >= 0)
        c2 = (sum(w) == 1)
        c_slacks = (zetas >= 0)
        c_soft_margins = [(modeling.dot(matrix(self.u[sample_idx].astype(float).T), w) >= (rho - zetas[int(idx)])) for
                          idx, sample_idx in enumerate(batch_indices)]

        # Solve optimisation problems
        lp = modeling.op(-(rho - self.kappa * modeling.sum(zetas)), [c1, c2, c_slacks] + c_soft_margins)
        solvers.options['show_progress'] = False
        lp.solve()

        return w.value

    def _fitString(self, X, y):
        """
        Perform LPBoost on string features. Usually a l2 normalisation is performed. If the hypothesis space contains
        positive/ negative features only, then the space needs to be duplicated by its additive inverse. This is to
        ensure the performance of LPBoost.
        Parameters
        -------
        :param X: array_like, shape (n_samples, n_features)
            Data matrix of explicit features
        :param y: array_like, shape (n_samples,)
            Desired labels
        Returns
        -------
        :return:
        """
        self.n_samples = y.size

        self.d_0 = np.ones(self.n_samples) / self.n_samples
        d = self.d_0.copy()

        y_t = y[:, np.newaxis]

        # A matrix (n_samples * n_weak_learners) where the i,j entry is h_j(x_i)
        # (the prediction of the j_th weak learner on that enrty)
        P = np.matrix(np.squeeze([np.apply_along_axis(h, 1, X) for h in self.H_weak])).T
        # A matrix (n_samples * n_weak_learners) where the i,j entry is y_i*h_j(x_i)
        # (the true label of the i_th data entry * the prediction of the j_th weak learner on that enrty)
        M = np.multiply(P, y_t)

        # Initializing the weak learners vector with 2 random one to prevent always
        # choosing the same weak learner
        random.seed(RANDOM_SEED)
        init_wl_idx = random.sample(range(len(self.H_weak)), 2)
        if self.verbose:
            print("Initialized H with the weak learners %s" % init_wl_idx)
        self.H = [self.H_weak[i] for i in init_wl_idx]
        self.u = np.matrix(M[:, init_wl_idx])
        chosen_wls = init_wl_idx

        for t in range(3, self.T + 1):
            if not self.silent:
                print('=== %s === Iteration %d ===' % (datetime.now(), (t - 2)))
            # A vector (n_weak_learners) where an entry j is the weighted (by d) sum of correct and incorrect
            # predictions of the corresponding j-th weak learner
            h_score = np.squeeze(np.dot(d, M).T).T

            # Add the current best weak learner
            best_weak_idx = np.argmax(h_score)
            chosen_wls.append(best_weak_idx)
            self.H.append(self.H_weak[best_weak_idx])
            # Add a cloumn of the t_th added weak learner correctness to u (n_samples * t)
            self.u = np.hstack((self.u, M[:, best_weak_idx]))

            # update the weight of the weak learners in the strong learner ouput
            self.w = np.matrix(self._update_learners_weights(t, d))

            # update the samples weight using the Riemannian distance
            d = self._update_samples_weight(d)

            if self.verbose:
                print('chose weak learner %d, with score: %10.6f.' % (best_weak_idx, h_score[best_weak_idx]))
                print('calculated w min %10.6f w max %10.6f' % (np.min(self.w), np.max(self.w)))
                print('calculated d min %10.6f d max %10.6f' % (np.min(d), np.max(d)))

        if np.abs(sum(self.w) - 1.0) > (1e-5):
            self.w = self.softmax(self.w)
        # self.w = self.w[np.where(self.w >= self.threshold)]
        if self.verbose:
            print("Final w is %s" % self.w)
            print("Chose the weak learners %s" % chosen_wls)

    @staticmethod
    def generate_weak_learner(X, y, feature):
        sampled_X = X[feature]
        feature_idx = list(X.columns).index(feature)
        sorted_values = sampled_X.sort_values()
        consecutive_means = sorted_values.rolling(2, min_periods=1).mean()

        max_thresholds= 25.0
        means = set(consecutive_means.values)
        stride = int(np.ceil(len(means) / max_thresholds))
        thresholds = list(sorted(means)[i] for i in range(0, len(means), stride))
        thresholds.append(sorted_values.max())

        max_acc = -np.inf
        best_threshold = thresholds[0]
        best_threshold_idx = 0
        
        for idx, thresh in enumerate(thresholds):
            y_pred = (sampled_X >= thresh).astype(int).replace(0, -1)
            curr_acc = (y.values * y_pred.values.T).sum()
            if curr_acc > max_acc:
                best_threshold, best_threshold_idx = thresh, idx
                max_acc = curr_acc
        return lambda x: 1 if x[feature_idx] >= best_threshold else -1

    @staticmethod
    def generate_H(X, y, size_H=None):
        if type(X) is not pd.DataFrame:
            X = pd.DataFrame(data=X)
        if type(y) is not pd.Series:
            y = pd.DataFrame(data=y)
        if size_H is None:
            size_H = len(X.columns)
        size_H = min(size_H, len(X.columns))
        sampled_X = X.sample(n=size_H, axis=1, replace=False)
        return [RBoost.generate_weak_learner(X, y, feature) for feature in sampled_X.columns]

    def fit(self, X, y):
        """
        Fit LPBoost model to data.
        Parameters
        -------
        :param X: array_like, shape (n_samples, n_features)
            Data matrix of explicit features
        :param y: array_like, shape (n_samples,)
            Desired labels
        Returns:
        -------
        :return:
        """
        positive_mask = y == self.POSITIVE_CLASS
        negative_mask = y == self.NEGATIVE_CLASS
        y[positive_mask] = 1
        y[negative_mask] = -1
        self.H_weak = RBoost.generate_H(X, y)
        return self._fitString(X, y)


# %% [markdown] {"papermill":{"duration":0.010328,"end_time":"2020-08-26T15:32:49.738411","exception":false,"start_time":"2020-08-26T15:32:49.728083","status":"completed"},"tags":[]}
# # Data Loading

# %% [code] {"execution":{"iopub.execute_input":"2020-08-26T15:32:49.773567Z","iopub.status.busy":"2020-08-26T15:32:49.772655Z","iopub.status.idle":"2020-08-26T15:32:51.790174Z","shell.execute_reply":"2020-08-26T15:32:51.789349Z"},"papermill":{"duration":2.041142,"end_time":"2020-08-26T15:32:51.790302","exception":false,"start_time":"2020-08-26T15:32:49.749160","status":"completed"},"tags":[]}
PART_NUMBER = 0
if LOAD_ALL:
    # dataset_name = "spambase.csv"
    dataset_paths = [os.path.join(CLASS_DBS_PATH, dataset_name) for dataset_name in sorted(os.listdir(CLASS_DBS_PATH))]
    # [("db_name", read_cvs)]
    raw_dbs = [(os.path.basename(dataset_path), pd.read_csv(dataset_path)) for dataset_path in dataset_paths]
    # [("db_name", X, y)]
    raw_dbs = [(raw_db[0], \
                raw_db[1].loc[:, raw_db[1].columns != raw_db[1].columns[-1]], \
                raw_db[1].loc[:, raw_db[1].columns[-1]]) \
               for raw_db in raw_dbs]

    raw_dbs = sorted(raw_dbs, key=lambda x: len(x[1]))  # sort by db length
    
    if len(sys.argv) > 1:
        num_parts = int(sys.argv[1])
        curr_part = int(sys.argv[2])
        assert curr_part <= num_parts
        assert curr_part >= 1
     
        PART_NUMBER = curr_part
        part_size = int(np.ceil(len(raw_dbs) / num_parts))

        #lower_idx, upper_idx = (curr_part - 1) * part_size, min(curr_part * part_size, len(raw_dbs) -1)
        #print("Working on dbs %d to %d" % (lower_idx, upper_idx))
        #raw_dbs = raw_dbs[lower_idx:upper_idx]
        print("working on dbs %s" % str(list(range(curr_part-1, len(raw_dbs), num_parts))))
        raw_dbs = [raw_dbs[i] for i in range(curr_part-1, len(raw_dbs), num_parts)]
        
        

else:
    dataset_name = "teachingAssistant.csv"
    dataset_path = os.path.join(CLASS_DBS_PATH, dataset_name)

    raw_dbs = [(os.path.basename(dataset_path), pd.read_csv(dataset_path))]

    raw_dbs = [(raw_db[0], \
                raw_db[1].loc[:, raw_db[1].columns != raw_db[1].columns[-1]], \
                raw_db[1].loc[:, raw_db[1].columns[-1]]) \
               for raw_db in raw_dbs]


# %% [markdown] {"papermill":{"duration":0.010602,"end_time":"2020-08-26T15:32:51.933506","exception":false,"start_time":"2020-08-26T15:32:51.922904","status":"completed"},"tags":[]}
# # Pre-processing Pipeline

# %% [code] {"execution":{"iopub.execute_input":"2020-08-26T15:32:51.971788Z","iopub.status.busy":"2020-08-26T15:32:51.970800Z","iopub.status.idle":"2020-08-26T15:32:51.974164Z","shell.execute_reply":"2020-08-26T15:32:51.973524Z"},"papermill":{"duration":0.029429,"end_time":"2020-08-26T15:32:51.974295","exception":false,"start_time":"2020-08-26T15:32:51.944866","status":"completed"},"tags":[]}
class DelayedColumnTransformer(TransformerMixin, BaseEstimator):
    @staticmethod
    def get_dtype_columns_indices(df, dtype):
        columns = list(df.columns)
        return [columns.index(col) for col in df.select_dtypes(dtype).columns]

    def __init__(self, dtype_to_transformers):
        self.dtype_to_transformers = dtype_to_transformers
        self.pipeline = None

    def fit(self, X, y=None):
        # print("Number of columns: %d" % len(X.columns))
        # print("Number of categorical %d, numerical %d" % (len(self.get_dtype_columns_indices(X, np.object)), len(self.get_dtype_columns_indices(X, np.number))))
        self.pipeline = ColumnTransformer([
            ("%s" % str(dtype),
             Pipeline([("%s_%d" % (transformer.__class__.__name__, idx), transformer) for transformer in transformers]),
             self.get_dtype_columns_indices(X, dtype)) for
            idx, (dtype, transformers) in enumerate(self.dtype_to_transformers)]
            , remainder='drop')  # Should not drop anything

        return self.pipeline.fit(X, y)

    def transform(self, X, y=None):
        res = self.pipeline.transform(X)
        return res


preprocessing = DelayedColumnTransformer([
    (np.object, [SimpleImputer(strategy='constant'), OneHotEncoder(handle_unknown='ignore')]),
    (np.number, [SimpleImputer(strategy='mean'), VarianceThreshold(0.0)])
])


# %% [markdown] {"papermill":{"duration":0.010696,"end_time":"2020-08-26T15:32:51.995940","exception":false,"start_time":"2020-08-26T15:32:51.985244","status":"completed"},"tags":[]}
# # Metrics Utilities

# %% [code] {"execution":{"iopub.execute_input":"2020-08-26T15:32:52.057379Z","iopub.status.busy":"2020-08-26T15:32:52.056285Z","iopub.status.idle":"2020-08-26T15:32:52.059597Z","shell.execute_reply":"2020-08-26T15:32:52.058879Z"},"papermill":{"duration":0.052939,"end_time":"2020-08-26T15:32:52.059741","exception":false,"start_time":"2020-08-26T15:32:52.006802","status":"completed"},"tags":[]}
def nanaverage(A, weights, axis):
    return np.nansum(A * weights, axis=axis) / ((~np.isnan(A)) * weights).sum(axis=axis)


def get_time_metrics(cv, X_train, y_train, X_test):
    t0 = time.time()
    cv.fit(X_train, y_train)
    training_time = time.time() - t0

    t0 = time.time()
    cv.predict(X_test)  # N x L, values in [0, 1]
    inference_time_per_entry = (time.time() - t0) / len(X_test)
    inference_time_for_1000 = inference_time_per_entry * 1000

    return training_time, inference_time_for_1000


def get_multiclass_metrics(y_test, y_test_pred):
    acc = balanced_accuracy_score(y_test, y_test_pred) if WEIGHTED_METRICS \
        else accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, average='weighted' if WEIGHTED_METRICS \
        else 'macro')

    return acc, prec


def binarize_zero_one(y, labels):
    y_ret = y.copy()
    y_ret[y == train_labels[0]] = 0
    y_ret[y == train_labels[1]] = 1
    return y_ret.astype(int)


def get_binary_metrics(y_test, y_test_pred, y_test_pred_per_label_probs, train_labels):
    per_label_fpr = []
    per_label_tpr = []
    per_label_pr_auc = []
    per_label_roc_auc = []

    def set_metrics(y_true, y_pred, y_pred_probs):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        per_label_fpr.append(fpr[np.asarray(thresholds) == 1][0])
        per_label_tpr.append(tpr[np.asarray(thresholds) == 1][0])
        per_label_roc_auc.append(auc(fpr, tpr))
        pr_auc = average_precision_score(y_true, y_pred_probs)
        per_label_pr_auc.append(pr_auc)

    test_labels = y_test.unique()
    test_labels_dist = (y_test.value_counts() / len(y_test)).values
    assert set(test_labels).issubset(set(train_labels)), "TEST LABELS NOT IN TRAIN LABELS"  # Shouldn't happen

    is_binary = len(train_labels) == 2

    if is_binary:  # Binary case is considered as single labeled
        curr_y_true = binarize_zero_one(y_test.values, train_labels)
        curr_y_pred = binarize_zero_one(y_test_pred, train_labels)
        curr_y_pred_probs = y_test_pred_per_label_probs[:, 1]

        set_metrics(curr_y_true, curr_y_pred, curr_y_pred_probs)

        weights = [1] if not WEIGHTED_METRICS else [np.mean(curr_y_true)]

    else:
        lb = LabelBinarizer()
        y_true_binarized = lb.fit_transform(y_test)
        y_pred_binarized = lb.transform(y_test_pred)
        lb_classes = list(lb.classes_)

        if len(lb_classes) == 2:
            y_true_binarized = np.hstack((y_true_binarized, 1-y_true_binarized))
            y_pred_binarized = np.hstack((y_pred_binarized, 1-y_pred_binarized))

        for label_idx, curr_label in enumerate(test_labels):
            curr_y_true = y_true_binarized[:, lb_classes.index(curr_label)]
            curr_y_pred = y_pred_binarized[:, lb_classes.index(curr_label)]
            curr_y_pred_probs = y_test_pred_per_label_probs[:, label_idx]

            set_metrics(curr_y_true, curr_y_pred, curr_y_pred_probs)

        weights = [1 for i in range(len(test_labels))] if not WEIGHTED_METRICS else test_labels_dist

    fpr = np.average(per_label_fpr, weights=weights)
    tpr = np.average(per_label_tpr, weights=weights)
    pr_auc = np.average(per_label_pr_auc, weights=weights)
    roc_auc = np.average(per_label_roc_auc, weights=weights)

    return fpr, tpr, pr_auc, roc_auc


def write_all_results(dbs_results):
    if PART_NUMBER > 0:
        results_path = RESULTS_CSV_PATH + (".%d" % PART_NUMBER)
    else:
        results_path = RESULTS_CSV_PATH
    with open(results_path, "w") as csvfile:

        fieldnames = ['dataset_name', 'alg_name']

        # getting the metrics name (using the first db and our model)
        first_db_name = list(dbs_results.keys())[0]
        metrics_names = list(dbs_results[first_db_name][OUR_MODEL][0].keys())
        fieldnames += metrics_names

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for db_idx, db_name in enumerate(dbs_results):
            for model_idx, model_name in enumerate(MODELS_LIST):
                for fold_num in range(EVAL_FOLDS):
                    info_dict = {'dataset_name': db_name, 'alg_name': model_name}

                    metrics_dict = dbs_results[db_name][model_name][fold_num]

                    writer.writerow({**info_dict, **metrics_dict})


def write_single_db_results(db_results, db_name):
    print("writing single db results %s" % db_name)
    with open(RESULT_CSV_PATH(db_name), "w") as csvfile:

        fieldnames = ['dataset_name', 'alg_name']

        # getting the metrics name (using the first db and our model)
        first_db_name = list(dbs_results.keys())[0]
        metrics_names = list(dbs_results[first_db_name][OUR_MODEL][0].keys())
        fieldnames += metrics_names

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model_idx, model_name in enumerate(MODELS_LIST):
            for fold_num in range(EVAL_FOLDS):
                info_dict = {'dataset_name': db_name, 'alg_name': model_name}

                metrics_dict = db_results[model_name][fold_num]

                writer.writerow({**info_dict, **metrics_dict})


# %% [markdown] {"papermill":{"duration":0.011252,"end_time":"2020-08-26T15:32:52.082063","exception":false,"start_time":"2020-08-26T15:32:52.070811","status":"completed"},"tags":[]}
# # Train

# %% [code] {"execution":{"iopub.execute_input":"2020-08-26T15:32:52.160880Z","iopub.status.busy":"2020-08-26T15:32:52.130691Z"},"papermill":{"duration":null,"end_time":null,"exception":false,"start_time":"2020-08-26T15:32:52.093332","status":"running"},"tags":[]}

# Fixed replacements info
y_values_encoding = {'tested_negative': -1, 'schizophrenic': 1, 'Lewis': -1, 'sn': -1, 'cn': 1, 'excellent': 2,
                     'not_suitable': 0, 'Tile': -1, 'republican': -1, 'democrat': 1, 'ok': 1, 'dead': -1,
                     'non-schizophrenic': -1, 'Holyfield': 1, 'bad': -1, 'tested_positive': 1, 'P': 1, 'good': 1,
                     'alive': 1, 'N': -1, 'Insulation': 1}
X_nan_values = {'labor.csv': ['?'], 'braziltourism.csv': ['?']}
X_func_replacement = {'braziltourism.csv': functools.partial(pd.to_numeric, errors='ignore')}


# ---------------------------------------

def db_encode(db_name, X, y):
    if y.dtype == np.object:
        y = y.replace(y_values_encoding)
    if db_name in X_nan_values:
        X = X.replace(X_nan_values[db_name], np.nan)
    if db_name in X_func_replacement:
        X = X.apply(X_func_replacement[db_name])
    return X, y


eval_metric = balanced_accuracy_score
results = {}

kf = StratifiedKFold(n_splits=EVAL_FOLDS, random_state=RANDOM_SEED)
model = Pipeline(steps=[('model', RBoost() if USE_RBOOST else ELPBoost())])
comp_model = Pipeline(steps=[('model', lgb.LGBMClassifier())])
ova_model = OneVsRestClassifier(model)
ova_comp_model = OneVsRestClassifier(comp_model)

# {db_name: {our_model: reulsts, compare_model: results}}
dbs_results = {}

with open(os.path.join(WORKING_DIR, "bad-dbs.txt"), "w") as f:
    pass

os.system('mkdir -p {}'.format(MODELS_DIR))

for db_name, X, y in raw_dbs:
    #     if db_name != "braziltourism.csv":
    #         continue
    dbs_results[db_name] = {}
    X, y = db_encode(db_name, X, y)

    #     X = X.iloc[:max(100, len(X)), :]
    #     y = y.iloc[:max(100, len(y))]
    N = len(X) * (1 - (1 / EVAL_FOLDS))

    model_params = {
        'estimator__model__kappa': [1 / 3, 1 / N, 2 / N, 3 / N],
        'estimator__model__T': [3, 5, 10],
        'estimator__model__reg': [1, 10, 20, 50, 100],
        'estimator__model__silent': [True],
        'estimator__model__verbose': [False]
    }

    comp_model_params = {
        "estimator__model__n_estimators": explored_n_estimators,
        "estimator__model__learning_rate": explored_learning_rate,
        "estimator__model__max_depth": explored_max_depth,
        "estimator__model__reg_lambda": explored_lambda,
        "estimator__model__num_leaves": explored_num_leaves,
        "estimator__model__objective": ['binary'],
    }

    # comp_model_params = {
    #   "estimator__model__n_estimators": [226],
    #   "estimator__model__learning_rate": (1e-10, 1),
    #   "estimator__model__max_depth": [11],
    #   "estimator__model__reg_lambda": (0, 100),
    #   "estimator__model__num_leaves": [28],
    #   "estimator__model__objective": ['binary'],
    # }

    results_per_fold = {}
    comp_results_per_fold = {}
    fold_num = 1

    # list of results per fold
    folds_results = []
    comp_folds_results = []

    is_binary = len(y.unique()) == 2  # No special case for binary
    try:
        for train_index, test_index in kf.split(X, y):
            print("{}:{}:Fold_{}".format(datetime.now(), db_name, fold_num))
            # --- get fold and preprocess ---
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            invalid_labels = set(y_test.unique()) - set(y_train.unique())
            if len(invalid_labels) > 0:  # Introduce new labels, should occur only for outliers due to StratifiedKFold
                X_train = pd.concat([X_train, pd.DataFrame(
                    [[np.nan for _ in range(len(X_train.columns))] for _ in range(len(invalid_labels))],
                    columns=X_train.columns)], ignore_index=True)
                y_train = y_train.append(pd.Series(list(invalid_labels)), ignore_index=True)
                X_train, y_train = db_encode(db_name, X_train, y_train)

            # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
            preprocessing.fit(X_train, y_train)
            X_train = preprocessing.transform(X_train)
            X_test = preprocessing.transform(X_test)

            # --- random search ---
            cv = RandomizedSearchCV(estimator=ova_model, param_distributions=model_params, \
                                    scoring=make_scorer(eval_metric), cv=HPT_FOLDS, \
                                    n_iter=RANDOM_CV_ITER, random_state=RANDOM_SEED)
            comp_cv = RandomizedSearchCV(estimator=ova_comp_model, param_distributions=comp_model_params, \
                                         scoring=make_scorer(eval_metric), cv=HPT_FOLDS, \
                                         n_iter=RANDOM_CV_ITER, random_state=RANDOM_SEED)

            curr_fold_results = {'fold_num': fold_num}
            curr_fold_comp_results = {'fold_num': fold_num}

            # --- measure times - FIT + INFER --- #
            print("training our model")
            curr_fold_results['train_time'], curr_fold_results['infer_time'] = get_time_metrics(cv, X_train, y_train,
                                                                                                X_test)
            print("Finished training our model")
            print("training comparison model")
            curr_fold_comp_results['train_time'], curr_fold_comp_results['infer_time'] = get_time_metrics(comp_cv,
                                                                                                          X_train,
                                                                                                          y_train,
                                                                                                          X_test)
            print("Finished training comparison model")

            # --- save trained models ---
            model_path = MODELS_DIR + "/model_fold_" + str(fold_num) + "_db_name_" + db_name
            comp_model_path = MODELS_DIR + "/comp_model_fold_" + str(fold_num) + "_db_name_" + db_name
            dill.dump(cv.best_estimator_, open(model_path, 'wb'))
            dill.dump(comp_cv.best_estimator_, open(comp_model_path, 'wb'))

            # --- register best params ---
            best_comp_params = comp_cv.best_params_
            best_params = cv.best_params_
            curr_fold_comp_results['best_params'] = best_comp_params
            curr_fold_results['best_params'] = best_params
            # --- get predictions for MultiRBoost --- #
            y_test_pred_per_label_scores = cv.predict_proba(X_test)
            y_test_pred_per_label_scores[np.isnan(y_test_pred_per_label_scores)] = 1.0/y_test_pred_per_label_scores.shape[1]  # nan refers to 1.0 for each binary
            y_test_pred = cv.predict(X_test)

            train_labels = cv.best_estimator_.classes_
            comp_train_labels = comp_cv.best_estimator_.classes_  # can be sorted differently

            # --- get predictions for LightGBM --- #
            y_test_pred_comp_per_label_scores = comp_cv.predict_proba(X_test)
            y_test_pred_comp_per_label_scores[np.isnan(y_test_pred_comp_per_label_scores)] = 1.0/y_test_pred_comp_per_label_scores.shape[1]  # nan refers to 1.0 for each binary
            y_test_pred_comp = comp_cv.predict(X_test)

            # metrics applicable in multiclass setting ---percision--- #
            multiclass_metrics_dict = {0: 'accuracy', 1: 'precision'}

            multiclass_metrics = get_multiclass_metrics(y_test, y_test_pred)
            multiclass_comp_metrics = get_multiclass_metrics(y_test, y_test_pred_comp)

            for metric_pos, metric_name in multiclass_metrics_dict.items():
                curr_fold_results[metric_name] = multiclass_metrics[metric_pos]
                curr_fold_comp_results[metric_name] = multiclass_comp_metrics[metric_pos]

            # Metrics only applicable in a binary setting ---fpr, tpr, pr_auc, roc-auc--- #
            binary_metrics_dict = {0: 'fpr', 1: 'tpr', 2: 'pr_auc', 3: 'roc_auc'}

            binary_metrics = get_binary_metrics(y_test, y_test_pred, y_test_pred_per_label_scores, \
                                                train_labels)
            binary_comp_metrics = get_binary_metrics(y_test, y_test_pred_comp, y_test_pred_comp_per_label_scores, \
                                                     comp_train_labels)

            for metric_pos, metric_name in binary_metrics_dict.items():
                curr_fold_results[metric_name] = binary_metrics[metric_pos]
                curr_fold_comp_results[metric_name] = binary_comp_metrics[metric_pos]

            # add the current fold results to the results list
            folds_results.append(curr_fold_results)
            comp_folds_results.append(curr_fold_comp_results)

            fold_num += 1
        dbs_results[db_name][OUR_MODEL] = folds_results
        dbs_results[db_name][COMP_MODEL] = comp_folds_results

        write_single_db_results(dbs_results[db_name], db_name)

    except Exception as e:
        print("ERROR!", e)
        # catching wierd values
        with open(os.path.join(WORKING_DIR, "bad-dbs.txt"), "a") as f:
            dbs_results.pop(db_name)
            f.write("{db_name}: {error}\n".format(db_name=db_name, error=e))

        continue

print(dbs_results)
write_all_results(dbs_results)

print("Done writing results in part %d" % PART_NUMBER)
sys.exit(1)

# %% [markdown] {"papermill":{"duration":null,"end_time":null,"exception":null,"start_time":null,"status":"pending"},"tags":[]}
# # Statistic hypothesis

# %% [code] {"papermill":{"duration":null,"end_time":null,"exception":null,"start_time":null,"status":"pending"},"tags":[]}
# vector (n_dbs * 2)
models_measures = np.zeros(shape=(len(dbs_results), 2))

for model_idx, model_name in enumerate(MODELS_LIST):
    for db_idx, db_name in enumerate(dbs_results):
        models_measures[db_idx][model_idx] = np.average([dbs_results[db_name][model_name][i][STAT_CHOSEN_METRIC] \
                                                         for i in range(EVAL_FOLDS)])

stats_per_db = [models_measures[i, :] for i in range(models_measures.shape[0])]

p_value = friedmanchisquare(*stats_per_db).pvalue
print(p_value)

if p_value <= P_THRESH:
    print("Statistically significant!")
    post_hoc_res = posthoc_nemenyi_friedman(models_measures)
    print("nemenyi post-hoc result: {res}".format(res=post_hoc_res))

else:
    print("Not statistically significant!")


# %% [markdown] {"papermill":{"duration":null,"end_time":null,"exception":null,"start_time":null,"status":"pending"},"tags":[]}
# # Importance & SHAP

# %% [code] {"papermill":{"duration":null,"end_time":null,"exception":null,"start_time":null,"status":"pending"},"tags":[]}
def close_and_increase_plt():
    plt.close(globals()["figure_num"])
    globals()["figure_num"] += 1


def get_plot_path(innder_dir, desc):
    return os.path.join(PLOTS_DIR + '/' + innder_dir + '/'
                        + desc
                        + '.' + PLOTS_FORMAT)


def generate_importance(model, test_db_name):
    for imp_type in IMPORTANCE_TYPES:
        feature_important = model.get_booster().get_score(importance_type=imp_type)
        keys = list(feature_important.keys())
        values = list(feature_important.values())

        data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
        ax = data.plot(kind='barh')
        imp_plot_path = get_plot_path(IMPORTANCE_DIR, '{test_db_name}_{imp_type}'.format(test_db_name=test_db_name, \
                                                                                         imp_type=imp_type))
        ax.figure.savefig(imp_plot_path, bbox_inches='tight')


def generate_shap(model, test_db_name, X_test):
    booster_bytearray = model.get_booster().save_raw()[4:]

    def myfun(self=None):
        return booster_bytearray

    model.get_booster().save_raw = myfun

    # create a SHAP explainer and values
    explainer = shap.TreeExplainer(model)

    # summary_plot
    plt.figure(globals()["figure_num"])
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, max_display=PLOT_TOP_FEATURES, show=False, plot_type="bar")

    shap_summary_plot_path = get_plot_path(SHAP_DIR, '{test_db_name}_shap'.format(test_db_name=test_db_name))
    plt.savefig(shap_summary_plot_path, bbox_inches='tight', format=PLOTS_FORMAT)
    close_and_increase_plt()


# %% [markdown] {"papermill":{"duration":null,"end_time":null,"exception":null,"start_time":null,"status":"pending"},"tags":[]}
# # Meta-Learning

# %% [code] {"papermill":{"duration":null,"end_time":null,"exception":null,"start_time":null,"status":"pending"},"tags":[]}
def summarize_metrics(folds_metrics):
    return np.average([folds_metrics[i][STAT_CHOSEN_METRIC] for i in range(len(folds_metrics))])


per_dataset_winner = {}
for db_name in dbs_results:
    our_model_metrics = dbs_results[db_name][OUR_MODEL]
    comp_model_metrics = dbs_results[db_name][COMP_MODEL]
    we_win = summarize_metrics(our_model_metrics) >= summarize_metrics(comp_model_metrics)
    per_dataset_winner[db_name.split('.')[0]] = 1 if we_win else -1

X_raw = pd.read_csv(META_DBS_PATH, header=0, index_col='dataset')
X = X_raw.loc[list(per_dataset_winner.keys()), :]
y = pd.Series([per_dataset_winner[db_name] for db_name in per_dataset_winner])

db_names = [db_name for db_name in per_dataset_winner]

loo = LeaveOneOut()
meta_model_results = {}

os.system('mkdir -p {plots_dir}/{inner_dir}'.format(plots_dir=PLOTS_DIR, inner_dir=IMPORTANCE_DIR))
os.system('mkdir -p {plots_dir}/{inner_dir}'.format(plots_dir=PLOTS_DIR, inner_dir=SHAP_DIR))

for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    curr_dataset = X_test.index[0]
    meta_model = xgb.XGBClassifier(booster='gbtree')
    meta_model.fit(X_train, y_train)
    y_pred = meta_model.predict(X_test)[0]

    meta_model_results[curr_dataset] = y_pred

    generate_importance(meta_model, db_names[test_index[0]])
    generate_shap(meta_model, db_names[test_index[0]], X_test)

y_pred = pd.Series([meta_model_results[db_name] for db_name in meta_model_results])

print("Meta Model Accuracy: %f" % accuracy_score(y, y_pred))
