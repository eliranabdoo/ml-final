from sklearn.base import BaseEstimator
from constants import MAX_BATCH_SIZE, RANDOM_SEED
import numpy as np
from cvxopt import modeling, matrix, solvers
import math
import random
from datetime import datetime
import pandas as pd

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

        max_thresholds = 25.0
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