from sklearn.base import BaseEstimator
import numpy as np
from cvxopt import matrix, solvers
import random
from constants import RANDOM_SEED
import pandas as pd

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

    def __init__(self, kappa=1, threshold=10 ** -3, T=1000, reg=0.01, verbose=False, silent=False):
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
            dfdphiiwj = reg_inv * ((uw_mult * sum_weighted_scores_exp - weighted_scores_exp * uw_mult_sum) / sum_weighted_scores_exp_square)

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
