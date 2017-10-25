from helpers import prod, positive
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import numexpr as ne
from scipy.sparse import issparse
from scipy.sparse import csr_matrix
from scipy.misc import logsumexp
import multiprocessing as multi
from helpers import num_rows


# equations from "partially supervised classification of text documents"
# TODO don't use proba tuple per doc, but only pos, if neg is always 1-pos

class proba_label_MNB(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        return

    def fit(self, X, y):
        """calculate class and word probabilities from labels or probabilistic labels"""

        yp = labels2probs(y)
        npX = unsparse(X)

        self.pr_c = np.array(self.proba_c(yp))
        self.log_pr_c = np.log(self.pr_c)
        print("Class distribution:", self.pr_c)

        print("Computing attribute probabilities")
        self.log_pr_w = np.log(np.array([self.pr_w_given_c(npX, yp, cls=0),
                                         self.pr_w_given_c(npX, yp, cls=1)]))

        return self

    def predict_proba(self, X):
        """predicts probabilistic class labels for set of docs.
        """

        # TODO multithread / make this an inner product (Pool.map can't handle sparse matrix)

        return np.exp([self.log_proba(x) for x in X])

    # TODO rewrite as inner product
    def log_proba(self, x):
        """predict probabilities of a given class for one doc, using fitted class proba and word proba per class"""

        pos_log_probs = self.log_pr_w[0][np.nonzero(x)]
        neg_log_probs = self.log_pr_w[1][np.nonzero(x)]

        numerators = (self.log_pr_c[0] + np.sum(pos_log_probs),
                      self.log_pr_c[1] + np.sum(neg_log_probs))
        denominator = logsumexp(numerators)

        return numerators[0] - denominator

    def predict(self, X):
        """predict class labels using probabilistic class labels"""
        return np.round(self.predict_proba(X))

    def proba_c(self, y):
        """the two classes' prior probabilities: average of training data"""
        return sum(y) / num_rows(y)

    def pr_w_given_c(self, X, y, cls):
        """probabilities per word (attribute), given a class label"""

        # select columns of probabilities for class 1 or 0
        p_c_given_x = y[:, cls]

        numerators = np.dot(X.transpose(), p_c_given_x)

        if (issparse(numerators[0])):
            numerators = numerators.toarray()  # in case of sparse rows

        numerators += self.alpha  # Lidstone smoothing
        denominator = np.sum(numerators)

        return numerators / denominator


# ----------------------------------------------------------------
# helpers

def unsparse(X):
    if issparse(X):
        return X.todense()
    else:
        return np.array(X)


def labels2probs(y):
    if np.isscalar(y[0]):
        yp = np.array([[label2num(label),
                        1 - label2num(label)]
                       for label in y])
    else:
        yp = np.array(y)
    return yp


def label2num(label):
    if isinstance(label, (int, float)):
        return 1.0 * label
    elif label in ['pos', 'POS', 'Pos', 'positive', 'Positive', 'yes', '1']:
        return 1.0
    else:
        return 0.0


# ----------------------------------------------------------------


# misc.

#
# legacy version: numerical issues because numbers are too small!
def proba_nolog(x, pr_c, posprobs, negprobs):
    numerators = [pr_c[0] * np.prod(posprobs[np.nonzero(posprobs)]),
                  pr_c[1] * np.prod(negprobs[np.nonzero(negprobs)])]
    denominator = np.sum(numerators)

    return (numerators[0] / denominator, numerators[1] / denominator)
