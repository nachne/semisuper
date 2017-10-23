from helpers import prod, positive
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import numexpr as ne
from scipy.sparse import issparse
from scipy.sparse import csr_matrix
from scipy.misc import logsumexp
import multiprocessing as multi


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
        print("class probability:", self.pr_c)

        self.pr_w = np.array([self.pr_w_given_c(npX, yp, cls=0),
                              self.pr_w_given_c(npX, yp, cls=1)])
        print("attribute probabilities:", self.pr_w)

        return self

    def predict_proba(self, X):
        """predicts probabilistic class labels for set of docs.
        """

        # TODO multithread / make this an inner product (Pool.map can't handle sparse matrix)

        return [self.proba(x) for x in X]

    # TODO rewrite as inner product
    def proba(self, x):
        """predict probabilities of a given class for one doc

        uses fitted class probability and fitted word probability per class"""

        posprobs = self.pr_w[0][np.nonzero(x)]
        negprobs = self.pr_w[1][np.nonzero(x)]

        numerators = (np.log(self.pr_c[0]) + np.sum(np.log(posprobs)),
                      np.log(self.pr_c[1]) + np.sum(np.log(negprobs)))
        denominator = logsumexp(numerators)
        # print("proba(x) numerators", numerators)
        # print("denominator", denominator)
        return (np.exp(numerators[0] - denominator), np.exp(numerators[1] - denominator))

        ## legacy version: numerical issues because numbers are too small!
        # return proba_nolog(x, self.pr_c, posprobs, negprobs)

        # new version with log probabilities
        return proba_log(x, self.pr_c, posprobs, negprobs)

    def predict(self, X):
        """predict class labels using probabilistic class labels"""
        proba_labels = self.predict_proba(X)

        print("pos probs", [p[0] for p in proba_labels])

        return [int(round(p[0])) for p in proba_labels]

    def proba_c(self, y):
        """the two classes' prior probabilities: average of training data"""
        return sum(y) / np.shape(y)[0]

    def pr_w_given_c(self, X, y, cls):
        """probabilities per word (attribute), given a class label"""

        # select columns of probabilities for class 1 or 0
        p_c_given_x = y[:, cls]

        ## legacy version with loop instead of matrix multiplication:
        # numerators = [self.alpha + sum([X[i][idx_w] * p_c_given_x[i] for i in range(np.shape(X)[0])])
        #               for idx_w in range(np.shape(X)[1])]
        # denominator = sum(numerators)

        numerators = np.dot(X.transpose(), p_c_given_x)

        if (issparse(numerators[0])):
            numerators = numerators.toarray()

        numerators += self.alpha
        denominator = np.sum(numerators)

        # print("denom:", denominator)
        # print("nums/denom", numerators/denominator)

        return numerators / denominator


# legacy version: numerical issues because numbers are too small!
def proba_nolog(x, pr_c, posprobs, negprobs):
    numerators = [pr_c[0] * np.prod(posprobs[np.nonzero(posprobs)]),
                  pr_c[1] * np.prod(negprobs[np.nonzero(negprobs)])]
    denominator = np.sum(numerators)

    return (numerators[0] / denominator, numerators[1] / denominator)


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
