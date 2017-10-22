from helpers import prod, positive
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from scipy.sparse import issparse
from scipy.sparse import csr_matrix
from scipy.misc import logsumexp


# equations from "partially supervised classification of text documents"
# TODO make classifier that accepts continuous probabilistic labels (or probability tuples) using these formulae
# TODO don't use proba tuple per doc, but only pos, if neg is always 1-pos

class proba_label_MNB(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        return

    def fit(self, X, y):
        """calculate class and word priors from labels or probabilistic labels"""

        if np.isscalar(y[0]):
            yp = np.array([[self.label2num(label),
                            1 - self.label2num(label)]
                           for label in y])
        else:
            yp = np.array(y)

        if issparse(X):
            npX = X.todense()
        else:
            npX = np.array(X)

        print("calculating class prior")
        self.pr_c = np.array(self.prior_c(yp))
        print(self.pr_c)
        print("calculating attribute priors")
        self.pr_w = np.array([self.prior_w_given_c(npX, yp, cls=0),
                              self.prior_w_given_c(npX, yp, cls=1)])
        print(self.pr_w)

        return self

    # Pr[c_j | d_i]: probabilistic Label for a document
    # w_{d_i},k makes no sense in BOW: multiply vocabulary-word pros with binary word vectors to get equivalent result
    def predict_proba(self, X):
        return [self.proba(x) for x in X]

    def proba(self, x):
        """predict probabilities of a given class for one doc"""
        # uses fitted class probability,
        # fitted word probability per class

        posprobs = np.transpose(x) * self.pr_w[0]
        negprobs = np.transpose(x) * self.pr_w[1]

        # print("pr_c_0", self.pr_c[0], "posprobs > 0:", posprobs[np.nonzero(posprobs)], np.prod(posprobs[np.nonzero(posprobs)]))
        # print("pr_c_1", self.pr_c[1], "negprobs > 0:", negprobs[np.nonzero(negprobs)], np.prod(negprobs[np.nonzero(negprobs)]))

        # TODO log-Version so gut wie legacy-Version machen
        # legacy version: numerical issues because numbers are too small!
        return proba_nolog(x, self.pr_c, posprobs, negprobs)

        # new version with log probabilities: something is wrong!
        return proba_log(x, self.pr_c, posprobs, negprobs)

    def predict(self, X):
        print("pos probs", [p for p in self.predict_proba(X)])
        return [round(p[0]) for p in self.predict_proba(X)]

    # Pr[c_j]: probability of a class label (mean of probabilities per doc)
    def prior_c(self, y):
        """the two classes' prior probabilities: average of training data"""
        return sum(y) / np.shape(y)[0]

    # Pr[w_t | c_j]: probability of each attribute, given a class label
    def prior_w_given_c(self, X, y, cls=1):
        """prior probabilities per word, given a class"""

        # select columns of probabilities for class 1 or 0
        p_c_given_x = y[:, cls]

        # legacy version
        # numerators = [self.alpha + sum([X[i][idx_w] * p_c_given_x[i] for i in range(np.shape(X)[0])])
        #               for idx_w in range(np.shape(X)[1])]
        #
        # denominator = sum(numerators)
        #
        # print("numerators", numerators)
        # print(numerators)
        # print("denom:", denominator)

        # TODO weird sparse matrix errors
        #
        numerators = np.dot(X.transpose(), p_c_given_x)
        print("nums", numerators)
        print("nums[0]", numerators[0])

        if (issparse(numerators[0])):
            numerators = numerators.toarray()
            # numerators = np.array([n.toarray() for n in numerators])

        numerators += self.alpha

        # print("array mult. numerators", numerators)

        denominator = np.sum(numerators)

        # print("denom:", denominator)
        # print("nums/denom", numerators/denominator)

        # # numerators = np.full(np.shape(numerators), self.alpha) + (numerators.todense() if issparse(numerators) else numerators)
        #
        # denominators = [sum(numerators[0]), sum(numerators[1])]
        #
        # print("num:", np.shape(numerators), numerators, "\ndenom 0:", np.shape(denominators), denominators)
        #
        # return np.array([numerators[0] / denominators[0], numerators[1] / denominators[1]])

        return numerators / denominator

    def label2num(self, label):
        if isinstance(label, (int, float)):
            return 1.0 * label
        elif label in ['pos', 'POS', 'Pos', 'positive', 'Positive', 'yes', '1']:
            return 1.0
        else:
            return 0.0


def proba_log(x, pr_c, posprobs, negprobs):
    numerators = np.exp([np.log(pr_c[0])
                         + np.sum(np.log(posprobs[np.nonzero(posprobs)])),
                         np.log(pr_c[1])
                         + np.sum(np.log(negprobs[np.nonzero(negprobs)]))])
    denominator = logsumexp(numerators)
    # print("proba(x) numerators", numerators)
    # print("denominator", denominator)
    return (np.exp(numerators[0] - denominator), np.exp(numerators[1] - denominator))


# legacy version: numerical issues because numbers are too small!
def proba_nolog(x, pr_c, posprobs, negprobs):
    numerators = [pr_c[0] * np.prod(posprobs[np.nonzero(posprobs)]),
                  pr_c[1] * np.prod(negprobs[np.nonzero(negprobs)])]
    denominator = np.sum(numerators)

    return (numerators[0] / denominator, numerators[1] / denominator)
