from helpers import prod, positive
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from scipy.sparse import issparse, csr_matrix
from scipy.sparse import find


# equations from "partially supervised classification of text documents"
# TODO make classifier that accepts continuous probabilistic labels (or probability tuples) using these formulae
# TODO don't use proba tuple per doc, but only pos, if neg is always 1-pos

class proba_label_MNB(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.1):
        self.classes = [0, 1]
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

        self.pr_c = np.array(self.prior_c(yp))
        self.pr_w = self.prior_w_given_c(np.array(X), yp)
        return self

    # Pr[c_j | d_i]: probabilistic Label for a document
    # w_{d_i},k makes no sense in BOW: multiply vocabulary-word pros with binary word vectors to get equivalent result
    def predict_proba(self, X):
        return [self.proba(x) for x in X]

    def proba(self, x):
        """predict probabilities of a given class for one doc"""
        # uses fitted class probability,
        # fitted word probability per class

        numerators = [self.pr_c[j] *
                      prod(filter(positive,
                                  x * self.pr_w[j]))
                      for j in self.classes]

        denominator = sum(numerators)

        return (numerators[0] / denominator, numerators[1] / denominator)

    def predict(self, X):
        return [round(p) for p, n in self.predict_proba(X)]

    # Pr[c_j]: probability of a class label (mean of probabilities per doc)
    def prior_c(self, y):
        """the two classes' prior probabilities: average of training data"""
        return sum(y) / np.shape(y)[0]

    # Pr[w_t | c_j]: probability of each attribute, given a class label
    def prior_w_given_c(self, X, y):
        """prior probabilities per word, given a class"""

        # numerators = [self.alpha + np.sum([X[i,idx_w] * p_c_given_x[i] for i in range(np.shape(X)[0])])
        #               for idx_w in range(np.shape(X)[1])]

        print("shape of X:", np.shape(X), "y:", np.shape(y))

        numerators = np.dot(X.transpose(), y).transpose()

        # numerators = np.full(np.shape(numerators), self.alpha) + (numerators.todense() if issparse(numerators) else numerators)

        denominators = [sum(numerators[0]), sum(numerators[1])]

        print("num:", np.shape(numerators), numerators, "\ndenom 0:", np.shape(denominators), denominators)

        return np.array([numerators[0] / denominators[0], numerators[1]/denominators[1]])

    def label2num(self, label):
        if isinstance(label, (int, float)):
            return label
        elif label in ['pos', 'POS', 'Pos', 'positive', 'Positive', 'yes', '1']:
            return 1
        else:
            return 0
