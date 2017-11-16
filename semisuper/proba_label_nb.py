from semisuper.helpers import num_rows, label2num, unsparsify, identity
from semisuper.transformers import TokenizePreprocessor, TextStats, FeatureNamePipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer, VectorizerMixin
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from scipy.misc import logsumexp
import multiprocessing as multi
import time


# ----------------------------------------------------------------
# general MNB model builder
# ----------------------------------------------------------------

def build_proba_MNB(X, y, verbose=True, text=True):
    """build multinomial Naive Bayes classifier that accepts probabilistic labels

    if text is true, preprocess Text with binary encoding"""
    clf = ProbaLabelMNB(alpha=0.1)

    # TODO make it work for char n-grams
    if text:
        model = Pipeline([
            ('preprocessor', TokenizePreprocessor()),
            # TODO ngrams back to 3
            ('vectorizer',
             CountVectorizer(binary=True, tokenizer=identity, lowercase=False, ngram_range=(1, 3), analyzer='word')),
            ('classifier', clf)
        ])
    else:
        model = Pipeline([
            ('classifier', clf)
        ])
    model.fit(X, y)
    return model  # equations from "partially supervised classification of text documents"


class ProbaLabelMNB(BaseEstimator, ClassifierMixin):
    """multinomial Naive Bayes classifier that accepts probabilistic labels"""

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.pr_c = None
        self.log_pr_c = None
        self.log_pr_w = None
        return

    def fit(self, X, y):
        """calculate class and word probabilities from labels or probabilistic labels"""

        yp = labels2probs(y)
        npX = unsparsify(X)

        self.pr_c = np.array(self.proba_c(yp))
        self.log_pr_c = np.log(self.pr_c)
        print("Class distribution:", self.pr_c)

        print("Computing attribute probabilities",
              "for", np.shape(npX)[1], "attributes")
        self.log_pr_w = np.log(np.array([self.pr_w_given_c(npX, yp, cls=0),
                                         self.pr_w_given_c(npX, yp, cls=1)]))

        return self

    def predict_proba(self, X):
        """predicts probabilistic class labels for set of docs."""

        with multi.Pool(multi.cpu_count()) as p:
            log_probas = np.array(p.map(self.log_proba, [x for x in X]))

        return np.exp(log_probas)

    def log_proba(self, x):
        """predict probabilities of a given class for one doc, using fitted class proba and word proba per class"""

        pos_log_probs = self.log_pr_w[0][np.nonzero(x)]
        neg_log_probs = self.log_pr_w[1][np.nonzero(x)]

        numerators = (self.log_pr_c[0] + np.sum(pos_log_probs),
                      self.log_pr_c[1] + np.sum(neg_log_probs))
        denominator = logsumexp(numerators)

        log_proba_pos = numerators[0] - denominator
        log_proba_neg = numerators[1] - denominator

        return [log_proba_neg, log_proba_pos]

    def predict(self, X):
        """predict class labels using probabilistic class labels"""
        return np.round(self.predict_proba(X)[:,1])

    @staticmethod
    def proba_c(y):
        """the two classes' prior probabilities: average of training data"""
        return np.sum(y, 0) / num_rows(y)

    def pr_w_given_c(self, X, y, cls):
        """probabilities per word (attribute), given a class label"""

        # select columns of probabilities for class 1 or 0
        p_c_given_x = y[:, cls]

        numerators = np.dot(X.transpose(), p_c_given_x)

        if hasattr(numerators, 'toarray'):
            numerators = numerators.toarray()  # in case of sparse rows

        numerators += self.alpha  # Lidstone smoothing
        denominator = np.sum(numerators)

        return numerators / denominator


# ----------------------------------------------------------------
# helpers

def labels2probs(y):
    if np.isscalar(y[0]):
        yp = np.array([[label2num(label),
                        1 - label2num(label)]
                       for label in y])
    else:
        yp = np.array(y)
    return yp

# ----------------------------------------------------------------
