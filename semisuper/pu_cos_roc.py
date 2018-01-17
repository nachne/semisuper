from numpy import ones, zeros
from scipy import vstack
from semisuper.helpers import partition_pos_neg, num_rows, label2num, densify, concatenate
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# ----------------------------------------------------------------
# pipeline builders
# ----------------------------------------------------------------

def ranking_cos_sim(X, threshold=0.1, compute_thresh=False):
    """fits mean training vector and predicts whether cosine similarity is above threshold (default: 0.0)

    predict_proba returns similarity scores.
    if X_thresh is true, uses the training vectors' similarity scores to compute a threshold.
    """

    clf = SimRanker(threshold, compute_thresh)

    model = clf.fit(X, ones(num_rows(X)))
    return model


def rocchio(P, N, alpha=16, beta=4, binary=False):
    """fits mean training vector and predicts whether cosine similarity is above threshold (default: 0.0)

    predict_proba returns similarity scores.
    if X_thresh is true, uses the training vectors' similarity scores to compute a threshold.
    """

    clf = BinaryRocchio(alpha=alpha, beta=beta)

    X = concatenate((P, N))
    y = concatenate((ones(num_rows(P)), zeros(num_rows(N))))

    model = clf.fit(X, y)

    return model


# ----------------------------------------------------------------
# classifiers
# ----------------------------------------------------------------

class SimRanker(BaseEstimator, ClassifierMixin):
    """fits mean training vector, predicts cosine similarity ranking scores

    predict_proba returns scores, predict returns 1 if they are above a given (or calculated) threshold, else 0 """

    def __init__(self, threshold, compute_thresh, verbose=False):
        self.threshold = threshold
        self.compute_thresh = compute_thresh
        self.mean_X = None
        self.verbose = verbose
        return

    def fit(self, X, y=None):
        self.mean_X = X.mean(axis=0).reshape(1, X.shape[1])

        if self.compute_thresh:
            self.threshold = self.dummy_threshold(self.mean_X, X)
            if self.verbose: print("Threshold:", self.threshold)

        return self

    def predict(self, X):
        sims = self.predict_proba(X)
        return [1 if s > [self.threshold] else 0 for s in sims]

    def predict_proba(self, X):
        proba = cosine_similarity(self.mean_X, X)[0]
        return proba

    def ranking(self, X):
        return self.predict_proba(X)

    def dummy_threshold(self, mean_X, X):
        cos_sim = cosine_similarity(mean_X, X)
        return (cos_sim.mean() + cos_sim.min()) / 2


class BinaryRocchio(BaseEstimator, ClassifierMixin):
    """fits prototype vectors for positive and negative training docs

     predicts 1 if a doc is more similar to the positive prototype, 0 otherwise.
     alpha and beta influence how strongly prototype vectors weigh the respective other class.
     input vectors should be normalized in preprocessing."""

    def __init__(self, alpha=16, beta=4):
        self.proto_p = None
        self.proto_n = None
        self.alpha = alpha
        self.beta = beta
        return

    def fit(self, X, y):
        """learn prototype vectors for positive and negative docs"""
        y = [label2num(l) for l in y]

        P, N = partition_pos_neg(X, y)

        normalized_p = normalize(P.mean(axis=0))
        normalized_n = normalize(N.mean(axis=0))

        self.proto_p = normalize(self.alpha * normalized_p - self.beta * normalized_n)
        self.proto_n = normalize(self.alpha * normalized_n - self.beta * normalized_p)

        return self

    def predict_proba(self, X):
        """returns values in [0, 1]; >0.5 means x is rather positive. Not a proper probability!"""

        sim_p = cosine_similarity(self.proto_p, X)[0]
        sim_n = cosine_similarity(self.proto_n, X)[0]

        proba = (sim_p - sim_n) / 2 + 0.5

        return vstack((1 - proba, proba)).T

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]

        return proba.round()
