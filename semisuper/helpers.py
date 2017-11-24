from itertools import islice
from functools import reduce
from operator import itemgetter, mul
import numpy as np
from scipy.sparse import issparse
from sklearn.metrics import classification_report as clsr, accuracy_score
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils import check_array
import os.path


# helper

def flatten(l):
    """flatten 2-dimensional sequence to one-dimensional"""
    return [item for sublist in l for item in sublist]


def take(n, iterable):
    """return first n elements of iterable"""
    return list(islice(iterable, n))


def prod(iterable):
    """reduce iterable to the product of its elements"""
    return reduce(mul, iterable, 1)


def identity(x):
    """identity function"""
    return x


class identitySelector():
    def __init__(self):
        return

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X


def positive(x):
    return x > 0


def num_rows(a):
    """returns length of array or vector, number of rows for 2-dimensional arrays"""
    if issparse(a):
        return np.shape(a)[0]
    return len(a)


def arrays(args):
    """make numpy arrays from args list"""
    return [np.array(a) for a in args]


def partition_pos_neg(X, y):
    """partitions X into elements where corresponding y element is nonzero VS zero"""
    pos_idx = np.nonzero(y)
    neg_idx = np.ones(num_rows(y), dtype=bool)
    neg_idx[pos_idx] = False
    return X[pos_idx], X[neg_idx]


def partition_pos_neg_unsure(X, y_pred, confidence):
    """partitions X into positive, negative or undefined elements given y probabilities and confidence threshold"""
    pos_idx = np.where(y_pred[:, 1] >= confidence)
    neg_idx = np.where(y_pred[:, 0] >= confidence)

    unsure_idx = np.ones(num_rows(X), dtype=bool)
    unsure_idx[pos_idx] = False
    unsure_idx[neg_idx] = False

    return X[pos_idx], X[neg_idx], X[unsure_idx]


def label2num(label):
    """convert labels like POS into 0 or 1 values; 0 for anything not in the positive list. don't touch floats"""
    if isinstance(label, (int, float)):
        return 1.0 * label
    elif label in ['pos', 'POS', 'Pos', 'positive', 'Positive', 'yes', '1', '1.0']:
        return 1.0
    else:
        return 0.0


def unsparsify(X):
    if issparse(X):
        return X.todense()
    else:
        return np.array(X)


def pu_measure(y_P, y_U):
    """performance measure for PU problems (r^2)/Pr[f(X)=1], approximates (p*r)/Pr[Y=1]

    requires validation set to be partitioned into P and U before classification, labels to be 1 and 0"""

    if np.sum(np.round(y_U)) == num_rows(y_U) or np.sum(np.round(y_P)) == 0:
        return 0

    r_sq = (np.sum(np.round(y_P)) / num_rows(y_P)) ** 2
    Pr_fx_1 = (np.sum(np.round(y_P)) + np.sum(np.round(y_U))) / (num_rows(y_P) + num_rows(y_U))

    return r_sq / Pr_fx_1


def eval_model(model, X, y):
    if X is not None and y is not None:
        y_pred = model.predict(X)
        print("Accuracy:", accuracy_score(y, y_pred))
        print(clsr(y, y_pred))
    return


def train_report(model, P, N):
    print("Classification Report (on training, not on test data!):\n")
    y_pred = model.predict(np.concatenate((P, N)))
    print(clsr([1. for _ in P] + [0. for _ in N], y_pred))
    return


def file_path(file_relative):
    """return the correct file path given the file's path relative to helpers"""
    return os.path.join(os.path.dirname(__file__), file_relative)
