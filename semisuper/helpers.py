from __future__ import absolute_import, division, print_function

import os.path
from functools import reduce
from itertools import islice, groupby
from operator import mul, or_, itemgetter

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import classification_report as clsr, accuracy_score


# helper

def flatten(l):
    """flatten 2-dimensional sequence to one-dimensional"""
    return [item for sublist in l for item in sublist]

def groupby_index(iterable, elem_idx):
    return groupby(iterable, itemgetter(elem_idx))

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

def merge_dicts(dicts):
    dicts = list(dicts)
    z = dicts[0].copy()
    for d in dicts[1:]:
        z.update(d)
    return z

def run_fun(fun):
    return fun()


def identity(x):
    """identity function"""
    return x


def num_rows(a):
    """returns length of array or vector, number of rows for 2-dimensional arrays"""
    if sp.issparse(a):
        return np.shape(a)[0]
    return len(a)


def arrays(args):
    """make numpy arrays from args list"""
    if type(args[0]) == str:
        return [np.array(a, dtype=object) for a in args]
    return [np.array(a) for a in args]


def partition(l, chunksize):
    l = list(l)
    start = 0
    result = []
    for i in range(int(len(l) / chunksize + 1)):
        chunksize = min(chunksize, num_rows(l[start:]))
        subl = l[start:(start + chunksize)]
        if subl:
            result.append(subl) # TODO yield instead?
        start += chunksize
    return result


def partition_pos_neg(X, y):
    """partitions X into elements where corresponding y element is nonzero VS zero"""
    if not np.isscalar(y[0]):
        y = np.round(y[:, 1])
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


def densify(X):
    if sp.issparse(X):
        return X.todense()
    else:
        return np.array(X)


def ngrams(input_list, n):
    return set(map("".join, zip(*[input_list[i:] for i in range(n)])))


def pu_score(y_P, y_U):
    """performance measure for PU problems (r^2)/Pr[f(X)=1], approximates (p*r)/Pr[Y=1]

    requires validation set to be partitioned into P and U before classification, labels to be 1 and 0"""

    if np.sum(np.round(y_U)) == num_rows(y_U) or np.sum(np.round(y_P)) == 0:
        return 0

    r_sq = (np.sum(np.round(y_P)) / num_rows(y_P)) ** 2
    Pr_fx_1 = (np.sum(np.round(y_P)) + np.sum(np.round(y_U))) / (num_rows(y_P) + num_rows(y_U))

    return r_sq / Pr_fx_1


def eval_model(model, X, y):
    if X is not None and y is not None:
        y_pred = np.round(model.predict(X))
        print("Accuracy:", accuracy_score(y, y_pred))
        print(clsr(y, y_pred))
    return


def train_report(model, P, N):
    print("Classification Report (on training, not on test data!):\n")
    y_pred = model.predict(np.concatenate((P, N)))
    print(clsr([1. for _ in P] + [0. for _ in N], y_pred))
    return


def select_PN_below_score(y_pos, U, y_U, noise_lvl=0.1, verbose=False):
    """given the scores of positive docs, a set of unlabelled docs, and their scores, extract potential negative set"""

    y_pos_sorted = np.sort(y_pos)

    # choose probability threshold such that a noise_lvl-th part of spy docs is rated lower
    threshold = y_pos_sorted[int(noise_lvl * num_rows(y_pos_sorted))]
    if verbose: print("Threshold given noise level:", threshold)

    neg_idx = np.where(y_U <= threshold)

    pos_idx = np.ones(num_rows(y_U), dtype=bool)
    pos_idx[neg_idx] = False

    PN = U[neg_idx]
    if verbose: print("Unlabelled docs below threshold:", num_rows(PN), "of", num_rows(U), "\n")

    U_minus_PN = U[pos_idx]

    return U_minus_PN, PN

def concatenate(tup):
    """vertically stack arrays/csr matrices in tup, preserving any sparsity"""

    if reduce(or_, map(sp.issparse, tup)):
        return sp.vstack(tup, format='csr')
    else:
        return np.concatenate(tup)


def file_path(file_relative):
    """return the correct file path given the file's path relative to helpers"""
    return os.path.join(os.path.dirname(__file__), file_relative)
