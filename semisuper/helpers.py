from itertools import islice
from functools import reduce
from operator import itemgetter, mul
from numpy import shape, array, nonzero, ones, sum, round
from scipy.sparse import issparse
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


def positive(x):
    return x > 0


def num_rows(a):
    """returns length of array or vector, number of rows for 2-dimensional arrays"""
    return itemgetter(0)(shape(a))


def arrays(args):
    """make numpy arrays from args list"""
    return [array(a) for a in args]


def partition_pos_neg(X, y):
    """partitions X into elements where corresponding y element is nonzero VS zero"""
    # TODO array/sparse/list dispatch

    pos_idx = nonzero(y)

    neg_idx = ones(num_rows(y), dtype=bool)
    neg_idx[pos_idx] = False

    # print(X[pos_idx])
    # print(X[neg_idx])

    return X[pos_idx], X[neg_idx]


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
        return array(X)


def pu_measure(y_P, y_U):
    """performance measure for PU problems (r^2)/Pr[f(X)=1], equivalent to (p*r)/Pr[Y=1]

    i.e. divide
    requires validation set to be partitioned into P and U before classification, labels to be 1 and 0"""

    if sum(round(y_U)) == num_rows(y_U) or sum(round(y_P)) == 0:
        return 0

    r_sq = (sum(round(y_P)) / num_rows(y_P)) ** 2
    Pr_fx_1 = (sum(round(y_P)) + sum(round(y_U))) / (num_rows(y_P) + num_rows(y_U))

    return r_sq / Pr_fx_1


def file_path(file_relative):
    """return the correct file path given the file's path relative to helpers"""
    return os.path.join(os.path.dirname(__file__), file_relative)
