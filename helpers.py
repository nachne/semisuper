from itertools import islice
from functools import reduce
from operator import itemgetter, mul
from numpy import shape, array, nonzero, ones


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