from itertools import islice
from functools import reduce
from operator import itemgetter, mul
from numpy import shape

# helper

def flatten(l):
    return [item for sublist in l for item in sublist]

def take(n, iterable):
    return list(islice(iterable, n))

def prod(iterable):
    return reduce(mul, iterable, 1)

def identity(x):
    return x

def positive(x):
    return x > 0

def num_rows(a):
    return shape(a)[0]