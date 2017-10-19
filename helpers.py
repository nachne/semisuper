from itertools import islice
from functools import reduce
import operator

# helper

def flatten(l):
    return [item for sublist in l for item in sublist]

def take(n, iterable):
    return list(islice(iterable, n))

def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def identity(x):
    return x

def positive(x):
    return x > 0