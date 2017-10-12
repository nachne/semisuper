from itertools import islice

# helper

def flatten(l):
    return [item for sublist in l for item in sublist]

def take(n, iterable):
    return list(islice(iterable, n))


def identity(x):
    return x