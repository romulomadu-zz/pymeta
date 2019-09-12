import pandas
import numpy
import numbers


def IQR(X) -> tuple:
    '''Inter-quantile range.'''

    if not isinstance(X, pandas.DataFrame):
        X = pandas.DataFrame(X)
    
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)

    return Q3 - Q1, Q1, Q3


def compute_metric(arr, metric):
    """Compute normalized chosen metric."""
    n = max(arr.shape)
    # Select metric to return
    if metric == 'mae':
        return numpy.abs(numpy.array(arr)).sum() / n
    if metric == 'mse':
        return (numpy.array(arr) ** 2).sum() / (n - 1)
    if metric == 'rmse':
        return (numpy.sqrt(numpy.array(arr) ** 2)).sum() / n


def min_max(x):
    """Min-max scaler."""    
    min_ = x.min()
    max_ = x.max()
    if min_ == max_:
        return x / min_
    return (x - min_) / (max_- min_)


def check_cat(X):
    m = X.shape[1]
    n = X.shape[0]
    n_cat = 0
    cat_idx = list()
    for i in range(m):
        n_unique = len(numpy.unique(X[:, i]))
        if n_unique==2:
            n_cat = n_cat + 1
            cat_idx.append(i)

    return n_cat, cat_idx

# Credits: scikit-learn project.
# https://github.com/scikit-learn/scikit-learn
def check_random_state(seed):
    """Turn seed into a numpy.random.RandomState instance.

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by numpy.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is numpy.random:
        return numpy.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return numpy.random.RandomState(seed)
    if isinstance(seed, numpy.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

