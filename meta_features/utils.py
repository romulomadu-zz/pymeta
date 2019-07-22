import pandas


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
        return np.abs(np.array(arr)).sum() / n
    if metric == 'mse':
        return (np.array(arr) ** 2).sum() / (n - 1)
    if metric == 'rmse':
        return (np.sqrt(np.array(arr) ** 2)).sum() / n


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
        n_unique = len(np.unique(X[:, i]))
        if n_unique==2:
            n_cat = n_cat + 1
            cat_idx.append(i)

    return n_cat, cat_idx

