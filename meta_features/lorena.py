import numpy
import multiprocessing
import networkx as nx
import category_encoders as ce
import pandas

from utils import *
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KDTree
from sklearn.feature_selection import f_regression, mutual_info_regression
from random import uniform, seed, randint
from scipy.stats.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
from joblib import Parallel, delayed
from scipy.stats import rankdata
from carlos import *
from utils import *


def max_feature_correlation_target(X: numpy.array, y: numpy.array, method: str='spearman') -> float:
    """
    Maximum feature correlation to the output.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.

    Return
    ------
    float:
        Maximum feature correlation to the output.
    """
    # check if not dataframe
    if not isinstance(X, pandas.DataFrame):
        X = pandas.DataFrame(X)  
    y = pandas.Series(y) 

    if method=='spearman':
        rho = spearmanr
    if method=='pearson':
        rho = pearsonr    

    return X.apply(lambda x: abs(rho(x, y).correlation)).max()


def c1(X: numpy.array, y: numpy.array, method: str='spearman') -> float:
    """
    Alias for max_feature_correlation_target (Lorena 2018).
    """

    return max_feature_correlation_target(X, y, method)


def c2(X: numpy.array, y: numpy.array, method: str='spearman') -> float:
    """Alias for mean_feature_correlation_target (Lorena 2018)."""

    return mean_feature_correlation_target(X, y, method)
    

def c3(X: numpy.array, y: numpy.array, n_jobs: int=1, correlation_threshhold: float=.9) -> float:
    """
    Individual feature efficiency.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.

    Return
    ------
    float:
        Minimum features elements droped to achieve correlation of 0.9 divided by n.
    """
    # check if is dataframe
    if isinstance(X, pandas.DataFrame):
        X = X.values

    # check if y is dataframe or series
    if isinstance(y, pandas.DataFrame) or isinstance(y, pandas.Series):
        y = y.values

    # Initial variables
    ncol = X.shape[1]
    n = X.shape[0]    
    n_j = list()

    rank_all_y = rankdata(y)
    rank_all_y_inv = rank_all_y[::-1]

    def removeCorrId(x_j: numpy.array, rank_all_y: numpy.array, rank_all_y_inv: numpy.array) -> int:
        """Calculate rank vectors to Spearman correlation."""
        rank_x = rankdata(x_j)
        rank_y = rank_all_y
        rank_dif = rank_x - rank_y
            
        def rho_spearman(d: numpy.array) -> float:
            """Calculate rho of Spearman."""    
            n = d.shape[0]        
            return 1 - 6 * (d**2).sum() / (n**3 - n) 
        
        if rho_spearman(rank_dif) < 0:
            rank_y = rank_all_y_inv
            rank_dif = rank_x - rank_y            

        while abs(rho_spearman(rank_dif)) <= correlation_threshhold:
            id_r = abs(rank_dif).argmax()
            rank_dif = rank_dif + (rank_y > rank_y[id_r]) - (rank_x > rank_x[id_r])            
            rank_dif = np.delete(rank_dif, id_r)
            rank_x = np.delete(rank_x, id_r)
            rank_y = np.delete(rank_y, id_r)

        return len(rank_dif)   
    
    num_cores = multiprocessing.cpu_count()
    n_jobs = (num_cores if (num_cores < n_jobs) or (n_jobs==-1) else n_jobs)
    n_j = Parallel(n_jobs=n_jobs)(delayed(removeCorrId)(X[:,col], rank_all_y, rank_all_y_inv) for col in range(ncol))
        
    return min(-np.array(n_j) + n) / n


def c4(X: numpy.array, y: numpy.array, min_resid: int=0.1, n_jobs: int=1) -> float:
    """
    Collective feature efficiency.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.

    Return
    ------
    float:
        Ratio between number of points that put residuos lower than 0.1
        and total number fo points.
    """
    # check if is dataframe
    if isinstance(X, pandas.DataFrame):
        X = X.values

    # check if y is dataframe or series
    if isinstance(y, pandas.DataFrame) or isinstance(y, pandas.Series):
        y = y.values

    A = list(range(X.shape[1]))
    n = X.shape[0]
    mcol = X.shape[1]
    num_cores = multiprocessing.cpu_count()
    n_jobs = (num_cores if (num_cores < n_jobs) or (n_jobs==-1) else n_jobs)

    def calculateCorr(x_j: numpy.array, y: numpy.array, j: int, A: list) -> tuple:
        """Calculate absolute Spearman correlation for x_j in set A."""
        corr = (abs(spearmanr(x_j, y)[0]) if j in A else .0)
    
        return (j, corr)  
       
    while A and X.any():
        pos_rho_list = Parallel(n_jobs=n_jobs)(delayed(calculateCorr)(X[:, j], y, j, A) for j in range(mcol))
        rho_list = [t[1] for t in sorted(pos_rho_list)]
        
        if sum(rho_list) == .0:
            break                    
        m = np.array(rho_list).max()
        A.remove(m)
        model = LinearRegression()
        x_j = X[:, m].reshape((-1, 1))
        y = y.reshape((-1, 1))
        model.fit(x_j, y)

        resid = y - model.predict(x_j)
        id_remove = abs(resid.flatten()) > min_resid
        X = X[id_remove, :]
        y = y[id_remove]
      
    return len(y) / n


def c5(X: numpy.array, y: numpy.array=None, method: str='spearman'):
    """
    Alias for mean_feature_correlation_target (Lorena 2018).
    """

    return mean_feature_correlation(X, y, method)


def s1(y: numpy.array, dist_matrix: numpy.array=None) -> float:
    """
    Output distribution.

    Parameters
    ----------
    y : numpy.array
        Array of response values.
    dist_matrix : numpy.array
        2d-array with euclidean distances between features.

    Return
    ------
    float:
        Normalized output distribution mean value.
    """
    if not dist_matrix.size():
        dist_matrix = numpy.corrcoef(X)

    G = nx.from_numpy_matrix(dist_matrix)
    T = nx.minimum_spanning_tree(G)
    edges = T.edges()
    edges_dist_norm = np.array([abs(y[i] - y[j]) for i, j in edges])

    return edges_dist_norm.sum() / len(edges)
        

def s2(X: numpy.array, y: numpy.array=None) -> float:
    """
    Input distribution.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.

    Return
    ------
    float:
        Normalized input distribution mean value.
    """
    # check if is dataframe
    if isinstance(X, pandas.DataFrame):
        X = X.values

    # check if y is dataframe or series
    if isinstance(y, pandas.DataFrame) or isinstance(y, pandas.Series):
        y = y.values

    X_y = np.hstack((X,y.reshape(X.shape[0], 1)))
    X = X_y[X_y[:, -1].argsort()][:, :-1]
    n = X.shape[0]    
    i = 1    
    d = list()

    while i < n:
        d.append(np.linalg.norm(X[i, :]-X[i-1, :]))
        i = i + 1

    return np.array(d).sum() / (n - 2)


def s3(X: numpy.array, y: numpy.array, dist_matrix: numpy.array=None, metric: str='mae') -> float:
    """
    Error of the nearest neighbor regressor.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.
    dist_matrix : numpy.array
        2d-array with euclidean distances between features.
    metric: str
        Error calculation metric.

    Return
    ------
    float:
        Normalized 1-NN mean error.
    """
    if not dist_matrix:
        dist_matrix = numpy.corrcoef(X)

    n = X.shape[0]    
    error = list()
     
    for i in range(n):
        i_nn = np.argmin(np.delete(dist_matrix[i, :], i))
        # Add 1 to i_nn in case equals to i
        if i==i_nn:
            i_nn = i_nn + 1
        error.append(y[i]-y[i_nn])

    return compute_metric(np.array(error), metric)


def s4(X: numpy.array, y: numpy.array, random_state: int=0, metric: str='mae') -> float:
    """
    Non-linearity of nearest neighbor regressor.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.
    random_state : int
        Seed to random calculations.
    metric: str
        Error calculation metric.        

    Return
    ------
    float:
        Normalized 1-NN error.
    """
    # check if is dataframe
    if isinstance(X, pandas.DataFrame):
        X = X.values

    # check if y is dataframe or series
    if isinstance(y, pandas.DataFrame) or isinstance(y, pandas.Series):
        y = y.values

    seed(random_state)
    tree = KDTree(X)
    n, m = X.shape
    y = y.flatten()
    idx_sorted_y = y.argsort()
    X_sorted = X[idx_sorted_y, :]
    y_sorted  = y[idx_sorted_y]
    i = 1
    X_list = list()
    y_list = list()

    while i < n:        
        x_i_list = list()
        for j in range(m):
            uniques_values = np.unique(X_sorted[:, j])
            if len(uniques_values) <= 2:
                x_i_list.append(randint(0, 1))
            else:
                x_i_list.append(uniform(X_sorted[i, j], X_sorted[i-1, j]))
        x_i = np.array(x_i_list)
        y_i = np.array([uniform(y_sorted[i], y_sorted[i-1])])
        
        X_list.append(x_i)
        y_list.append(y_i)
        i = i + 1

    X_ = np.array(X_list)
    y_ = np.array(y_list)   

    nearest_dist, nearest_ind = tree.query(X_, k=1)
    error = np.array([y[int(nearest_ind[i])]-y_[i] for i in range(y_.shape[0])])

    return compute_metric(error, metric)


def l1(X: numpy.array, y: numpy.array, model: LinearRegression=None) -> float:
    """
    Mean absolute error of OLS.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.
    model :np.array
       Linear regression model residuals between X,y.

    Return
    ------
    float:
        Mean absolute error
    """
    # check if is dataframe
    if isinstance(X, pandas.DataFrame):
        X = X.values

    # check if y is dataframe or series
    if isinstance(y, pandas.DataFrame) or isinstance(y, pandas.Series):
        y = y.values

    _, cat_idx = check_cat(X)
    X_ = np.delete(X, cat_idx, axis=1)
    if not model:
        model = LinearRegression().fit(X_, y)
    resid = y - model.predict(X_)

    return np.mean(abs(resid))


def l2(X: numpy.array, y: numpy.array) -> float:
    """
    Mean squared error of OLS.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns
    y : numpy.array
        Array of response values
    model :np.array
       Linear regression model residuals between X,y
    Return
    ------
    float:
        Mean squared error
    """
    # check if is dataframe
    if isinstance(X, pandas.DataFrame):
        X = X.values

    # check if y is dataframe or series
    if isinstance(y, pandas.DataFrame) or isinstance(y, pandas.Series):
        y = y.values

    _, cat_idx = check_cat(X)
    X_ = np.delete(X, cat_idx, axis=1)
    if not model:
        model = LinearRegression().fit(X_, y)
    resid = y - model.predict(X_)

    # Normalize squared residuous
    res_norm = resid ** 2
    
    return np.mean(res_norm)


def l3(X: numpy.array, y: numpy.array, model: LinearRegression=None, random_state: int=0, metric: str='mse') -> float:
    """
    Calculate the non-linearity of a linear regressor

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns
    y : numpy.array
        Array of response values
    model : LinearRegression
        Ordinary least square model between X,y
    random_state : int
        Seed to random calculations

    Return
    ------
    float:
        Normalized mean error
    """
    # check if is dataframe
    if isinstance(X, pandas.DataFrame):
        X = X.values

    # check if y is dataframe or series
    if isinstance(y, pandas.DataFrame) or isinstance(y, pandas.Series):
        y = y.values

    _, cat_idx = check_cat(X)
    X_ = np.delete(X, cat_idx, axis=1)
    if not model:
        model = LinearRegression().fit(X_, y)    

    seed(random_state) 
    n, m = X.shape
    y = y.flatten()
    idx_sorted_y = y.argsort()
    X_sorted = X[idx_sorted_y, :]
    y_sorted  = y[idx_sorted_y]
    i = 1
    X_list = list()
    y_list = list()

    while i < n:        
        x_i_list = list()
        for j in range(m):
            uniques_values = np.unique(X_sorted[:, j])
            if len(uniques_values) <= 2:
                x_i_list.append(randint(0, 1))
            else:
                x_i_list.append(uniform(X_sorted[i, j], X_sorted[i-1, j]))
        x_i = np.array(x_i_list)
        y_i = np.array([uniform(y_sorted[i], y_sorted[i-1])])
        
        X_list.append(x_i)
        y_list.append(y_i)
        i = i + 1

    X_ = np.array(X_list)
    y_ = np.array(y_list)   
    error = model.predict(X_).reshape((n-1,)) - np.array(y_).reshape((n-1,))

    return compute_metric(error, metric)


def t2(X: numpy:array, y: numpy:array=None) -> float:
    """
    Number of examples per dimension (n / m).

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns

    Return
    ------
    float:
        Ratio between number of examples and number of features
    """

    return X.shape[0] / X.shape[1]


