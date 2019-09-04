import numpy
import multiprocessing
import networkx as nx
import pandas
import sys

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
from os.path import join
from pathlib import Path

project_dir = Path(__file__).resolve().parents[1]
sys.path.append(join(project_dir, 'pymeta', 'meta_features'))
from .utils import *
from .carlos import *


def max_feature_correlation_target(X: numpy.array, y: numpy.array, method: str='spearman') -> float:
    """
    Maximum feature correlation to the output.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.
    method : {'pearson', 'kendall', 'spearman'} or callable
        * pearson : standard correlation coefficient
        * kendall : Kendall Tau correlation coefficient
        * spearman : Spearman rank correlation
        * callable: callable with input two 1d ndarrays
            and returning a float. Note that the returned matrix from corr
            will have 1 along the diagonals and will be symmetric
            regardless of the callable's behavior
            .. versionadded:: 0.24.0

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
    Alias for ``max_feature_correlation_target`` (Lorena 2018).

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.
    method : {'pearson', 'kendall', 'spearman'} or callable
        * pearson : standard correlation coefficient
        * kendall : Kendall Tau correlation coefficient
        * spearman : Spearman rank correlation
        * callable: callable with input two 1d ndarrays
            and returning a float. Note that the returned matrix from corr
            will have 1 along the diagonals and will be symmetric
            regardless of the callable's behavior
            .. versionadded:: 0.24.0
    Return
    ------
    float:
        Maximum feature correlation to the output.
    """

    return max_feature_correlation_target(X, y, method)


def c2(X: numpy.array, y: numpy.array, method: str='spearman') -> float:
    """
    Alias for ``mean_feature_correlation_target`` (Lorena 2018).

    Parameters
    ----------    
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.
    method : {'pearson', 'kendall', 'spearman'} or callable
        * pearson : standard correlation coefficient
        * kendall : Kendall Tau correlation coefficient
        * spearman : Spearman rank correlation
        * callable: callable with input two 1d ndarrays
            and returning a float. Note that the returned matrix from corr
            will have 1 along the diagonals and will be symmetric
            regardless of the callable's behavior
            .. versionadded:: 0.24.0

    Return
    ------
    float:
        Average feature correlation to the output.
    """

    return mean_feature_correlation_target(X, y, method)
    

def individual_feature_efficiency(X: numpy.array, y: numpy.array, correlation_threshhold: float=.9, n_jobs: int=None) -> float:
    """
    Individual feature efficiency. 
    Calculates, for each feature, the number of examples that must be removed from the dataset until a high correlation value to the output is achieved.
    Lower values of C3 indicate simpler problems.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.
    correlation_threshhold : float, optional (default=0.9)
        Threshold which correlation is already significant.
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel. None means 1 unless. -1 means using all processors.

    Return
    ------
    float:
        Minimum number of observations removed from dataset over total number of observations.
        
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
            rank_dif = numpy.delete(rank_dif, id_r)
            rank_x = numpy.delete(rank_x, id_r)
            rank_y = numpy.delete(rank_y, id_r)

        return len(rank_dif)   
    
    num_cores = multiprocessing.cpu_count()
    n_jobs = (1 if not n_jobs else (num_cores if (num_cores < n_jobs) or (n_jobs==-1) else n_jobs))
    n_j = Parallel(n_jobs=n_jobs)(delayed(removeCorrId)(X[:,col], rank_all_y, rank_all_y_inv) for col in range(ncol))
        
    return min(-numpy.array(n_j) + n) / n


def c3(X: numpy.array, y: numpy.array, correlation_threshhold: float=.9, n_jobs: int=None) -> float:
    """
    Alias for ``individual_feature_efficiency`` (Lorena 2018).

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.
    correlation_threshhold : float, optional (default=0.9)
        Threshold which correlation is already significant.
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel. None means 1 unless. -1 means using all processors.

    Return
    ------
    float:
        Minimum number of observations removed from dataset over total number of observations.
        
    """

    return individual_feature_efficiency(x, y, correlation_threshhold, n_jobs)


def collective_feature_efficiency(X: numpy.array, y: numpy.array, min_resid: float=0.1, n_jobs: int=None) -> float:
    """
    Collective feature efficiency.
    C4 starts by identifying the feature with highest correlation to the output. 
    All examples with asmall residual value (|εi|≤0.1) after a linear fit between this feature and the target attribute are removed. 
    Then, the most correlated feature to the remaining data points is found and the previous process is repeated until all features have been analyzed or no example remains.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.
    min_resid : float, optional (default=0.1)
        Minimum residual value for observation remotion.        
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel. None means 1 unless. -1 means using all processors.

    Return
    ------
    float:
        The number of observations that put residuos lower than 0.1
        over total number of observations.
    """
    # check if is dataframe
    X = (X.values if isinstance(X, pandas.DataFrame) else X)

    # check if y is dataframe or series
    y = (y.values if isinstance(y, pandas.DataFrame) or isinstance(y, pandas.Series) else y)

    # initial parameters
    A = list(range(X.shape[1]))
    n = X.shape[0]
    mcol = X.shape[1]
    num_cores = multiprocessing.cpu_count()
    n_jobs = (1 if not n_jobs else (num_cores if (num_cores < n_jobs) or (n_jobs==-1) else n_jobs))

    def calculateCorr(x_j: numpy.array, y: numpy.array, j: int, A: list) -> tuple:
        """Calculate absolute Spearman correlation for x_j in set A."""
        corr = (abs(spearmanr(x_j, y)[0]) if j in A else .0)
    
        return (j, corr)  
       
    while A and X.any():
        pos_rho_list = Parallel(n_jobs=n_jobs)(delayed(calculateCorr)(X[:, j], y, j, A) for j in range(mcol))
        rho_list = [t[1] for t in sorted(pos_rho_list)]
        
        if sum(rho_list) == .0:
            break                    
        m = numpy.ndarray.argmax(numpy.array(rho_list))
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


def c4(X: numpy.array, y: numpy.array, min_resid: float=0.1, n_jobs: int=None) -> float:
    """
    Alias for ``collective_feature_efficiency``.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.
    min_resid : float, optional (default=0.1)
        Minimum residual value for observation remotion.        
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel. None means 1 unless. -1 means using all processors.

    Return
    ------
    float:
        The number of observations that put residuos lower than 0.1
        over total number of observations.
    """
   
    return collective_feature_efficiency(X, y, min_resid, n_jobs)


def c5(X: numpy.array, y: numpy.array=None, method: str='spearman'):
    """
    Alias for ``mean_feature_correlation`` (Lorena 2018).

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.
    method : {'pearson', 'kendall', 'spearman'} or callable
        * pearson : standard correlation coefficient
        * kendall : Kendall Tau correlation coefficient
        * spearman : Spearman rank correlation
        * callable: callable with input two 1d ndarrays
            and returning a float. Note that the returned matrix from corr
            will have 1 along the diagonals and will be symmetric
            regardless of the callable's behavior
            .. versionadded:: 0.24.0

    Return
    ------
    float:
        Average feature correlation between features.
    """

    return mean_feature_correlation(X, y, method)


def output_distribution(X: numpy.array, y: numpy.array, dist_matrix: numpy.array=None) -> float:
    """
    Output distribution.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.    
    y : numpy.array
        Array of response values.
    dist_matrix : numpy.array
        2d-array with euclidean distances between features.

    Return
    ------
    float:
        Normalized output distribution mean value.
    """
    if not isinstance(dist_matrix, numpy.ndarray):
        dist_matrix = numpy.corrcoef(X)

    G = nx.from_numpy_matrix(dist_matrix)
    T = nx.minimum_spanning_tree(G)
    edges = T.edges()
    edges_dist_norm = numpy.array([abs(y[i] - y[j]) for i, j in edges])

    return edges_dist_norm.sum() / len(edges)


def s1(X: numpy.array, y: numpy.array, dist_matrix: numpy.array=None) -> float:
    """
    Alias for ``output_distribution``.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.
    dist_matrix : numpy.array
        2d-array with euclidean distances between features.

    Return
    ------
    float:
        Normalized output distribution mean value.
    """

    return output_distribution(X, y, dist_matrix)
        

def input_distribution(X: numpy.array, y: numpy.array) -> float:
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

    X_y = numpy.hstack((X,y.reshape(X.shape[0], 1)))
    X = X_y[X_y[:, -1].argsort()][:, :-1]
    n = X.shape[0]    
    i = 1    
    d = list()

    while i < n:
        d.append(numpy.linalg.norm(X[i, :]-X[i-1, :]))
        i = i + 1

    return numpy.array(d).sum() / (n - 2)


def s2(X: numpy.array, y: numpy.array=None) -> float:
    """
    Alias for ``input_distribution``.

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

    return input_distribution(X, y)


def error_of_nn_regressor(X: numpy.array, y: numpy.array, dist_matrix: numpy.array=None, metric: str='mae') -> float:
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
    metric: str, optional (default='mae')
        Error calculation metric.  

    Return
    ------
    float:
        Normalized 1-NN mean error.
    """
    if not isinstance(dist_matrix, numpy.ndarray):
        dist_matrix = numpy.corrcoef(X)

    n = X.shape[0]    
    error = list()
     
    for i in range(n):
        i_nn = numpy.argmin(numpy.delete(dist_matrix[i, :], i))
        # Add 1 to i_nn in case equals to i
        if i==i_nn:
            i_nn = i_nn + 1
        error.append(y[i]-y[i_nn])

    return compute_metric(numpy.array(error), metric)


def s3(X: numpy.array, y: numpy.array, dist_matrix: numpy.array=None, metric: str='mae') -> float:
    """
    Alias for ``error_of_nn_regressor``.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.
    dist_matrix : numpy.array
        2d-array with euclidean distances between features.
    metric: str, optional (default='mae')
        Error calculation metric.  

    Return
    ------
    float:
        Normalized 1-NN mean error.
    """
    
    return error_of_nn_regressor(X, y, dist_matrix, metric)


def nonlinearity_of_nn_regressor(X: numpy.array, y: numpy.array, random_state: int=None, metric: str='mae') -> float:
    """
    Non-linearity of nearest neighbor regressor.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.
    random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `numpy.random`.
    metric: str, optional (default='mae')
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

    seed_ = check_random_state(random_state)
    seed(seed_)
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
            uniques_values = numpy.unique(X_sorted[:, j])
            if len(uniques_values) <= 2:
                x_i_list.append(randint(0, 1))
            else:
                x_i_list.append(uniform(X_sorted[i, j], X_sorted[i-1, j]))
        x_i = numpy.array(x_i_list)
        y_i = numpy.array([uniform(y_sorted[i], y_sorted[i-1])])
        
        X_list.append(x_i)
        y_list.append(y_i)
        i = i + 1

    X_ = numpy.array(X_list)
    y_ = numpy.array(y_list)   

    nearest_dist, nearest_ind = tree.query(X_, k=1)
    error = numpy.array([y[int(nearest_ind[i])]-y_[i] for i in range(y_.shape[0])])

    return compute_metric(error, metric)


def s4(X: numpy.array, y: numpy.array, random_state: int=None, metric: str='mae') -> float:
    """
    Alias for ``nonlinearity_of_nn_regressor``.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.
    random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `numpy.random`.
    metric: str, optional (default='mae')
        Error calculation metric.        

    Return
    ------
    float:
        Normalized 1-NN error.
    """

    return nonlinearity_of_nn_regressor(X, y, random_state)


def mean_absolute_residuos(X: numpy.array, y: numpy.array, model: LinearRegression=None, ignore_categorical: bool=False) -> float:
    """
    Mean absolute error of OLS.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns
    y : numpy.array
        Array of response values
    model : LinearRegression 
       Linear regression model between X,y
    ignore_categorical : bool
        Flag to ignore categorical features in OLS regression 

    Return
    ------
    float:
        Average absolute error
    """
    # check if is dataframe
    X = (X.values if isinstance(X, pandas.DataFrame) else X)

    # check if y is dataframe or series
    y = (y.values if isinstance(y, pandas.DataFrame) or isinstance(y, pandas.Series) else y)
    
    if ignore_categorical:
        _, cat_idx = check_cat(X)
        X_ = numpy.delete(X, cat_idx, axis=1)
        model = LinearRegression().fit(X_, y)
    else:
        model = (LinearRegression().fit(X, y) if not model else model)
    resid = y - model.predict(X)

    return numpy.mean(abs(resid))


def l1(X: numpy.array, y: numpy.array, model: LinearRegression=None) -> float:
    """
    Mean absolute error of OLS.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.
    model :numpy.array
       Linear regression model residuals between X,y.

    Return
    ------
    float:
        Average absolute error
    """

    return mean_absolute_residuos(X, y, model)


def mean_squared_residuos(X: numpy.array, y: numpy.array, model: LinearRegression=None, ignore_categorical: bool=False) -> float:
    """
    Mean squared error of OLS.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns
    y : numpy.array
        Array of response values
    model : LinearRegression 
       Linear regression model between X,y
    ignore_categorical : bool
        Flag to ignore categorical features in OLS regression 

    Return
    ------
    float:
        Average squared error
    """
    # check if is dataframe
    X = (X.values if isinstance(X, pandas.DataFrame) else X)

    # check if y is dataframe or series
    y = (y.values if isinstance(y, pandas.DataFrame) or isinstance(y, pandas.Series) else y)

    if ignore_categorical:
        _, cat_idx = check_cat(X)
        X_ = numpy.delete(X, cat_idx, axis=1)
        model = LinearRegression().fit(X_, y)
    else:
        model = (LinearRegression().fit(X, y) if not model else model)
    resid = y - model.predict(X)

    # Normalize squared residuous
    res_norm = resid ** 2
    
    return numpy.mean(res_norm)


def l2(X: numpy.array, y: numpy.array, model: LinearRegression=None) -> float:
    """
    Alias for ``mean_squared_residuos``.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns
    y : numpy.array
        Array of response values
    model :numpy.array
       Linear regression model between X,y

    Return
    ------
    float:
        Average squared error
    """
    
    return mean_squared_residuos(X, y, model)


def nonlinearity_of_linear_regressor(X: numpy.array, y: numpy.array, model: LinearRegression=None, random_state: int=None, metric: str='mse') -> float:
    """
    Calculate the non-linearity of a linear regressor

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns
    y : numpy.array
        Array of response values
    model : LinearRegression
        Ordinary least square model between X,y, in case of already trained model.
    random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `numpy.random`.
    metric: str, optional (default='mae')
        Error calculation metric.  

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
    X_ = numpy.delete(X, cat_idx, axis=1)    
    model = (LinearRegression().fit(X_, y) if not model else model)

    seed_ = check_random_state(random_state)
    seed(seed_) 
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
            uniques_values = numpy.unique(X_sorted[:, j])
            if len(uniques_values) <= 2:
                x_i_list.append(randint(0, 1))
            else:
                x_i_list.append(uniform(X_sorted[i, j], X_sorted[i-1, j]))
        x_i = numpy.array(x_i_list)
        y_i = numpy.array([uniform(y_sorted[i], y_sorted[i-1])])
        
        X_list.append(x_i)
        y_list.append(y_i)
        i = i + 1

    X_ = numpy.array(X_list)
    y_ = numpy.array(y_list)   
    error = model.predict(X_).reshape((n-1,)) - numpy.array(y_).reshape((n-1,))

    return compute_metric(error, metric)


def l3(X: numpy.array, y: numpy.array, model: LinearRegression=None, random_state: int=None, metric: str='mse') -> float:
    """
    Alias for ``nonlinearity_of_linear_regressor``.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns
    y : numpy.array
        Array of response values
    model : LinearRegression
        Ordinary least square model between X,y, in case of already trained model.
    random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `numpy.random`.
    metric: str, optional (default='mae')
        Error calculation metric.  

    Return
    ------
    float:
        Normalized mean error
    """
   
    return nonlinearity_of_linear_regressor(X, y, model)


def t2(X: numpy.array, y: numpy.array=None) -> float:
    """
    Alias for ``example_features_ratio``.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns

    Return
    ------
    float:
        Ratio between number of examples and number of features
    """

    return example_features_ratio(X, y)


