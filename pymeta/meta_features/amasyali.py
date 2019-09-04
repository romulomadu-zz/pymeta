import numpy
import pandas
import sys

from os.path import join
from pathlib import Path
from scipy.stats.stats import pearsonr, spearmanr

project_dir = Path(__file__).resolve().parents[1]
sys.path.append(join(project_dir, 'pymeta', 'meta_features'))
from .utils import *


def proportion_of_outliers_target(y: numpy.array) -> float:
    '''
    Proportion of outliers on target.

    Parameters
    ----------    
    y : numpy.array
        Array of response values.

    Return
    ------
    float:
        number of outliers over total number of observations            
    '''

    iqr, q1, q3 = IQR(y)
    is_outlier = (y < (q1 - 1.5 * iqr)[0]) | (y > (q3 + 1.5 * iqr)[0])

    return is_outlier.sum() / len(y)


def proportion_of_binary_features(X: numpy.array, y: numpy.array=None) -> float:
    '''
    Proportion of binary features.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.

    Return
    ------
    float:
        Number of binary features over total number of features.    
    '''

    X = (X if isinstance(X, pandas.DataFrame) else pandas.DataFrame(X))
    is_binary = X.apply(lambda x: len(numpy.unique(x))) == 2

    return sum(is_binary) / X.shape[1]


def min_kurtosis_numerical_features(X: numpy.array, y: numpy.array=None) -> float:
    '''
    Minimum kurtosis of numerical features.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.

    Return
    ------
    float:
        Minimum kurtosis.    
    '''

    X = (X if isinstance(X, pandas.DataFrame) else pandas.DataFrame(X))
    numerical_features = numpy.where(X.dtypes == numpy.float)[0]
    
    return X.iloc[:, numerical_features].kurtosis().min()


def max_kurtosis_numerical_features(X: numpy.array, y: numpy.array=None) -> float:
    '''
    Maximum kurtosis of numerical features.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.

    Return
    ------
    float:
        Maximum kurtosis.    
    '''

    X = (X if isinstance(X, pandas.DataFrame) else pandas.DataFrame(X))
    numerical_features = numpy.where(X.dtypes == numpy.float)[0]
    
    return X.iloc[:, numerical_features].kurtosis().max()


def mean_kurtosis_numerical_features(X: numpy.array, y: numpy.array=None) -> float:
    '''
    Mean kurtosis of numerical features.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.

    Return
    ------
    float:
        Average kurtosis.    
    '''

    X = (X if isinstance(X, pandas.DataFrame) else pandas.DataFrame(X))
    numerical_features = numpy.where(X.dtypes == numpy.float)[0]
    
    return X.iloc[:, numerical_features].kurtosis().mean()


def min_skewness_numerical_features(X: numpy.array, y: numpy.array=None) -> float:
    '''
    Minimum skewness of numerical features.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.

    Return
    ------
    float:
        Minimum skewness.    
    '''

    X = (X if isinstance(X, pandas.DataFrame) else pandas.DataFrame(X))
    numerical_features = numpy.where(X.dtypes == numpy.float)[0]
    
    return X.iloc[:, numerical_features].skew().min()


def max_skewness_numerical_features(X: numpy.array, y: numpy.array=None) -> float:
    '''
    Maximum skewness of numerical features.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.

    Return
    ------
    float:
        Maximum skewness.    
    '''

    X = (X if isinstance(X, pandas.DataFrame) else pandas.DataFrame(X))
    numerical_features = numpy.where(X.dtypes == numpy.float)[0]
    
    return X.iloc[:, numerical_features].skew().max()


def mean_skewness_numerical_features(X: numpy.array, y: numpy.array=None) -> float:
    '''
    Mean skewness of numerical features.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.

    Return
    ------
    float:
        Average skewness.    
    '''
    
    X = (X if isinstance(X, pandas.DataFrame) else pandas.DataFrame(X))
    numerical_features = numpy.where(X.dtypes == numpy.float)[0]
    
    return X.iloc[:, numerical_features].skew().mean()


def min_mean_numerical_features(X: numpy.array, y: numpy.array=None) -> float:
    '''
    Minimum mean of numerical features.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.

    Return
    ------
    float:
        Minimum mean.    
    '''

    X = (X if isinstance(X, pandas.DataFrame) else pandas.DataFrame(X))
    numerical_features = numpy.where(X.dtypes == numpy.float)[0]
    
    return X.iloc[:, numerical_features].mean().min()


def max_mean_numerical_features(X: numpy.array, y: numpy.array=None) -> float:
    '''
    Maximum mean of numerical features.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.

    Return
    ------
    float:
        Maximum mean.    
    '''

    X = (X if isinstance(X, pandas.DataFrame) else pandas.DataFrame(X))
    numerical_features = numpy.where(X.dtypes == numpy.float)[0]
    
    return X.iloc[:, numerical_features].mean().max()


def mean_mean_numerical_features(X: numpy.array, y: numpy.array=None) -> float:
    '''
    Mean mean of numerical features.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.

    Return
    ------
    float:
        Average mean.    
    '''
    
    X = (X if isinstance(X, pandas.DataFrame) else pandas.DataFrame(X))
    numerical_features = numpy.where(X.dtypes == numpy.float)[0]
    
    return X.iloc[:, numerical_features].mean().mean()


def min_std_numerical_features(X: numpy.array, y: numpy.array=None) -> float:
    '''
    Minimum standard deviation of numerical features.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.

    Return
    ------
    float:
        Minimum standard deviation.    
    '''

    X = (X if isinstance(X, pandas.DataFrame) else pandas.DataFrame(X))
    numerical_features = numpy.where(X.dtypes == numpy.float)[0]
    
    return X.iloc[:, numerical_features].std().min()


def max_std_numerical_features(X: numpy.array, y: numpy.array=None) -> float:
    '''
    Maximum standard deviation of numerical features.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.

    Return
    ------
    float:
        Maximum standard deviation.    
    '''

    X = (X if isinstance(X, pandas.DataFrame) else pandas.DataFrame(X))
    numerical_features = numpy.where(X.dtypes == numpy.float)[0]
    
    return X.iloc[:, numerical_features].std().max()


def mean_std_numerical_features(X: numpy.array, y: numpy.array=None) -> float:
    '''
    Mean standard deviation of numerical features.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.

    Return
    ------
    float:
        Average standard deviation.    
    '''
    
    X = (X if isinstance(X, pandas.DataFrame) else pandas.DataFrame(X))
    numerical_features = numpy.where(X.dtypes == numpy.float)[0]
    
    return X.iloc[:, numerical_features].std().mean()


def proportion_of_features_with_na(X: numpy.array, y: numpy.array=None) -> float:
    '''
    Proportion of features with missing values.

    Parameters
    ----------
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.

    Return
    ------
    float:
        Number of features with missing values over total number of features.    
    '''

    X = (X if isinstance(X, pandas.DataFrame) else pandas.DataFrame(X))
    has_na = X.isnull().any()
    
    return has_na.sum() / X.shape[1]


def proportion_of_correlated_features_target(X: numpy.array, y: numpy.array=None, method: str='spearman', coef: float=0.5) -> float:
    '''
    Proportion of features correlated with target.

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
    coef : float, optional (default=0.5)
        Threshold to mean correlated.

    Return
    ------
    float:
        Number of features correlated to target over total number of features.    
    '''
    # check if not dataframe
    X = (X if isinstance(X, pandas.DataFrame) else pandas.DataFrame(X))
    y = pandas.Series(y) 

    if method=='spearman':
        rho = spearmanr
    if method=='pearson':
        rho = pearsonr    

    is_corr = X.apply(lambda x: abs(rho(x, y).correlation)) > coef

    return is_corr.sum() / X.shape[1]


def proportion_of_correlated_features(X: numpy.array, y: numpy.array=None, method: str='spearman', coef: float=0.5) -> float:
    '''
    Proportion of features correlated.

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
    coef : float, optional (default=0.5)
        Threshold to mean correlated.

    Return
    ------
    float:
        Number of features correlated over total number of features.   
    '''
    # check if not dataframe
    X = (X if isinstance(X, pandas.DataFrame) else pandas.DataFrame(X))

    # get numerical features indexes and filt.
    numerical_features = numpy.where(X.dtypes == numpy.float)[0]
    X_numerical = X.iloc[:, numerical_features]
    m = X_numerical.shape[1]

    # get correlation matrix.
    correlation_matrix = X_numerical.corr(method=method).values
    ids = numpy.triu_indices(m, k=1)

    is_corr = numpy.abs(correlation_matrix[ids]) > coef
    
    return is_corr.sum() / X.shape[1]




    



