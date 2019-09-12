import numpy
import pandas
import sys
import category_encoders as ce

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats.stats import pearsonr, spearmanr
from os.path import join
from pathlib import Path

project_dir = Path(__file__).resolve().parents[1]
sys.path.append(join(project_dir, 'pymeta', 'meta_features'))
from .utils import *


def n_of_examples(X: numpy.array, y: numpy.array=None) -> int:
    '''
    Number of examples in a dataset.
    
    Parameters
    ----------    
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.

    Return
    ------
    int:
        Number of observations.
    '''

    return X.shape[0]


def n(X: numpy.array, y: numpy.array=None) -> int:
    '''
    Alias for ``n_of_examples``.
    
    Parameters
    ----------    
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.

    Return
    ------
    int:
        Number of observations.
    '''

    return X.shape[0]


def n_of_features(X: numpy.array, y: numpy.array=None) -> int:
    '''
    Number of attributes in a dataset.
    
    Parameters
    ----------    
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.

    Return
    ------
    int:
        Number of features.    
    '''

    return X.shape[1]


def m(X: numpy.array, y: numpy.array=None) -> int:
    '''
    Alias for ``n_of_features``.
    
    Parameters
    ----------    
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.

    Return
    ------
    int:
        Number of features. 
    '''

    return X.shape[1]


def proportion_of_categorical(X: numpy.array, y: numpy.array=None, categorical_mask: list=None) -> float:
    '''
    Proportion of categorical features in a dataset.

    Parameters
    ----------    
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.
    categorical_mask : list, optional(default=None)
        List of bool mask for categorical columns.

    Return
    ------
    float:
        Number of categorical features over total number of features. 
    '''
    
    # check if categorical mask was passed
    if isinstance(categorical_mask, list):
        return sum(categorical_mask) / len(categorical_mask)
    else:

        # check if not dataframe
        X = (pandas.DataFrame(X) if not isinstance(X, pandas.DataFrame) else X)
    
        # get number of categorical columns
        n_categorical = sum(X.dtypes != numpy.float)
        
        return n_categorical / n_of_features(X)


def example_features_ratio(X: numpy.array, y: numpy.array=None) -> float:
    '''
    Ratio of the number of examples to the number of features.

    Parameters
    ----------    
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.

    Return
    ------
    float:
        Number of observations over number of features. 
    
    '''

    return n_of_examples(X) / n_of_features(X)


def proportion_of_attributes_outliers(X: numpy.array, y: numpy.array=None) -> float:
    '''
    Propotion of attributes with outliers.
    
    Parameters
    ----------    
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.

    Return
    ------
    float:
        Number of features with outliers over total number of features. 

    '''
  
    # check if not dataframe
    X = (pandas.DataFrame(X) if not isinstance(X, pandas.DataFrame) else X)
    
    # get number of columns with outliers
    iqr, q1, q3 = IQR(X)
    is_outlier = (X < (q1 - 1.5 * iqr)) | (X > (q3 + 1.5 * iqr))
    m_with_outlier = is_outlier.any(axis=0).sum()

    # get number of numeric columns
    m_numeric = sum(X.dtypes == numpy.float)

    return m_with_outlier / m_numeric


def coeficient_of_variation_target(y: numpy.array) -> float:
    '''
    Coefficient of variation of the target (ratio of the standard deviation to the mean).
    
    Parameters
    ----------    
    y : numpy.array
        Array of response values.

    Return
    ------
    float:
        Standard deviation of the target over mean;
    
    '''

    return numpy.std(y) / numpy.mean(y)

#TODO
# def sparsity_of_target(y: numpy.array) -> int:
#     '''Sparsity of the target. (Coeficient of variation of the target discretized into three values).'''
#     pass


def outliers_on_target(y: numpy.array) -> int:
    '''
    Presence of outliers in the target.
    
    Parameters
    ----------    
    y : numpy.array
        Array of response values.

    Return
    ------
    int:
        1 if there is outliers in target, 0 instead.

    '''
    iqr, q1, q3 = IQR(y)
    is_outlier = (y < (q1 - 1.5 * iqr)[0]) | (y > (q3 + 1.5 * iqr)[0])

    return int(is_outlier.sum() > 0)


def stationarity_of_target(y: numpy.array) -> int:
    '''
    Stationarity of the target (check if the standard deviation is larger than mean).
    
    Parameters
    ----------    
    y : numpy.array
        Array of response values.

    Return
    ------
    int:
        1 if the standard deviation of the target is larger than mean, 0 instead.

    '''

    return int(y.std() > y.mean())


def r2_without_categorical(X: numpy.array, y: numpy.array, model: LinearRegression=None) -> float:
    '''
    R2 coefficient of linear regression (without categorical attributes).

    Parameters
    ----------    
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.
    model : LineaRegression, optional(default=None)
        Linear model pre fitted to pass.

    Return
    ------
    float:
        R2 score.
    
    '''

    # check if not dataframe.
    X = (pandas.DataFrame(X) if not isinstance(X, pandas.DataFrame) else X)
    
    # check if categorical mask was passed and get numerical features.
    numerical_features = (X.dtypes == numpy.float)

    # get numerical features filt.
    X_numerical = X.loc[:, numerical_features]

    # train model and make prediction.
    model = (LinearRegression().fit(X_numerical, y) if not isinstance(model, LinearRegression) else model)
    y_pred = model.predict(X_numerical)

    return r2_score(y, y_pred)


def r2_with_binarized_categorical(X: numpy.array, y: numpy.array=None, categorical_mask: list=None) -> float:
    '''
    R2 coefficient of linear regression (without categorical attributes).
    
    Parameters
    ----------    
    X : numpy.array
        2d-array with features columns.
    y : numpy.array
        Array of response values.
    categorical_mask : list, optional(default=None)
        List of bool mask for categorical columns.

    Return
    ------
    float:
        R2 score. 
    '''

    # check if not dataframe
    X = (pandas.DataFrame(X) if not isinstance(X, pandas.DataFrame) else X)
    
    # check if categorical mask was passed.
    categorical_mask = (categorical_mask if isinstance(categorical_mask, list) else (X.dtypes != numpy.float))
    
    # binarize categorical data    
    categorical_index = numpy.where(categorical_mask)[0]
    if len(categorical_index) > 0:
        binary = ce.BinaryEncoder(cols=categorical_index)
        X = binary.fit_transform(X.values)

    # train model and make prediction.
    model  = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    
    return r2_score(y, y_pred)


def mean_feature_correlation(X: numpy.array, y: numpy.array=None, method: str='spearman') -> float:
    '''
    Average correlation between the numeric features.
    
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
        Mean correlation between the features.   
    
    '''

    # check if not dataframe
    X = (pandas.DataFrame(X) if not isinstance(X, pandas.DataFrame) else X)
    
    # get numerical features indexes and filt.
    numerical_features = numpy.where(X.dtypes == numpy.float)[0]
    X_numerical = X.iloc[:, numerical_features]
    m = X_numerical.shape[1]

    # get correlation matrix.
    correlation_matrix = X_numerical.corr(method=method).values
    ids = numpy.triu_indices(m, k=1)

    return numpy.abs(correlation_matrix[ids]).mean()


def mean_feature_correlation_target(X: numpy.array, y: numpy.array, method: str='spearman') -> float:
    """
    Average feature correlation to the output.

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
    # check if not dataframe
    X = (pandas.DataFrame(X) if not isinstance(X, pandas.DataFrame) else X)
    
    y = pandas.Series(y) 

    if method=='spearman':
        rho = spearmanr
    if method=='pearson':
        rho = pearsonr
    #TODO
    # add other methods    

    return X.apply(lambda x: abs(rho(x, y).correlation)).mean()