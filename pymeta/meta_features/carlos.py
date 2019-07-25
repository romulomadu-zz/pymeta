import numpy
import category_encoders as ce
import pandas

from utils import *
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats.stats import pearsonr, spearmanr


def n_of_examples(X: numpy.array, y: numpy.array=None) -> int:
    '''Number of examples in a dataset.'''

    return X.shape[0]


def n(X: numpy.array, y: numpy.array=None) -> int:
    '''Alias for number of examples in a dataset.'''

    return X.shape[0]


def n_of_features(X: numpy.array, y: numpy.array=None) -> int:
    '''Number of attributes in a dataset.'''

    return X.shape[1]


def m(X: numpy.array, y: numpy.array=None) -> int:
    '''Alias for number of attributes in a dataset.'''

    return X.shape[1]


def proportion_of_categorical(X: numpy.array, y: numpy.array=None) -> int:
    '''Proportion of categorical features in a dataset.'''

    # check if not dataframe
    if not isinstance(X, pandas.DataFrame):
        X = pandas.DataFrame(X)

    # get number of categorical columns
    n_categorical = len(numpy.where(X.dtypes != numpy.float)[0])

    return n_categorical / n_of_features(X)


def example_features_ratio(X: numpy.array, y: numpy.array=None) -> float:
    '''Ratio of the number of examples to the number of features.'''

    return n_of_examples(X) / n_of_features(X)


def proportion_of_attributes_outliers(X: numpy.array, y: numpy.array=None) -> float:
    '''Propotion of attributes with outliers.'''

    # check if not dataframe
    if not isinstance(X, pandas.DataFrame):
        X = pandas.DataFrame(X)

    # get number of columns with outliers
    iqr, q1, q3 = IQR(X)
    is_outlier = (X < (q1 - 1.5 * iqr)) | (X > (q3 + 1.5 * iqr))
    m_with_outlier = is_outlier.any(axis=0).sum()

    # get number of numeric columns
    m_numeric = len(numpy.where(X.dtypes == numpy.float)[0])

    return m_with_outlier / m_numeric


def coeficient_of_variation_target(y: numpy.array) -> float:
    '''Coefficient of variation of the target (ratio of the standard deviation to the mean).'''

    return numpy.std(y) / numpy.mean(y)

#TODO
# def sparsity_of_target(y: numpy.array) -> int:
#     '''Sparsity of the target. (Coeficient of variation of the target discretized into three values).'''
#     pass


def outliers_on_target(y: numpy.array) -> int:
    '''Presence of outliers in the target.'''

    iqr, q1, q3 = IQR(y)
    is_outlier = (y < (q1 - 1.5 * iqr)[0]) | (y > (q3 + 1.5 * iqr)[0])

    return int(is_outlier.sum() > 0)


def stationarity_of_target(y: numpy.array) -> int:
    '''Stationarity of the target (check if the standard deviation is larger than mean).'''

    return int(y.std() > y.mean())


def r2_without_categorical(X: numpy.array, y: numpy.array) -> float:
    '''R2 coefficient of linear regression (without categorical attributes).'''

    # check if not dataframe
    if not isinstance(X, pandas.DataFrame):
        X =  pandas.DataFrame(X)
    
    # get numerical features indexes and filt.
    numerical_features = numpy.where(X.dtypes == numpy.float)[0]
    X_numerical = X.iloc[:, numerical_features]

    # train model and make prediction.
    model  = LinearRegression().fit(X_numerical, y)
    y_pred = model.predict(X_numerical)

    return r2_score(y, y_pred)


def r2_with_binarized_categorical(X: numpy.array, y: numpy.array=None) -> float:
    '''R2 coefficient of linear regression (without categorical attributes).'''

    # check if not dataframe
    if not isinstance(X, pandas.DataFrame):
        X = pandas.DataFrame(X)
    
    # binarize categorical data
    categorical_features = numpy.where(X.dtypes != numpy.float)[0]
    if categorical_features.size:
        binary = ce.BinaryEncoder(cols=categorical_features)
        X = binary.fit_transform(X, y)

    # train model and make prediction.
    model  = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    
    return r2_score(y, y_pred)


def mean_feature_correlation(X: numpy.array, y: numpy.array=None, method: str='spearman') -> float:
    '''Average correlation between the numeric features.'''

    # check if not dataframe
    if not isinstance(X, pandas.DataFrame):
        X = pandas.DataFrame(X)
    
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

    Return
    ------
    float:
        Average feature correlation to the output.
    """
    # check if not dataframe
    if not isinstance(X, pandas.DataFrame):
        X = pandas.DataFrame(X)  
    y = pandas.Series(y) 

    if method=='spearman':
        rho = spearmanr
    if method=='pearson':
        rho = pearsonr    

    return X.apply(lambda x: abs(rho(x, y).correlation)).mean()