import numpy
import category_encoders as ce

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


def n_of_examples(X: numpy.array, y: numpy.array=None) -> int:
    '''Number of examples in a dataset.'''

    return X.shape[0]

def n_of_features(X: numpy.array, y: numpy.array=None) -> int:
    '''Number of attributes in a dataset.'''

    return X.shape[1]

def proportion_of_categorical(X: numpy.array, y: numpy.array=None) -> int:
    '''Proportion of categorical features in a dataset.'''
    n_categorical = len(numpy.where(X.dtypes != numpy.float)[0])

    return n_categorical / n_of_features(X)

def example_features_ratio(X: numpy.array, y: numpy.array=None) -> float:
    '''Ratio of the number of examples to the number of features.'''

    return n_of_examples(X) / n_of_features(X)

def proportion_of_attributes_outliers(X: numpy.array, y: numpy.array=None) -> float:
    '''Propotion of attributes with outliers.'''

    pass

def coeficient_of_variation_target(y: numpy.array) -> float:
    '''Coefficient of variation of the target (ratio of the standard deviation to the mean).'''

    return numpy.std(y) / numpy.mean(y)

def stationarity_of_target(y: numpy.array) -> int:
    '''Stationarity of the target (check if the standard deviation is larger than mean).'''

    return int(numpy.std(y) > numpy.mean(y))

def r2_without_categorical(X: numpy.array, y: numpy.array) -> float:
    '''R2 coefficient of linear regression (without categorical attributes).'''
    # get numerical features indexes and filt.
    numerical_features = numpy.where(X.dtypes != numpy.float)[0]
    X_numerical = X[:, numerical_features]

    # train model and make prediction.
    model  = LinearRegression().fit(X_numerical, y)
    y_pred = model.predict(X_numerical)

    return r2_score(y, y_pred)

def r2_with_binarized_categorical(X: numpy.array, y: numpy.array=None) -> float:
    '''R2 coefficient of linear regression (without categorical attributes).'''
    # binarize categorical data
    binary = ce.BinaryEncoder()
    X_binary = binary.fit_transform(X, y)

    # train model and make prediction.
    model  = LinearRegression().fit(X_binary, y)
    y_pred = model.predict(X_binary)
    
    return r2_score(y, y_pred)

def average_correlation_between_features(X: numpy.array, y: numpy.array=None) -> float:
    '''Average correlation between the numeric features.'''
    # get numerical features indexes and filt.
    numerical_features = numpy.where(X.dtypes != numpy.float)[0]
    X_numerical = X[:, numerical_features]
    m = X_numerical.shape[1]

    # get correlation matrix.
    correlation_matrix = numpy.corrcoef(X_numerical)
    ids = numpy.triu_indices(m, k=1)

    return numpy.abs(correlation_matrix[ids]).mean()
