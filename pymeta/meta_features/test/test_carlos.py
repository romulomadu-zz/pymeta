import sys
import os
from pathlib import Path
from sklearn.datasets import load_boston

src_dir = Path(__file__).resolve().parents[1] 
sys.path.append(os.path.abspath(src_dir))
from carlos import *

boston = load_boston()
X = boston.data
y = boston.target


def test_n_of_examples():
    assert X.shape[0] == n_of_examples(X) 


def test_n_of_features():
    assert X.shape[1] == n_of_features(X) 


def test_n():
    assert X.shape[0] == n(X) 


def test_m():
    assert X.shape[1] == m(X) 


def test_proportion_of_categorical():
    assert 0.0 == proportion_of_categorical(X)


def test_example_features_ratio():
    assert (n(X) / m(X)) == example_features_ratio(X)


def test_proportion_of_attributes_outliers():
    assert (8 / 13) == proportion_of_attributes_outliers(X)


def test_coeficient_of_variation_target():
    assert 0.40776152837415536 == coeficient_of_variation_target(y)


# def test_sparsity_of_target():
#     pass


def test_outliers_on_target():
    assert True == outliers_on_target(y)


def test_stationarity_of_target():
    assert False == stationarity_of_target(y)


def test_r2_without_categorical():
    assert r2_without_categorical(X, y) == r2_with_binarized_categorical(X, y)


def test_mean_feature_correlation():
    assert 0.4186358986029782 == mean_feature_correlation(X)


def test_mean_feature_correlation_target():
    assert 0.49301623665125105 == mean_feature_correlation_target(X, y)

