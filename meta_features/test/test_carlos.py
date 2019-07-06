import sys
from pathlib import Path
from sklearn.datasets import load_boston

src_dir = Path(__file__).resolve().parents[2] 
print(src_dir)
sys.path.append(src_dir)
from .meta_features.carlos import *

boston = load_boston()
X = boston.data
y = boston.target

def test_n_of_examples():
    assert X.shape[0] == n_of_examples(X) 

def test_n_of_features():
    assert X.shape[1] == n_of_features(X) 

def test_proportion_of_categorical():
    pass

def test_example_features_ratio():
    pass

def test_proportion_of_attributes_outliers():
    pass

def test_coeficient_of_variation_target():
    pass

def test_stationarity_of_target():
    pass

def test_r2_without_categorical():
    pass

def test_r2_with_binarized_categorical():
    pass

def test_average_correlation_between_features():
    pass
