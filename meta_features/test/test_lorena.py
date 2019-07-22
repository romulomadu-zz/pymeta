import sys
import os
from pathlib import Path
from sklearn.datasets import load_boston

src_dir = Path(__file__).resolve().parents[1] 
sys.path.append(os.path.abspath(src_dir))
from lorena import *

boston = load_boston()
X = boston.data
y = boston.target


def test_max_feature_correlation_target():
    assert 0.8529141394922163 == max_feature_correlation_target(X, y)


def test_c1():
    assert 0.8529141394922163 == c1(X, y)


def test_c2():
    assert 0.49301623665125105 == c2(X, y)


def test_c3():
    pass
    # assert 0.49301623665125105 == c3(X, y)


def test_c4():
    pass
    # assert 0.49301623665125105 == c4(X, y)


def test_c5():
    assert 0.4186358986029782 == c5(X, y)
