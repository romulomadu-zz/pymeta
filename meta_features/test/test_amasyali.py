import sys
import os
from pathlib import Path
from sklearn.datasets import load_boston

src_dir = Path(__file__).resolve().parents[1] 
sys.path.append(os.path.abspath(src_dir))
from amasyali import *

boston = load_boston()
X = boston.data
y = boston.target


def test_proportion_of_binary_features():
    pass
    

def test_proportion_of_outlier_target():
    pass




