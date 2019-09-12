import pandas
import numpy
import sys
from pathlib import Path
from os.path import join
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

project_dir = Path(__file__).resolve().parents[2]
sys.path.append(join(project_dir, 'pymeta'))
import meta_features
sys.path.append(join(project_dir, 'pymeta', 'meta_learning'))
from .utils import _pprint

class BaseMeta(object):
    """
    Base class for meta features evaluators objects.
    """
    
    def get_meta(self):
        pass
    
    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.qualities(result='dict'),
                                               offset=len(class_name),),)

class MetaFeaturesRegression(BaseMeta):
    '''Build meta features for regression tasks.'''
    def __init__(self, dataset_name: str='None', random_state: int=None, metric: str='mae',
                    ignore_categorical: bool=False, normalize: bool=True, 
                        correlation_method: str='spearman', n_jobs: int=None,
                            coef_threshold: float=.5, categorical_mask: list=None,
                                distance_metric: str='euclidean'):
        self.random_state = random_state
        self.dataset_name = dataset_name
        #TODO 
        # pass self.metric as parameter in metafeatures calls.
        self.metric = metric
        self.n_jobs = n_jobs
        self.ignore_categorical = ignore_categorical
        self.normalize = normalize
        self.correlation_method = correlation_method
        self.coef_threshold = coef_threshold
        self.categorical_mask = categorical_mask
        self.distance_metric = distance_metric
        self.params_ = None
    
    def _calculate_meta_features(self, X: numpy.array, y: numpy.array) -> dict:
        """
        Calculate meta features for regression task.

        Parameters
        ----------
        X : numpy.array
            2D array with features columns
        y : numpy.array
            Array of true values

        Return
        ------
        dict :
            Dictionary with format {metafeature: value, ...} with all metafeatures.

        """

        # check if not dataframe
        X = (pandas.DataFrame(X) if not isinstance(X, pandas.DataFrame) else X)
            
        # get categorical mask if not passed.
        if not isinstance(self.categorical_mask, list):
            self.categorical_mask = (X.dtypes != numpy.float)

        # create array with only numeric features.
        X_numeric = X.loc[:, numpy.array(self.categorical_mask)==False]

        # normalize numeric data if flag is True.
        X_numeric = (MinMaxScaler().fit_transform(X_numeric) if self.normalize else X_numeric)
        
        # pre calculations.
        model = LinearRegression().fit(X_numeric, y)
        dist_matrix = squareform(pdist(X_numeric, metric=self.distance_metric))
                
        # feed dict and calculate metafeatures.
        params_dict = {
            'dataset' : self.dataset_name,
            'n_of_examples': meta_features.n_of_examples(X),
            'n_of_features': meta_features.n_of_features(X),
            'proportion_of_categorical': meta_features.proportion_of_categorical(X, categorical_mask=self.categorical_mask),
            'example_features_ratio': meta_features.example_features_ratio(X),
            'proportion_of_attributes_outliers': meta_features.proportion_of_attributes_outliers(X_numeric),
            'coeficient_of_variation_target': meta_features.coeficient_of_variation_target(y),
            'outliers_on_target': meta_features.outliers_on_target(y),
            'stationarity_of_target': meta_features.stationarity_of_target(y),
            'r2_without_categorical': meta_features.r2_without_categorical(X_numeric, y, model=model),
            'r2_with_binarized_categorical': meta_features.r2_with_binarized_categorical(X, y, categorical_mask=self.categorical_mask),
            'mean_feature_correlation': meta_features.mean_feature_correlation(X_numeric, method=self.correlation_method),
            'mean_feature_correlation_target': meta_features.mean_feature_correlation_target(X_numeric, y, method=self.correlation_method),
            'proportion_of_outliers_target': meta_features.proportion_of_outliers_target(y),
            'proportion_of_binary_features': meta_features.proportion_of_binary_features(X),
            'min_kurtosis_numerical_features': meta_features.min_kurtosis_numerical_features(X_numeric),
            'max_kurtosis_numerical_features': meta_features.max_kurtosis_numerical_features(X_numeric),
            'mean_kurtosis_numerical_features': meta_features.mean_kurtosis_numerical_features(X_numeric),
            'min_skewness_numerical_features': meta_features.min_skewness_numerical_features(X_numeric),
            'max_skewness_numerical_features': meta_features.max_skewness_numerical_features(X_numeric),
            'mean_skewness_numerical_features': meta_features.mean_skewness_numerical_features(X_numeric),
            'min_mean_numerical_features': meta_features.min_mean_numerical_features(X_numeric),
            'max_mean_numerical_features': meta_features.max_mean_numerical_features(X_numeric),
            'mean_mean_numerical_features': meta_features.mean_mean_numerical_features(X_numeric),
            'min_std_numerical_features': meta_features.min_std_numerical_features(X_numeric),
            'max_std_numerical_features': meta_features.max_std_numerical_features(X_numeric),
            'mean_std_numerical_features': meta_features.mean_std_numerical_features(X_numeric),
            'proportion_of_features_with_na': meta_features.proportion_of_features_with_na(X),
            'proportion_of_correlated_features_target': meta_features.proportion_of_correlated_features_target(X_numeric, y, method=self.correlation_method, coef=self.coef_threshold),
            'proportion_of_correlated_features': meta_features.proportion_of_correlated_features(X_numeric, method=self.correlation_method, coef=self.coef_threshold),
            'max_feature_correlation_target': meta_features.max_feature_correlation_target(X_numeric, y, method=self.correlation_method),
            'individual_feature_efficiency': meta_features.individual_feature_efficiency(X_numeric, y),
            'collective_feature_efficiency': meta_features.collective_feature_efficiency(X_numeric, y),
            'output_distribution': meta_features.output_distribution(X_numeric, y, dist_matrix),
            'input_distribution': meta_features.input_distribution(X_numeric, y),
            'error_of_nn_regressor': meta_features.error_of_nn_regressor(X_numeric, y, dist_matrix, metric=self.metric),
            'nonlinearity_of_nn_regressor': meta_features.nonlinearity_of_nn_regressor(X_numeric, y, self.random_state, metric=self.metric),
            'mean_absolute_residuos': meta_features.mean_absolute_residuos(X_numeric, y, model, ignore_categorical=self.ignore_categorical),
            'mean_squared_residuos': meta_features.mean_squared_residuos(X_numeric, y, model, ignore_categorical=self.ignore_categorical),
            'nonlinearity_of_linear_regressor': meta_features.nonlinearity_of_linear_regressor(X_numeric, y, model, self.random_state, metric=self.metric)
        }

        return params_dict

    def fit(self, X: numpy.array, y: numpy.array):
        """
        Call meta features calculation and store it.

        Parameters
        ----------
        X : numpy.array
            2D array with features columns
        y : numpy.array
            Array of true values

        Return
        ------
        object :
            MetaFeatures object with params calculated

        """        
        self.params_ = self._calculate_meta_features(X,y)
        return self

    def qualities(self, result: str='DataFrame') -> pandas.DataFrame:
        """
        Get fitted metafeatures.

        Parameters
        ----------
        result : str, optional(default='dataframe')
            Return format. 'dict' for dict and 'DataFrame' for pandas.DataFrame

        Return
        ------
        object :
            Table with metafeatures calculated

        """          

        return (pandas.DataFrame.from_dict(self.params_, orient='index') if result=='DataFrame' else self.params_)


if __name__ == "__main__":
    boston = load_boston()
    X = pandas.DataFrame(boston.data, columns=boston.feature_names)
    y = boston.target
    mfr = MetaFeaturesRegression(dataset_name='Boston')
    mfr.fit(X, y)
    print(mfr.qualities())


    




    


    