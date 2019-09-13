***************************************
Meta features for regression with OpenML
***************************************
 
The objective of this tutorial is to show how to download datasets from OpenML API and calculate his meta features using :code:`MetaRegressionFeatures` class from PyMeta.

Installing Python's OpenML API
##############################

To use OpenML API you need to install it using pip:

Then you need to sign up on `OpenML site <https://www.openml.org/>`_ to get your API key for authentication.

Downloading dataset from OpenML API
###################################

First we need to import :code:`openml` package and set the API key.

.. code-block:: python
   :linenos:

    import openml

    # you must config you api key
    openml.config.apikey = "your api key goes here"

After setting API key we can download a dataset, keep in mind that dataset is referenced are by a ID integer. This ID is the endpoint of the dataset page at OpenML.

For the Boston house-price dataset, for example, the endpoint is 531 as you can see on page url:

`<https://www.openml.org/d/531>`_

Let's download this dataset and see summary information about it:

.. code-block:: python
   :linenos:

    # This is done based on the dataset ID.
    dataset = openml.datasets.get_dataset(531)

    # Print a summary
    print("This is dataset '%s', the target feature is '%s'" %
        (dataset.name, dataset.default_target_attribute))
    print("URL: %s" % dataset.url)
    print(dataset.description)

For this task, we will consider :code:`MEDV` as the target columns passing it to the function get data and retrieving array of features, target, besides categorical columns mask and features names.

.. code-block:: python
   :linenos:

    X, y, categorical_columns, columns_names = dataset.get_data('MEDV')
    # view of first five samples of features dataframe
    X.head()
    # view of first five output target
    y.head()
    # view of first five saamples of the categorical columns
    X.loc[:, categorical_columns].head()

Get meta features with MetaFeaturesRegression
############################################

First, import the :code:`MetaFeaturesRegression` class from PyMeta.


.. code-block:: python
   :linenos:

    import sys
    from os.path import join, abspath
    from pathlib import Path

    # get project dir
    project_dir = Path(abspath('')).resolve().parent
    # add it to path
    sys.path.append(join(project_dir))

    # get MetaFeaturesRegression
    from pymeta.meta_learning import MetaFeaturesRegression

Then, instanciate the object.

.. code-block:: python
   :linenos:

    mfr = MetaFeaturesRegression(
            dataset_name='Boston',
            random_state=42,
            n_jobs=3,
            categorical_mask=categorical_columns        
    )

Fit the meta features for Boston dataset.

.. code-block:: python
   :linenos:

    mfr.fit(X, y)
    # get metafeatures as pandas.DataFrame
    mfr.qualities()


