"""
ado-ml-batch-train - sklearn_utils.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def create_train_test_split(ai_impact_score_ds, test_size=0.2, seed=223):
    """
    Create an X and Y Training and Testing set using SKLearn

    :param ai_impact_score_ds: Returns a :class:`azureml.data.TabularDataset` object, a pointer to Azure ML Dataset
    that contains AI Impact Score Sample data from the Azure Machine Learning Workspace
    :rtype: azureml.data.TabularDataset
    :param test_size: the proportion of the dataset to include in the test split
    :type test_size: float, int or None, optional (default=0.2)
    :param seed: seed used by the random number generator
    :type seed: int (default=223)
    :return: Returns four pandas DataFrames, X_Train, X_Test, Y_Train, Y_Test
    :rtype: pd.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series
    """
    ai_impact_score_pd = ai_impact_score_ds.to_pandas_dataframe()

    columns_to_remove = ["IsBlocker", "Pri", "LogScore", "DCReview"]
    for col in columns_to_remove:
        ai_impact_score_pd.pop(col)

    y_df = ai_impact_score_pd.pop("Score")
    x_df = ai_impact_score_pd

    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=test_size, random_state=seed)
    return x_train, x_test, y_train, y_test


def create_plots(local_run, x_train, x_test, y_train, y_test):
    """
    Create Histogram Plots for Residuals

    :param local_run: Completed Local AutoML Run
    :type local_run: AutoMLRun
    :param x_train: X Training DataSet
    :type x_train: pandas.DataFrame
    :param x_test: X Test DataSet
    :type x_test: pandas.DataFrame
    :param y_train: Y Training DataSet
    :type y_train: pandas.DataFrame
    :param y_test: Y Test DataSet
    :type y_test: pandas.DataFrame
    """
    _, fitted_model = local_run.get_output()

    y_pred_train = fitted_model.predict(x_train)
    y_residual_train = y_train - y_pred_train

    y_pred_test = fitted_model.predict(x_test)
    y_residual_test = y_test - y_pred_test

    # %%

    # Set up a multi-plot chart.
    figure, (axis_1, axis_2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1], 'wspace': 0, 'hspace': 0})
    figure.suptitle('Regression Residual Values', fontsize=18)
    figure.set_figheight(6)
    figure.set_figwidth(16)

    # Plot residual values of training set.
    axis_1.axis([0, 360, -200, 200])
    axis_1.plot(y_residual_train, 'bo', alpha=0.5)
    axis_1.plot([-10, 360], [0, 0], 'r-', lw=3)
    axis_1.text(16, 170, 'RMSE = {0:.2f}'.format(np.sqrt(mean_squared_error(y_train, y_pred_train))), fontsize=12)
    axis_1.text(16, 140, 'R2 score = {0:.2f}'.format(r2_score(y_train, y_pred_train)), fontsize=12)
    axis_1.set_xlabel('Training samples', fontsize=12)
    axis_1.set_ylabel('Residual Values', fontsize=12)

    # Plot a histogram.
    axis_1.hist(y_residual_train, orientation='horizontal', color='b', bins=10, histtype='step')
    axis_1.hist(y_residual_train, orientation='horizontal', color='b', alpha=0.2, bins=10)

    # Plot residual values of test set.
    axis_2.axis([0, 90, -200, 200])
    axis_2.plot(y_residual_test, 'bo', alpha=0.5)
    axis_2.plot([-10, 360], [0, 0], 'r-', lw=3)
    axis_2.text(5, 170, 'RMSE = {0:.2f}'.format(np.sqrt(mean_squared_error(y_test, y_pred_test))), fontsize=12)
    axis_2.text(5, 140, 'R2 score = {0:.2f}'.format(r2_score(y_test, y_pred_test)), fontsize=12)
    axis_2.set_xlabel('Test samples', fontsize=12)
    axis_2.set_yticklabels([])

    # Plot a histogram.
    axis_2.hist(y_residual_test, orientation='horizontal', color='b', bins=10, histtype='step')
    axis_2.hist(y_residual_test, orientation='horizontal', color='b', alpha=0.2, bins=10)

    plt.show()
