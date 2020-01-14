"""
ado-ml-batch-train - automl_utils.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import logging

import pandas as pd
from azureml.core.experiment import Experiment
from azureml.train.automl import AutoMLConfig


def get_local_automl_config(x_train, y_train):
    """
    Create a new :class:`azureml.train.automl.AutoMLConfig` for local runs.

    :param x_train: X Training DataFrame
    :type x_train: pandas.DataFrame
    :param y_train: Y Training DataFrame
    :type y_train: pandas.DataFrame
    :return: AutoMLCong for local run
    :rtype: azureml.train.automl.AutoMLConfig
    """
    automl_config = AutoMLConfig(task='regression',
                                 iteration_timeout_minutes=10,
                                 iterations=10,
                                 primary_metric='spearman_correlation',
                                 n_cross_validations=5,
                                 debug_log='automl.log',
                                 verbosity=logging.INFO,
                                 X=x_train,
                                 y=y_train,
                                 preprocess=True)
    return automl_config


def get_or_create_local_run(workspace, automl_config):
    """
    Get Or Create an AutoML Run

    If the local run has already ben completed this will retrieve the previous run from the current Azure Machine
    Learning Workspace. Otherwise this will create a new  run within the Azure Machine Learning Workspace.

    :param workspace:
    :type workspace:
    :param automl_config:
    :type automl_config:
    :return: run
    :rtype: AutoMLRun
    """
    experiment = Experiment(workspace, "ai-impact-score-experiment")

    runs = experiment.get_runs()

    if not runs:
        # noinspection PyTypeChecker
        return experiment.submit(automl_config, show_output=True)

    for run in runs:
        return run


def get_run_data(local_run):
    """
    Retrieve Run Data from a local run

    :param local_run: Local AutoML Run
    :type local_run: run
    :return: DataFrame of Local AutoML Run Metrics
    :rtype: pandas.DataFrame
    """
    children = list(local_run.get_children())
    metrics_list = {}
    for run in children:
        properties = run.get_properties()
        metrics = {k: v for k, v in run.get_metrics().items() if isinstance(v, float)}
        metrics_list[int(properties['iteration'])] = metrics

    return pd.DataFrame(metrics_list).sort_index(1)


def register_best_model(local_run):
    """
    Register Local AutoML Model with the Azure Machine Learning Workspace
    :param local_run: Completed Local AutoML Run
    :type local_run: run
    :return: The registered model.
    :rtype: azureml.core.model.Model
    """
    best_run, _ = local_run.get_output()

    model = best_run.register_model(model_name='best_impact_score_model', model_path='./outputs/model.pkl')
    print(
        "Registered model:\n --> Name: {}\n --> Version: {}\n --> URL: {}".format(model.name, model.version, model.url))
    return model
