"""
ado-ml-batch-train - automl_utils.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import logging

import pandas as pd
from azureml.automl.core.constants import FeaturizationConfigMode
from azureml.core import Workspace, Model, Dataset, Run
from azureml.core.experiment import Experiment
from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun


def get_local_automl_config(x_train: pd.DataFrame, y_train: pd.DataFrame) -> AutoMLConfig:
    """
    Create a new :class:`azureml.train.automl.AutoMLConfig` for local runs.

    :param x_train: X Training DataFrame
    :param y_train: Y Training DataFrame
    :return: AutoMLConfig for local run
    """
    return AutoMLConfig(task='regression', iteration_timeout_minutes=10, iterations=10,
                        primary_metric='spearman_correlation', n_cross_validations=5, debug_log='automl.log',
                        verbosity=logging.INFO, X=x_train, y=y_train, preprocess=True)


def get_remote_automl_config(train_data: Dataset, label: str) -> AutoMLConfig:
    """
    Create a new :class:`azureml.train.automl.AutoMLConfig` for remote runs.

    :param train_data: Training Dataset
    :param label: Dependant Variable Column
    :return: AutoMLConfig for remote runs
    """
    return AutoMLConfig(task='regression', debug_log='automl_errors.log', training_data=train_data,
                        label_column_name=label, verbosity=logging.INFO, enable_early_stopping=True,
                        max_concurrent_iterations=4, max_cores_per_iteration=-1, n_cross_validations=5,
                        primary_metric='spearman_correlation', preprocess=True,
                        featurization=FeaturizationConfigMode.Auto, enable_tf=True)


def get_or_create_local_run(workspace: Workspace, automl_config: AutoMLConfig) -> AutoMLRun:
    """
    Get Or Create an AutoML Run

    If the local run has already ben completed this will retrieve the previous run from the current Azure Machine
    Learning Workspace. Otherwise this will create a new  run within the Azure Machine Learning Workspace.

    :param workspace:
    :param automl_config:
    :return: run
    """
    experiment = Experiment(workspace, "ai-impact-score-experiment")

    runs = experiment.get_runs()

    def get_first_run_id():
        """ Retrieve the most recent run if it has already completed """
        if not runs:
            return experiment.submit(automl_config).id

        for run in runs:
            return run.id

    run_id = get_first_run_id()
    return AutoMLRun(experiment=experiment, run_id=run_id)


def get_run_data(local_run: AutoMLRun) -> pd.DataFrame:
    """
    Retrieve Run Data from a local run

    :param local_run: Local AutoML Run
    :return: DataFrame of Local AutoML Run Metrics
    """
    children = list(local_run.get_children())
    metrics_list = {}
    for run in children:
        properties = run.get_properties()
        metrics = {k: v for k, v in run.get_metrics().items() if isinstance(v, float)}
        metrics_list[int(properties['iteration'])] = metrics

    return pd.DataFrame(metrics_list).sort_index(1)


def register_best_local_model(local_run: AutoMLRun) -> Model:
    """
    Register Local AutoML Model with the Azure Machine Learning Workspace

    :param local_run: Completed Local AutoML Run
    :return: The registered model.
    """
    best_run, _ = local_run.get_output()

    model = best_run.register_model(model_name='best_impact_score_model', model_path='./outputs/model.pkl')
    print(
        "Registered model:\n --> Name: {}\n --> Version: {}\n --> URL: {}".format(model.name, model.version, model.url))
    return model


def register_best_remote_model(remote_run: AutoMLRun) -> Model:
    """
    Register Local AutoML Model with the Azure Machine Learning Workspace

    :param local_run: Completed Local AutoML Run
    :return: The registered model.
    """
    best_run, _ = remote_run.get_output()

    model = best_run.register_model(model_name='best_sql_dc_impact_score_model', model_path='./outputs/model.pkl')
    print(
        "Registered model:\n --> Name: {}\n --> Version: {}\n --> URL: {}".format(model.name, model.version, model.url))
    return model


def get_or_create_remote_run(workspace: Workspace, automl_config: AutoMLConfig) -> Run:
    """
    Create a new AutoML Experiment
    :param workspace: Azure Machine Learning Workspace
    :param automl_config: Remote AutoML Config
    :return: Remote Run
    """
    experiment = Experiment(workspace, "ai-impact-score-experiment-dc-sql")

    runs = experiment.get_runs()

    def get_first_run_id():
        """ Retrieve the most recent run if it has already completed """
        if not runs:
            return experiment.submit(automl_config).id

        for run in runs:
            return run.id

    run_id = get_first_run_id()
    return AutoMLRun(experiment=experiment, run_id=run_id)
