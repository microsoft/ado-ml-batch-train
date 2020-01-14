"""
ado-ml-batch-train - automl_utils

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import pandas as pd
from azureml.core import Model
from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun

from ado_ai_utils.aml_configuration.utils import get_workspace_from_config
from ado_ai_utils.automl_utils import get_local_automl_config, get_or_create_local_run, get_run_data, \
    register_best_model
from ado_ai_utils.project_datasets import get_ai_feedback_sample
from ado_ai_utils.sklearn_utils import create_train_test_split

WORKSPACE = get_workspace_from_config()
AI_IMPACT_SCORE_DS = get_ai_feedback_sample(WORKSPACE)
X_TRAIN, x_test, Y_TRAIN, y_test = create_train_test_split(AI_IMPACT_SCORE_DS)


def test_get_local_automl_config():
    """ Test Get Local AutoML Config"""

    automl_config = get_local_automl_config(X_TRAIN, Y_TRAIN)

    assert type(automl_config) is AutoMLConfig


def test_get_or_create_local_run():
    automl_config = get_local_automl_config(X_TRAIN, Y_TRAIN)
    local_run = get_or_create_local_run(WORKSPACE, automl_config)

    assert type(local_run) is AutoMLRun


def test_get_run_data():
    automl_config = get_local_automl_config(X_TRAIN, Y_TRAIN)
    local_run = get_or_create_local_run(WORKSPACE, automl_config)

    run_data = get_run_data(local_run)

    assert type(run_data) is pd.DataFrame


def test_register_best_model():
    automl_config = get_local_automl_config(X_TRAIN, Y_TRAIN)
    local_run = get_or_create_local_run(WORKSPACE, automl_config)

    model = register_best_model(local_run)

    assert type(model) is Model
