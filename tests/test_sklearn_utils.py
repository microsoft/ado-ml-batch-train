"""
ado-ml-batch-train - ${FILE_NAME}

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import pandas as pd

from ado_ai_utils.aml_configuration.utils import get_workspace_from_config
from ado_ai_utils.automl_utils import get_or_create_local_run, get_local_automl_config
from ado_ai_utils.project_datasets import get_ai_feedback_sample
from ado_ai_utils.sklearn_utils import create_plots
from ado_ai_utils.sklearn_utils import create_train_test_split

WORKSPACE = get_workspace_from_config()
AI_IMPACT_SCORE_DS = get_ai_feedback_sample(WORKSPACE)


def test_create_train_test_split():
    """ Test Create Train Test Split Function"""
    x_train, x_test, y_train, y_test = create_train_test_split(AI_IMPACT_SCORE_DS)

    assert type(x_train) is pd.DataFrame
    assert type(x_test) is pd.DataFrame
    assert type(y_train) is pd.Series
    assert type(y_test) is pd.Series


def test_create_plots():
    """
    Test Creation of Plots

    There are no real asserts in this test, if no errors are thrown, it will pass.
    """
    x_train, x_test, y_train, y_test = create_train_test_split(AI_IMPACT_SCORE_DS)
    automl_config = get_local_automl_config(x_train, y_train)
    local_run = get_or_create_local_run(WORKSPACE, automl_config)

    create_plots(local_run, x_train, x_test, y_train, y_test)
    assert True
