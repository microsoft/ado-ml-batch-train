"""
ado-ml-batch-train - project_datasets.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
from azureml.core import Dataset, Workspace
from azureml.data import TabularDataset


def get_ai_feedback_sample(workspace: Workspace) -> TabularDataset:
    """
    Get Pointer to AI Feedback Sample

    :param workspace: Azure Machine Learning Workspace
    :return: Returns a :class:`azureml.data.TabularDataset` object, a pointer to Dataset stored in Azure Machine
    Learning Workspace
    """
    dataset_name = 'ai_impact_scores'

    return Dataset.get_by_name(workspace=workspace, name=dataset_name)
