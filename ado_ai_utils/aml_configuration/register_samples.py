"""
ado-ml-batch-train - aml_configuration/register_samples.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
from azureml.core.dataset import Dataset


def register_improvements_sample(blob_datastore, workspace):
    """
    Register the sample file containing 500 Improvements from a CSV in Azure Blob Storage

    :param blob_datastore: Pointer to Azure Machine Learning Blob Datastore
    :type blob_datastore: azureml.data.azure_storage_datastore.AzureBlobDatastore
    :param workspace: Azure Machine Learning Workspace
    :type workspace: azureml.core.Workspace
    :return: Returns a :class:`azureml.data.TabularDataset` object, a pointer to Dataset stored in Azure Machine
    Learning Workspace
    :rtype: azureml.data.TabularDataset
    """
    datastore_paths = [(blob_datastore, 'improvements.csv')]
    ai_impact_scores = Dataset.Tabular.from_delimited_files(path=datastore_paths)
    ai_impact_scores.register(workspace=workspace,
                              name="improvements_sample",
                              description="subset of improvements items from SQL")

    return ai_impact_scores


def register_feedback_sample(blob_datastore, workspace):
    """
    Register the sample file containing 500 Feedback Items from a CSV in Azure Blob Storage

    :param blob_datastore: Pointer to Azure Machine Learning Blob Datastore
    :type blob_datastore: azureml.data.azure_storage_datastore.AzureBlobDatastore
    :param workspace: Azure Machine Learning Workspace
    :type workspace: azureml.core.Workspace
    :return: Returns a :class:`azureml.data.TabularDataset` object, a pointer to Dataset stored in Azure Machine
    Learning Workspace
    :rtype: azureml.data.TabularDataset
    """
    datastore_paths = [(blob_datastore, 'feedback_items.csv')]
    ai_impact_scores = Dataset.Tabular.from_delimited_files(path=datastore_paths)
    ai_impact_scores.register(workspace=workspace,
                              name="feedback_items_sample",
                              description="subset of feedback items from SQL")

    return ai_impact_scores


def register_ai_feedback_sample(blob_datastore, workspace):
    """
    Register the sample file containing AI Feedback Items from a CSV in Azure Blob Storage

    :param blob_datastore: Pointer to Azure Machine Learning Blob Datastore
    :type blob_datastore: azureml.data.azure_storage_datastore.AzureBlobDatastore
    :param workspace: Azure Machine Learning Workspace
    :type workspace: azureml.core.Workspace
    :return: Returns a :class:`azureml.data.TabularDataset` object, a pointer to Dataset stored in Azure Machine
    Learning Workspace
    :rtype: azureml.data.TabularDataset
    """
    datastore_paths = [(blob_datastore, 'ai_impact_scores.csv')]

    ai_impact_scores = Dataset.Tabular.from_delimited_files(path=datastore_paths)

    ai_impact_scores.register(workspace=workspace,
                              name="ai_impact_scores",
                              description="ai subset of feedback items")

    return ai_impact_scores
