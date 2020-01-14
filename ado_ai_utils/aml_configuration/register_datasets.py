"""
ado-ml-batch-train - register_datasets.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

from azureml.core import Dataset, Workspace
from azureml.data import TabularDataset
from azureml.data.azure_sql_database_datastore import AzureSqlDatabaseDatastore
from azureml.data.datapath import DataPath


def register_improvements(sql_datastore: AzureSqlDatabaseDatastore) -> TabularDataset:
    """
    Register Azure SQL Improvements Table with Azure Machine Learning

    :param sql_datastore: Azure Machine Learning SQL Datastore
    """
    query_string = "SELECT *  FROM Improvements"

    query = DataPath(sql_datastore, query_string)
    improvements_sql_ds = Dataset.Tabular.from_sql_query(query)

    improvements_sql_ds.register(workspace=sql_datastore.workspace,
                                 name="ai_ag_ado_improvements",
                                 description="Improvements from Azure DevOps",
                                 create_new_version=True)

    return improvements_sql_ds


def register_feedback(sql_datastore: AzureSqlDatabaseDatastore) -> TabularDataset:
    """
    Register Feedback Table with Impact Score calculation by dciborow

    :param sql_datastore: Azure DevOps SQL
    :return: Pointer to Feedback Table in Azure DevOps SQL
    """
    query_string = "SELECT * FROM FeedbackItems"

    query = DataPath(sql_datastore, query_string)
    feedback_sql_ds = Dataset.Tabular.from_sql_query(query)

    feedback_sql_ds.register(workspace=sql_datastore.workspace,
                             name="ai_ag_ado_feedack_table",
                             description="Feedback from Azure DevOps",
                             create_new_version=True)
    return feedback_sql_ds


def register_feedback_with_impact(sql_datastore: AzureSqlDatabaseDatastore) -> TabularDataset:
    """
    Register Feedback Table with Impact Score calculation by dciborow

    :param sql_datastore: Azure DevOps SQL
    :return: Pointer to Feedback Table in Azure DevOps SQL
    """
    query_string = 'SELECT MitigationScore, Priority, CONVERT(bit, IsBlocker) as IsBlocker , (POWER(1.5, ' \
                   'MitigationScore) * POWER(2,Priority) * POWER(6.585, CONVERT(bit, IsBlocker))) as dc_impact_score ' \
                   'FROM FeedbackItems'

    query = DataPath(sql_datastore, query_string)
    feedback_sql_ds = Dataset.Tabular.from_sql_query(query)

    feedback_sql_ds.register(workspace=sql_datastore.workspace,
                             name="ai_ag_ado_feedack",
                             description="Feedback from Azure DevOps with Impact Label",
                             create_new_version=True)
    return feedback_sql_ds


def register_feedback_train_test(workspace: Workspace, dataset: TabularDataset, percentage=0.8, seed=223) \
        -> (TabularDataset, TabularDataset):
    """
    Split the dataset into train and test datasets

    :param workspace: Azure Machine Learning Workspace Pointer
    :param dataset: Feedback Dataset Pointer
    :param percentage: Percentage in Train Dataset
    :param seed: Random Number Generator Seed using for splitting. Set to None for random
    :return: Train and Test Dataset Pointers
    """
    train_data, test_data = dataset.random_split(percentage=percentage, seed=seed)

    # Register the train dataset with your workspace
    train_data.register(workspace=workspace,
                        name='ai_ag_ado_feedack_train_dataset',
                        description='Feedback from Azure DevOps training data',
                        create_new_version=True)

    # Register the test dataset with your workspace
    test_data.register(workspace=workspace,
                       name='ai_ag_ado_feedack_test_dataset',
                       description='Feedback from Azure DevOps test data',
                       create_new_version=True)
    return train_data, test_data
