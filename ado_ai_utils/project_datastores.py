"""
ado-ml-batch-train - project_datastores.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
from azureml.core import Datastore, Workspace
from azureml.data.azure_sql_database_datastore import AzureSqlDatabaseDatastore

from ado_ai_utils.constants import ADO_SQL_DATASTORE


def get_ado_sql_datastore(workspace: Workspace) -> AzureSqlDatabaseDatastore:
    """
    Get Azure DevOps Azure SQL Database

    :param workspace: Azure Machine Learning Workspace Pointer
    :return: Azure Machine Learning SQL Datastore
    """
    return Datastore.get(workspace=workspace, datastore_name=ADO_SQL_DATASTORE)
