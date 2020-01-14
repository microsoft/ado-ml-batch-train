"""
ado-ml-batch-train - aml_configuration/register_datastores.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
from azureml.core import Datastore


def register_blob_datastore(workspace, blob_datastore_name, container_name, account_name, account_key, datastore_rg):
    """
    Register a Blob Storage Account with the Azure Machine Learning Workspace

    :param workspace: Azure Machine Learning Workspace
    :type workspace: azureml.core.Workspace
    :param blob_datastore_name: Name for blob datastore
    :type blob_datastore_name: str
    :param container_name: Name for blob container
    :type container_name: str
    :param account_name: Name for blob account
    :type account_name: str
    :param account_key: Blob Account Key using for auth
    :type account_key: str
    :param datastore_rg: Resource Group containing Azure Storage Account
    :type datastore_rg: str
    :return: Pointer to Azure Machine Learning Blob Datastore
    :rtype: azureml.data.azure_storage_datastore.AzureBlobDatastore
    """
    blob_datastore = Datastore.register_azure_blob_container(workspace=workspace,
                                                             datastore_name=blob_datastore_name,
                                                             container_name=container_name,
                                                             account_name=account_name,
                                                             account_key=account_key,
                                                             resource_group=datastore_rg,
                                                             overwrite=True)
    return blob_datastore


def register_sql_datastore(workspace, sql_datastore_name, sql_server_name, sql_database_name, sql_username,
                           sql_password):
    """
    Register a Azure SQL DB with the Azure Machine Learning Workspace

    :param workspace: Azure Machine Learning Workspace
    :type workspace: azureml.core.Workspace
    :param sql_datastore_name: Name used to id the SQL Datastore
    :type sql_datastore_name: str
    :param sql_server_name: Azure SQL Server Name
    :type sql_server_name: str
    :param sql_database_name: Azure SQL Database Name
    :type sql_database_name: str
    :param sql_username: Azure SQL Database Username
    :type sql_username: str
    :param sql_password: Azure SQL Database Password
    :type sql_password: str
    :return: Pointer to Azure Machine Learning SQL Datastore
    :rtype: azureml.data.azure_sql_database_datastore.AzureSqlDatabaseDatastore
    """
    sql_datastore = Datastore.register_azure_sql_database(workspace=workspace,
                                                          datastore_name=sql_datastore_name,
                                                          server_name=sql_server_name,
                                                          database_name=sql_database_name,
                                                          username=sql_username,
                                                          password=sql_password)
    return sql_datastore
