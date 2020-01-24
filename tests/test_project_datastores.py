"""
ado-ml-batch-train - ${FILE_NAME}

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
from azureml.data.azure_sql_database_datastore import AzureSqlDatabaseDatastore

from ado_ai_utils.aml_configuration.utils import get_workspace_from_config
from ado_ai_utils.project_datastores import get_ado_sql_datastore

WORKSPACE = get_workspace_from_config()


def test_get_ado_sql_datastore():
    """ Test Get Azure DevOps SQL Datastore """
    adl_sql_datastore = get_ado_sql_datastore(WORKSPACE)

    assert isinstance(adl_sql_datastore, AzureSqlDatabaseDatastore)
