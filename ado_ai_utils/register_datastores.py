from azureml.core import Datastore


def register_blob_datastore(ws, blob_datastore_name, container_name, account_name, account_key, datastore_rg):
    blob_datastore = Datastore.register_azure_blob_container(workspace=ws,
                                                             datastore_name=blob_datastore_name,
                                                             container_name=container_name,
                                                             account_name=account_name,
                                                             account_key=account_key,
                                                             resource_group=datastore_rg,
                                                             overwrite=True)
    return blob_datastore


def register_sql_datastore(ws, sql_datastore_name, sql_server_name, sql_database_name, sql_username, sql_password):
    sql_datastore = Datastore.register_azure_sql_database(workspace=ws,
                                                          datastore_name=sql_datastore_name,
                                                          server_name=sql_server_name,
                                                          database_name=sql_database_name,
                                                          username=sql_username,
                                                          password=sql_password)
    return sql_datastore
