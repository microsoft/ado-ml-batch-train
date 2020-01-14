"""
ado-ml-batch-train - aml_configuration/utils.py

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
from pathlib import Path

import yaml
from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from dotenv import find_dotenv


def init_dotenv(path=".env"):
    """
    Initialize a new DotEnv

    See PyPi for more details: https://pypi.org/project/python-dotenv/

    :param path: File Path for .env file to be loaded, or created
    :type path: str
    :return: Return a :class:`str` object, containing the path to the env file
    :rtype: str
    """
    env_path = find_dotenv()
    if env_path == "":
        Path(path).touch()
        env_path = find_dotenv()
    return env_path


def load_configuration(configuration_file):
    """
    Load the Workspace Configuration File.

    The workspace configuration file is used to protect against putting passwords within the code, or source control.
    To create the configuration file, make a copy of sample_workspace.conf named "workspace_conf.yml" and fill in
    each field.
    This file is set to in the .gitignore to prevent accidental comments.

    :param configuration_file: File Path to configuration yml
    :type configuration_file: str
    :return: Returns the parameters needed to configure the AML Workspace and Experiments
    :rtype: Union[Dict[Hashable, Any], list, None], str, str, str, str, Workspace, str, str
    """
    with open(configuration_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    return cfg


def get_or_create_workspace(workspace_name, subscription_id, resource_group, workspace_region):
    """
    Create a new Azure Machine Learning workspace. If the workspace already exists, the existing workspace will be
    returned. Also create a CONFIG file to quickly reload the workspace.

    This uses the :class:`azureml.core.authentication.InteractiveLoginAuthentication` or will default to use the
    :class:`azureml.core.authentication.AzureCliAuthentication` for logging into Azure.

    Run az login from the CLI in the project directory to avoid authentication when running the program.

    :param workspace_name: Name of Azure Machine Learning Workspace to get or create within the Azure Subscription
    :type workspace_name: str
    :param subscription_id: Azure Subscription ID
    :type subscription_id: str
    :param resource_group: Azure Resource Group to get or create the workspace within. If the resource group does not
    exist it will be created.
    :type resource_group: str
    :param workspace_region: The Azure region to deploy the workspace.
    :type workspace_region: str
    :return: Returns a :class:`azureml.core.Workspace` object, a pointer to Azure Machine Learning Workspace
    Learning Workspace
    :rtype: azureml.core.Workspace
    """
    auth = InteractiveLoginAuthentication()

    workspace = Workspace.create(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        location=workspace_region,
        create_resource_group=True,
        auth=auth,
        exist_ok=True,
    )
    workspace.write_config()

    return workspace


def get_workspace_from_config():
    """
    Retrieve an AML Workspace from a previously saved configuration

    :return: Azure Machine Learning Workspace
    :rtype: azureml.core.Workspace
    """
    return Workspace.from_config()
