from pathlib import Path
from dotenv import find_dotenv
from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication


def init_dotenv():
    env_path = find_dotenv()
    if env_path == "":
        Path(".env").touch()
        env_path = find_dotenv()
    return 4


def load_configuration(configuration_file):
    import yaml

    with open(configuration_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    return cfg


def get_or_create_workspace(workspace_name, subscription_id, resource_group, workspace_region):
    auth = InteractiveLoginAuthentication()

    ws = Workspace.create(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        location=workspace_region,
        create_resource_group=True,
        auth=auth,
        exist_ok=True,
    )
    ws.write_config()

    return ws
