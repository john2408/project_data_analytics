import yaml
import pickle
from typing import Dict, Any


def read_config(yaml_file_path: str) -> Dict:
    """Read config from yaml file

    Args:
        yaml_file_path (str): location of yaml file

    Returns:
        Dict: dict object
    """

    try:
        with open(yaml_file_path, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"The file {yaml_file_path} does not exist.")
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")

    return config