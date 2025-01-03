import yaml
import pickle
import numpy as np
from typing import Dict, Any, List, Tuple
import pandas as pd
from sklearn.metrics import mean_absolute_error

COLORS = [
    "#8c564b",
    "#0E4D64",
    "#17becf",
    "#fdae61",
    "#abdda4",
    "#fee08b",
    "#7f7f7f",
    "#e377c2",
    "#8c564b",
    "#9467bd",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#1f77b4",
    "#17b7bf",
]


data_dict_prod = pd.DataFrame(
    {
        "Name": ["Timestamp", "Plant", "Production"],
        "Description": [
            "Monthly date of the format YYYY-MM-DD",
            "Assembly Plant ID",
            "Production Volume in Number of Units",
        ],
        "Role": ["ID", "ID", "predictor"],
        "Type": ["ordinal", "nominal", "numeric"],
        "Format": ["datetime", "category", "int"],
    }
)

data_dict_vol = pd.DataFrame(
    {
        "Name": [
            "Timestamp",
            "Provider",
            "Plant",
            "Actual_Vol_[Kg]",
            "Expected_Vol_[Kg]",
            "Year",
            "Month",
            "ts_key",
            "Actual_Vol_[Tons]",
            "Expected_Vol_[Tons]",
        ],
        "Description": [
            "Monthly date of the format YYYY-MM-DD",
            "Logistics Provider ID",
            "Assembly Plant ID",
            "Actual transported volume from Provider to Plant in kg",
            "Expected transported volume from Provider to Plant in kg",
            "Year in which transport took place",
            "Month in which transport took place",
            "Timeseries key",
            "Actual transported volume from Provider to Plant in tons",
            "Expected transported volume from Provider to Plant in tons",
        ],
        "Role": [
            "ID",
            "ID",
            "ID",
            "response",
            "predictor",
            "predictor",
            "predictor",
            "ID",
            "predictor",
            "predictor",
        ],
        "Type": [
            "ordinal",
            "nominal",
            "nominal",
            "numeric",
            "numeric",
            "numeric",
            "numeric",
            "numeric",
            "numeric",
            "numeric",
        ],
        "Format": [
            "datetime",
            "category",
            "category",
            "float",
            "float",
            "int",
            "int",
            "category",
            "float",
            "float",
        ],
    }
)

data_dict_covid = pd.DataFrame(
    {
        "Name": ["Timestamp", "Country"],
        "Description": [
            "Monthly date of the format YYYY-MM-DD",
            "Monthly COVID-19 Rate Per 100k (14-Day Average) in the given country",
        ],
        "Role": ["ID", "predictor"],
        "Type": ["ordinal", "numeric"],
        "Format": ["datetime", "float"],
    }
)


def calculate_accuracy_metrics(
    evaluation_df: pd.DataFrame, model_names: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate accuracy metrics

    Args:
        evaluation_df (pd.DataFrame): model prediction with all models
        model_names (List[str]): model names

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: SMAPE and MAE
    """
    dfs_smape = []
    dfs_maes = []
    for model_name in model_names:
        df_smape = (
            evaluation_df.groupby(["test_frame", "ts_key"], group_keys=False)
            .apply(lambda x: smape(x["y_target"], x[f"{model_name}_target"]))
            .to_frame()
            .rename(columns={0: "smape"})
            .reset_index()
        )
        df_smape["model_name"] = model_name
        dfs_smape.append(df_smape)
        del df_smape

        dfs_mae = (
            evaluation_df.groupby(["test_frame", "ts_key"], group_keys=False)
            .apply(
                lambda x: mean_absolute_error(x["y_target"], x[f"{model_name}_target"])
            )
            .to_frame()
            .rename(columns={0: "mae"})
            .reset_index()
        )
        dfs_mae["model_name"] = model_name
        dfs_maes.append(dfs_mae)
        del dfs_mae

    df_accuracy_smape = pd.concat(dfs_smape)
    df_accuracy_mae = pd.concat(dfs_maes)

    return df_accuracy_smape, df_accuracy_mae


def read_config(yaml_file_path: str) -> Dict[str, Any]:
    """Read config from yaml file

    Args:
        yaml_file_path (str): location of yaml file

    Returns:
        Dict[str, Any]: dict object
    """
    try:
        with open(yaml_file_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"The file {yaml_file_path} does not exist.")
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return {}


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Symmetric Mean Absolute Error

    Args:
        y_true (np.ndarray): true values
        y_pred (np.ndarray): forecast values

    Returns:
        float: SMAPE
    """
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))
    )


def store_pickle(obj: Any, path: str) -> None:
    """Store object as pickle

    Args:
        obj (Any): Input Object
        path (str): Path to store the pickle file
    """
    with open(path, "wb") as file:
        pickle.dump(obj, file)
    print(f"Object has been stored at {path}.")


def read_pickle(path: str) -> Any:
    """Load pickle object

    Args:
        path (str): location path

    Returns:
        Any: object
    """
    with open(path, "rb") as file:
        obj = pickle.load(file)
    print(f"Loaded object from {path}.")
    return obj
