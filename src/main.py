import sys
import os

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import numpy as np
from src.data_preprocessing import (
    preprocessing_volume_data,
    data_quality_vol_analysis,
    apply_data_quality_timeseries,
    preprocessing_production,
)
from src.models import train_test_stats_models, train_test_lightgbm
from src.utils import store_pickle
from typing import Dict, Tuple, List


def generate_vol_prod_ratio_gold(
    df_vol_gold: pd.DataFrame, df_prod: pd.DataFrame
) -> pd.DataFrame:
    """Generate volume production ratio

    Args:
        df_vol_gold (pd.DataFrame): volume gold dataframe
        df_prod (pd.DataFrame): production dataframe

    Returns:
        pd.DataFrame: volume production ratio dataframe
    """
    # Add production information to the timeseries
    df_ratio_gold = pd.merge(
        df_vol_gold, df_prod, on=["Timestamp", "Plant"], how="left"
    )

    df_ratio_gold["Actual_Vol_[Kg]"] = df_ratio_gold["Actual_Vol_[Tons]"] * 1000
    df_ratio_gold["Expected_Vol_[Kg]"] = df_ratio_gold["Expected_Vol_[Tons]"] * 1000

    # Create New Feature:
    # Volume / production Ratio
    df_ratio_gold["Actual_Vol_[Kg]"] = df_ratio_gold["Actual_Vol_[Tons]"] * 1000
    df_ratio_gold["Expected_Vol_[Kg]"] = df_ratio_gold["Expected_Vol_[Tons]"] * 1000
    df_ratio_gold["Vol/Prod_ratio_ton"] = np.round(
        df_ratio_gold["Actual_Vol_[Tons]"] / df_ratio_gold["Production"], 5
    )
    df_ratio_gold["Vol/Prod_ratio_kg"] = np.round(
        df_ratio_gold["Actual_Vol_[Kg]"] / df_ratio_gold["Production"], 5
    )
    df_ratio_gold.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Add ts len
    df_ratio_gold["ts_len"] = df_ratio_gold.groupby("ts_key")["Timestamp"].transform(
        lambda x: len(x)
    )
    df_ratio_gold.sort_values(by=["ts_key", "ts_len", "Timestamp"], inplace=True)

    return df_ratio_gold


def main_lightgbm(df_timeseries_gold: pd.DataFrame, shards: list) -> tuple:
    """Train LightGBM model and generate forecasts

    Args:
        df_timeseries_gold (pd.DataFrame): _description_
        shards (list): _description_

    Returns:
        tuple: _description_
    """
    # Train LightGBM
    lgbm_model, df_lgbm_forecast = train_test_lightgbm(
        ts=df_timeseries_gold, shards=shards
    )

    # Store Model
    path = "../models/lgbm_forecast.pkl"
    store_pickle(obj=lgbm_model, path=path)

    # Store Forecasts
    df_lgbm_forecast.to_parquet("../data/forecasts/lgbm_forecast.parquet")

    return lgbm_model, df_lgbm_forecast


def main_stats_models(df_timeseries_gold: pd.DataFrame, shards: list) -> tuple:
    """Generate forecasts for the statistical models

    Args:
        df_timeseries_gold (pd.DataFrame): gold dataframe
        shards (list): train, test, validation shards

    Returns:
        tuple: model and forecasts
    """

    # this make it so that the outputs of the predict methods have the id as a column
    # instead of as the index
    os.environ["NIXTLA_ID_AS_COL"] = "1"

    stats_models, df_stats_forecast = train_test_stats_models(
        ts=df_timeseries_gold, shards=shards
    )

    # Store Model
    path = "../models/stats_forecast.pkl"
    store_pickle(obj=stats_models, path=path)

    # Store Forecasts
    df_stats_forecast.to_parquet("../data/forecasts/stats_forecast.parquet")

    return stats_models, df_stats_forecast


def ensemble_model(
    config: Dict, df_result_lgbm: pd.DataFrame, df_stats_forecast: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List]:
    """Create ensemble model from the forecasts of the different models

    Args:
        df_result_lgbm (pd.DataFrame): forecast of lightgbm model
        df_stats_forecast (pd.DataFrame): forecasts of statistical model

    Returns:
        pd.DataFrame: ensemble forecast
    """
    ml_model_names = ["LIGHTGBM"]
    stats_model_names = [
        "AutoARIMA",
        "AutoETS",
        "CES",
        "SeasonalNaive",
        "WindowAverage",
    ]
    model_names = ml_model_names + stats_model_names

    # df_stats_forecast_melted = pd.melt(
    #         df_stats_forecast,
    #         id_vars=['ts_key', 'Timestamp', 'test_frame'],
    #         value_vars=stats_model_names,
    #         var_name='forecasting_model',
    #         value_name='y_hat'
    #     )

    # Join all models in one single dataframe
    df_forecats = pd.merge(
        df_result_lgbm.reset_index(),
        df_stats_forecast,
        on=["ts_key", "Timestamp", "test_frame"],
        how="inner",
    )

    df_ensemble_forecast = pd.melt(
        df_forecats,
        id_vars=["ts_key", "Timestamp", "y_true", "test_frame"],  # Columns to keep
        value_vars=model_names,  # Columns to pivot
        var_name="Model",  # Name of the new column for model names
        value_name="y_pred",  # Name of the new column for predicted values
    )

    df_ensemble_forecast = (
        df_ensemble_forecast.filter(
            ["ts_key", "Timestamp", "test_frame", "Model", "y_pred"]
        )
        .groupby(["ts_key", "Timestamp", "test_frame"], group_keys=False)["y_pred"]
        .mean()
        .to_frame()
        .reset_index()
    )

    model_names.append("Ensemble")
    df_ensemble_forecast["Model"] = "Ensemble"

    df_ensemble_forecast = df_ensemble_forecast.drop(columns=["Model"]).rename(
        columns={"y_pred": "Ensemble"}
    )

    # Verify we joint data correctly, and there are not duplicated forecasts
    assert (
        df_forecats.shape[1] != df_ensemble_forecast.shape[1]
    ), "There are duplicated forecasts"

    df_forecats = pd.merge(
        df_forecats,
        df_ensemble_forecast,
        on=["ts_key", "Timestamp", "test_frame"],
        how="inner",
    )

    df_ratio_gold = pd.read_parquet(config["preprocessing"]["ratio_gold_path"])
    df_true = (
        df_ratio_gold[["ts_key", "Timestamp", "Actual_Vol_[Kg]", "Production"]]
        .rename(columns={"Actual_Vol_[Kg]": "y_target"})
        .copy()
    )

    evaluation_df = pd.merge(
        df_forecats.reset_index(), df_true, on=["ts_key", "Timestamp"], how="left"
    )

    for model_name in model_names:
        evaluation_df[f"{model_name}_target"] = (
            evaluation_df[model_name] * evaluation_df["Production"]
        )

    return df_true, df_forecats, evaluation_df, model_names


def preparation_production_data(
    config: Dict, df_prod_bronze: pd.DataFrame
) -> pd.DataFrame:
    """
    This function prepares the production data by applying the following steps:
    - Data Quality Analysis
    - Data Preprocessing
    - Feature Engineering
    - Data Quality Analysis

    Args:
        config: The configuration parameters
        df_prod: The production data

    Returns:
        df_prod: The prepared production data
    """
    df_prod = preprocessing_production(df_prod=df_prod_bronze)

    print(
        "The historical production data contains data since",
        df_prod["Timestamp"].min(),
        " until ",
        df_prod["Timestamp"].max(),
    )
    print("in Total it contains ", df_prod.shape[0], " rows.")
    print("in Total it contains ", df_prod.shape[1], " columns.")
    print("Total available Plants are: ", df_prod["Plant"].nunique())
    print(
        "Max Production Volume was: ",
        df_prod["Production"].max(),
        " units. In",
        df_prod[df_prod["Production"] == df_prod["Production"].max()][
            "Timestamp"
        ].values[0],
    )
    print(
        "Min Production Volume was: ",
        df_prod["Production"].min(),
        " units. In",
        df_prod[df_prod["Production"] == df_prod["Production"].min()][
            "Timestamp"
        ].values[0],
    )

    # Store to silver
    df_prod.to_parquet(config["preprocessing"]["prod_silver_path"])

    return df_prod


def data_preparation_and_data_quality(
    config: Dict, df_vol_bronze: pd.DataFrame
) -> pd.DataFrame:
    """Data preparation and data quality checks for the volume data

    Args:
        df_vol_bronze (pd.DataFrame): input bronze volume data

    Returns:
        pd.DataFrame: gold volume data
    """

    df_vol = preprocessing_volume_data(df_vol=df_vol_bronze)

    print(
        "The historical transport volume data contains data since",
        df_vol["Timestamp"].min(),
        " until ",
        df_vol["Timestamp"].max(),
    )
    print(
        "in Total it contains data for",
        df_vol["ts_key"].nunique(),
        " inbound logistics Provider-Plant connections",
    )
    print("in Total it contains data for", df_vol["Plant"].nunique(), " plants")
    print("in Total it contains data for", df_vol["Provider"].nunique(), " Providers")
    print("in Total it contains ", df_vol.shape[0], " rows.")
    print("in Total it contains ", df_vol.shape[1], " columns.")

    # Store to silver
    df_vol.to_parquet(config["preprocessing"]["vol_silver_path"])

    df_vol_summary = data_quality_vol_analysis(df_vol=df_vol)

    df_vol_gold = apply_data_quality_timeseries(
        df_vol=df_vol,
        df_vol_summary=df_vol_summary,
        ts_len_threshold=config["data_quality"]["ts_len_threshold"],
    )

    # Store clean volume data in gold layer
    df_vol_gold.to_parquet(config["preprocessing"]["vol_gold_path"])

    return df_vol_gold
