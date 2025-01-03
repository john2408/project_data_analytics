import optuna
import pandas as pd
from datetime import datetime
import lightgbm as lgb
from typing import List, Tuple
from sklearn.metrics import mean_absolute_error
from statsforecast import StatsForecast
import statsforecast
from dateutil.relativedelta import relativedelta
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    AutoCES,
    SeasonalNaive,
    WindowAverage,
)

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, KAN, TFT

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import read_pickle

import logging
import os

logging.getLogger().setLevel(logging.CRITICAL)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU logs
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # If TensorFlow is use


def format_ts_data_to_nn_forecast(ts: pd.DataFrame) -> pd.DataFrame:
    """Format the time series data to be used in the Neural Networks Forecasting

    Args:
        ts (pd.DataFrame): time series data

    Returns:
        pd.DataFrame: formatted time series data
    """

    try:
        # Create custom increasing index for NHiTS algorithm
        ts["time_idx"] = (
            ts.sort_values(by=["ts_key", "Timestamp"]).groupby("ts_key").cumcount()
        )

        _min = ts["ts_len"].min()
        _max = ts["ts_len"].max()
        print(f"Min/Max length of a time series: {_min} , {_max}")

        ts.drop(columns=["ts_len"], inplace=True)

        # Rename columns to expected format in neuralforecast library
        # Additionaly, keep only the target column
        ts = ts.rename(
            columns={"ts_key": "unique_id", "Vol/Prod_ratio_kg": "y", "Timestamp": "ds"}
        ).filter(["ds", "unique_id", "y"])

        return ts

    except Exception as e:
        print(e)
        return None


def train_test_deep_learning(
    ts: pd.DataFrame, shards: List[List[datetime]]
) -> Tuple[NeuralForecast, pd.DataFrame]:
    """Train Test Neural Networks Forecasting

    Args:
        ts (pd.DataFrame): _description_
        shards (List[List[datetime]]): _description_

    Returns:
        Tuple[NeuralForecast, pd.DataFrame, pd.DataFrame]: _description_
    """
    np.random.seed(42)

    # Neural Networks
    FREQUENCY = "MS"  # Month Start Frequency

    max_encoder_length = 6
    max_prediction_length = 4  # 4 Months forecast horizon
    val_size = 4  # 4 Months forecast horizon in validation
    TS_LEN_THRESHOLD = max_encoder_length + 2 * max_prediction_length

    print("Min Length per Timeseries: ", TS_LEN_THRESHOLD)

    # Data Quality Check for Neural Networks Forecasting
    assert (
        ts["ts_len"].min() > TS_LEN_THRESHOLD
    ), f"Not all timeseries len is greater than {TS_LEN_THRESHOLD}"
    assert (
        set(ts.columns[ts.isna().any()]) == set()
    ), "There are columns containing NA Values"

    ts = format_ts_data_to_nn_forecast(ts=ts)

    df_forecats = pd.DataFrame()

    for shard in shards:
        test_frame = (
            shard[3].strftime("%Y-%m-%d") + " - " + shard[4].strftime("%Y-%m-%d")
        )
        print(
            "Train-Testing for",
            shard[3].strftime("%Y-%m-%d"),
            shard[4].strftime("%Y-%m-%d"),
        )

        # Train and Validation Set are input together as one dataset
        # Prediction set is our 4-month window
        df_train_val, df_test = (
            ts[(ts["ds"] <= shard[2])],
            ts[(ts["ds"] >= shard[3]) & (ts["ds"] <= shard[4])],
        )

        # Configure Models
        models = [
            NBEATS(
                input_size=max_encoder_length,
                h=max_prediction_length,
                max_steps=max_encoder_length,
            ),
            NHITS(
                input_size=max_encoder_length,
                h=max_prediction_length,
                max_steps=max_encoder_length,
            ),
            TFT(
                input_size=max_encoder_length,
                h=max_prediction_length,
                max_steps=max_encoder_length,
            ),
        ]

        nf = NeuralForecast(models=models, freq=FREQUENCY)
        nf.fit(df=df_train_val, val_size=val_size, verbose=False)

        Y_hat_df = nf.predict().reset_index()

        df_metric_shard = pd.merge(
            df_test, Y_hat_df, how="left", on=["unique_id", "ds"]
        )
        df_metric_shard["test_frame"] = test_frame

        # Adjust negative forecast to be 0
        df_metric_shard["NBEATS"] = df_metric_shard["NBEATS"].apply(
            lambda x: 0 if x <= 0 else x
        )
        df_metric_shard["NHITS"] = df_metric_shard["NHITS"].apply(
            lambda x: 0 if x <= 0 else x
        )
        df_metric_shard["TFT"] = df_metric_shard["TFT"].apply(
            lambda x: 0 if x <= 0 else x
        )

        df_forecats = pd.concat([df_forecats, df_metric_shard])

    df_melted_forecast = pd.melt(
        df_forecats,
        id_vars=["unique_id", "ds", "test_frame"],
        value_vars=["NBEATS", "NHITS", "TFT"],
        var_name="forecasting_model",
        value_name="y_hat",
    )

    df_forecats = df_forecats.rename(
        columns={"unique_id": "ts_key", "ds": "Timestamp"}
    ).drop(columns=["index", "y"])

    return nf, df_forecats


def feature_importance_analysis(model_path: str, top: int = 5) -> None:
    """Generate a feature importance plot for a LightGBM model.

    Args:
        model_path (str): _description_
        top (int, optional): _description_. Defaults to 5.
    """
    lgb_model = read_pickle(path=model_path)
    # Get feature importances
    importance = lgb_model.feature_importance()
    feature_names = lgb_model.feature_name()
    feature_importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importance}
    )

    feature_importance_df = feature_importance_df.sort_values(
        by="Importance", ascending=False
    )

    # Plot feature importance with a different color palette
    plt.figure(figsize=(8, 4))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=feature_importance_df.head(top),
        palette="viridis",
    )
    plt.title("Feature Importance")
    plt.show()


def train_test_stats_models(
    ts: pd.DataFrame, shards: List[datetime]
) -> Tuple[statsforecast.core.StatsForecast, pd.DataFrame]:
    """_summary_

    Args:
        ts (pd.DataFrame): input timeseries data with features
        shards (List[datetime]): train/val/test chards

    Returns:
        Tuple[statsforecast.core.StatsForecast, pd.DataFrame]: stats model & forecasts dataframe
    """
    Y_df = (
        ts[["ts_key", "Timestamp", "Vol/Prod_ratio_kg"]]
        .rename(
            columns={"ts_key": "unique_id", "Timestamp": "ds", "Vol/Prod_ratio_kg": "y"}
        )
        .copy()
    )

    models = [
        AutoARIMA(),
        AutoETS(),
        AutoCES(),
        SeasonalNaive(season_length=12),
        WindowAverage(window_size=6),
    ]

    # Set random seed for reproducibility
    np.random.seed(42)

    # Instantiate StatsForecast class to train models
    # use all CPU cores in the process
    sf = StatsForecast(
        models=models,
        freq="MS",
        fallback_model=SeasonalNaive(season_length=12),
        n_jobs=-1,
    )

    # Select only start of testing frame
    # since nixtla stats package takes care of val split
    test_fames = [shard[3] for shard in shards]

    dfs = []
    forecast_horizon = 4
    for start_date in test_fames:
        end_date = start_date + relativedelta(months=3)
        test_frame_str = (
            start_date.strftime("%Y-%m-%d") + " - " + end_date.strftime("%Y-%m-%d")
        )
        print("Forecasting for test frame", test_frame_str)
        _y_df = Y_df[Y_df["ds"] < start_date].copy()

        # Model training and prediction
        _df = sf.forecast(df=_y_df, h=forecast_horizon)
        _df["test_frame"] = test_frame_str
        dfs.append(_df)

        # Delete tmp variables
        del _y_df
        del _df

    df_stats_forecast = pd.concat(dfs)

    df_stats_forecast.rename(
        columns={"unique_id": "ts_key", "ds": "Timestamp"}, inplace=True
    )

    return sf, df_stats_forecast


def train_test_lightgbm(
    ts: pd.DataFrame, shards: List[datetime]
) -> Tuple[lgb.basic.Booster, pd.DataFrame]:
    """Train/validate/test LightGBM Model

    Args:
        ts (pd.DataFrame): input timeseries data with features
        shards (List[datetime]): train/val/test chards

    Returns:
        Tuple[lgb.basic.Booster, pd.DataFrame]: model and forecast values
    """
    # ts = df_timeseries_gold.copy()

    # Set random seed for reproducibility
    random_seed = 42

    DROP_COLUMNS = []
    INDEX_COL = "Timestamp"
    CAT_FEATURES = ["Plant", "Provider"]
    TARGET = "Vol/Prod_ratio_kg"
    DROP_NAN = False

    ts.drop(columns=DROP_COLUMNS, inplace=True)

    # Convert Providers and Plant Column to binary format
    ts = pd.get_dummies(ts, columns=CAT_FEATURES)

    n_opt_trials = 12
    params = {
        "objective": "regression",
        "metric": "mean_absolute_error",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_seed": random_seed,
    }

    dfs = []
    for shard in shards:
        test_frame = (
            shard[3].strftime("%Y-%m-%d") + " - " + shard[4].strftime("%Y-%m-%d")
        )
        print(
            "Train-Testing for",
            shard[3].strftime("%Y-%m-%d"),
            shard[4].strftime("%Y-%m-%d"),
        )

        train, val, test = (
            ts[ts["Timestamp"] <= shard[0]].copy().drop(columns=[INDEX_COL]),
            ts[(ts["Timestamp"] >= shard[1]) & (ts["Timestamp"] <= shard[2])]
            .copy()
            .drop(columns=[INDEX_COL]),
            ts[(ts["Timestamp"] >= shard[3]) & (ts["Timestamp"] <= shard[4])].copy(),
        )

        train_x = train.drop(columns=[TARGET, "ts_key"])
        train_y = train[TARGET]

        val_x = val.drop(columns=[TARGET, "ts_key"])
        val_y = val.set_index("ts_key")[TARGET]

        test_timestamps = test[INDEX_COL].values
        test = test.drop(columns=[INDEX_COL])
        test_x = test.drop(columns=[TARGET, "ts_key"])
        test_y = test.set_index("ts_key")[TARGET]

        dtrain = lgb.Dataset(train_x, label=train_y)
        dval = lgb.Dataset(val_x, label=val_y)

        # callable for optimization
        def objective(trial):
            # Parameters
            params_tuning = {
                "objective": "regression",
                "metric": "mean_absolute_error",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-3, 0.1, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 2, 2**10),
                "subsample": trial.suggest_float("subsample", 0.05, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
                "feature_pre_filter": False,
            }

            gbm = lgb.train(params_tuning, dtrain)
            y_pred = gbm.predict(val_x)
            mae = mean_absolute_error(val_y, y_pred)
            return mae

        # start hyperparameter tuning
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_opt_trials)

        print("Best hyperparameters:", study.best_params)
        print("Best MAE:", study.best_value)

        # Model Training with best params and performance test
        train = ts[ts["Timestamp"] <= shard[2]].copy().drop(columns=["Timestamp"])
        train_x = train.drop(columns=[TARGET, "ts_key"])
        train_y = train[TARGET]
        dtrain = lgb.Dataset(train_x, label=train_y)

        model = lgb.train({**params, **study.best_params}, dtrain)

        # prediction
        y_pred = model.predict(test_x)

        _df_forecast = pd.DataFrame(
            {
                "Timestamp": test_timestamps,
                "y_true": test_y,
                "y_pred_lgbm": y_pred,
                "test_frame": test_frame,
            }
        )

        dfs.append(_df_forecast)

    df_result_lgbm = pd.concat(dfs)

    df_result_lgbm.rename(columns={"y_pred_lgbm": "LIGHTGBM"}, inplace=True)

    return model, df_result_lgbm
