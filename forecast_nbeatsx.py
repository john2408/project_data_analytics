import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic


def sample_nbeatsx_forecast():
    Y_train_df = AirPassengersPanel[
        AirPassengersPanel.ds < AirPassengersPanel["ds"].values[-12]
    ]  # 132 train
    Y_test_df = AirPassengersPanel[
        AirPassengersPanel.ds >= AirPassengersPanel["ds"].values[-12]
    ].reset_index(
        drop=True
    )  # 12 test

    model = NBEATSx(
        h=12,
        input_size=24,
        loss=MQLoss(level=[80, 90]),
        scaler_type="robust",
        dropout_prob_theta=0.5,
        stat_exog_list=["airline1"],
        futr_exog_list=["trend", "y_[lag12]"],
        max_steps=200,
        val_check_steps=10,
        early_stop_patience_steps=2,
    )

    nf = NeuralForecast(models=[model], freq="M")
    nf.fit(df=Y_train_df, static_df=AirPassengersStatic, val_size=12)
    Y_hat_df = nf.predict(futr_df=Y_test_df)

    # Plot quantile predictions
    Y_hat_df = Y_hat_df.reset_index(drop=False).drop(columns=["unique_id", "ds"])

    plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
    plot_df = pd.concat([Y_train_df, plot_df])

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Plot for Airline1
    plot_df1 = plot_df[plot_df.unique_id == "Airline1"].drop("unique_id", axis=1)
    axes[0].plot(plot_df1["ds"], plot_df1["y"], c="black", label="True")
    axes[0].plot(plot_df1["ds"], plot_df1["NBEATSx-median"], c="blue", label="median")
    axes[0].fill_between(
        x=plot_df1["ds"][-12:],
        y1=plot_df1["NBEATSx-lo-90"][-12:].values,
        y2=plot_df1["NBEATSx-hi-90"][-12:].values,
        alpha=0.4,
        label="level 90",
    )
    axes[0].legend()
    axes[0].grid()
    axes[0].set_title("Airline1")

    # Plot for Airline2
    plot_df2 = plot_df[plot_df.unique_id == "Airline2"].drop("unique_id", axis=1)
    axes[1].plot(plot_df2["ds"], plot_df2["y"], c="black", label="True")
    axes[1].plot(plot_df2["ds"], plot_df2["NBEATSx-median"], c="blue", label="median")
    axes[1].fill_between(
        x=plot_df2["ds"][-12:],
        y1=plot_df2["NBEATSx-lo-90"][-12:].values,
        y2=plot_df2["NBEATSx-hi-90"][-12:].values,
        alpha=0.4,
        label="level 90",
    )
    axes[1].legend()
    axes[1].grid()
    axes[1].set_title("Airline2")

    plt.tight_layout()

    fig.savefig("forecast_plot_test.png")


def test_train_nbeatsx(ts: pd.DataFrame, shards: list, feat_cols: list):
    ts = ts.filter(["ds", "unique_id", "y"] + feat_cols + ["ts_len"])

    df_forecats = pd.DataFrame()
    forecats_per_shard = []
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

        # test_dates = df_test["ds"].unique()

        df_forecast_shard = pd.DataFrame()

        for ts_len in df_train_val["ts_len"].unique():
            val_size = 4
            input_size = 4
            forecast_horizon = 4
            if ts_len < 32:
                val_size = 2
                input_size = 2
                forecast_horizon = 2

            Y_train_df = (
                df_train_val[df_train_val["ts_len"] == ts_len]
                .copy()
                .drop(columns=["ts_len"])
            )
            Y_test_df = (
                df_test[df_test["ts_len"] == ts_len].copy().drop(columns=["ts_len"])
            )

            # Create dummy variables for unique_id

            static_ts_key_cols = pd.get_dummies(Y_train_df["unique_id"])
            static_ts_key_cols["unique_id"] = Y_train_df["unique_id"]

            for col in static_ts_key_cols.columns:
                if col not in ["unique_id"]:
                    static_ts_key_cols[col] = static_ts_key_cols[col].astype("int32")

            model = NBEATSx(
                h=forecast_horizon,
                input_size=2 if ts_len < 32 else input_size,
                loss=MQLoss(level=[80, 90]),
                scaler_type="robust",
                dropout_prob_theta=0.5,
                stat_exog_list=Y_train_df.unique_id.unique().tolist(),
                futr_exog_list=feat_cols,
                max_steps=100,
                val_check_steps=5,
                early_stop_patience_steps=2,
            )

            nf = NeuralForecast(models=[model], freq="MS")
            nf.fit(df=Y_train_df, static_df=static_ts_key_cols, val_size=val_size)
            Y_hat_df = nf.predict(futr_df=Y_test_df)

            # Plot quantile predictions
            df_forecast_shard = (
                Y_hat_df.reset_index(drop=True)
                .rename(
                    columns={
                        "ds": "Timestamp",
                        "unique_id": "ts_key",
                        "NBEATSx-median": "NBEATSx",
                    }
                )
                .filter(["Timestamp", "ts_key", "NBEATSx"])
            )

            df_forecast_shard["test_frame"] = test_frame

            forecats_per_shard.append(df_forecast_shard)

        df_forecats = pd.concat(forecats_per_shard)

    return df_forecats


if __name__ == "__main__":
    # sample_nbeatsx_forecast()

    ts = pd.read_parquet("./data/gold/timeseries_gold.parquet")
    ts.rename(
        columns={"Timestamp": "ds", "ts_key": "unique_id", "Vol/Prod_ratio_kg": "y"},
        inplace=True,
    )

    # Create the Timestamps splits for Train, Validation and Test
    # Train = [: shard[0]]
    # Validation = [shard[1] : shard[2]]
    # Test = [shard[3] : shard[4]]
    shards = [
        [
            datetime(2021, 8, 1),
            datetime(2021, 9, 1),
            datetime(2021, 12, 1),
            datetime(2022, 1, 1),
            datetime(2022, 4, 1),
        ],
        [
            datetime(2021, 12, 1),
            datetime(2022, 1, 1),
            datetime(2022, 4, 1),
            datetime(2022, 5, 1),
            datetime(2022, 8, 1),
        ],
        [
            datetime(2022, 2, 1),
            datetime(2022, 3, 1),
            datetime(2022, 6, 1),
            datetime(2022, 7, 1),
            datetime(2022, 10, 1),
        ],
    ]

    feat_cols = [
        "trend_Lag_2",
        "sesonality_Lag_12",
        "Vol/Prod_ratio_kg_Rolling_Mean_6",
        "residuals_Rolling_Mean_6",
        "Vol/Prod_ratio_kg_Lag_2",
    ]

    df_feat_importance = pd.read_parquet(
        "./data/gold/lgbm_covid_feat_importance.parquet"
    )

    feat_cols = df_feat_importance["Feature"].head(20).tolist()

    df_result_nbeatsx = test_train_nbeatsx(ts=ts, shards=shards, feat_cols=feat_cols)

    df_result_nbeatsx.to_parquet("./data/forecasts/nbeatsx_forecast.parquet")
