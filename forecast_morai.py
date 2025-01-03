import torch
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from huggingface_hub import hf_hub_download

from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule


def test_sample_forecast():
    MODEL = "moirai-moe"  # model name: choose from {'moirai', 'moirai-moe'}
    SIZE = "small"  # model size: choose from {'small', 'base', 'large'}
    PDT = 20  # prediction length: any positive integer
    CTX = 200  # context length: any positive integer
    PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
    BSZ = 32  # batch size: any positive integer
    TEST = 100  # test set length: any positive integer

    # Read data into pandas DataFrame
    url = (
        "https://gist.githubusercontent.com/rsnirwan/c8c8654a98350fadd229b00167174ec4"
        "/raw/a42101c7786d4bc7695228a0f2c8cea41340e18f/ts_wide.csv"
    )
    df = pd.read_csv(url, index_col=0, parse_dates=True)

    # Convert into GluonTS dataset
    ds = PandasDataset(dict(df))

    # Split into train/test set
    train, test_template = split(
        ds, offset=-TEST
    )  # assign last TEST time steps as test set

    # Construct rolling window evaluation
    test_data = test_template.generate_instances(
        prediction_length=PDT,  # number of time steps for each prediction
        windows=TEST // PDT,  # number of windows in rolling window evaluation
        distance=PDT,  # number of time steps between each window - distance=PDT for non-overlapping windows
    )

    # Prepare pre-trained model by downloading model weights from huggingface hub
    if MODEL == "moirai":
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{SIZE}"),
            prediction_length=PDT,
            context_length=CTX,
            patch_size=PSZ,
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    elif MODEL == "moirai-moe":
        model = MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained(
                f"Salesforce/moirai-moe-1.0-R-{SIZE}"
            ),
            prediction_length=PDT,
            context_length=CTX,
            patch_size=16,
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )

    predictor = model.create_predictor(batch_size=BSZ)
    forecasts = predictor.predict(test_data.input)

    input_it = iter(test_data.input)
    label_it = iter(test_data.label)
    forecast_it = iter(forecasts)

    inp = next(input_it)
    label = next(label_it)
    forecast = next(forecast_it)

    # gluonts.model.forecast.SampleForecast
    print(type(forecast))

    forecast_df = pd.DataFrame(
        {
            "mean": forecast.mean,
            "median": forecast.quantile(0.5),
            "0.1_quantile": forecast.quantile(0.1),
            "0.9_quantile": forecast.quantile(0.9),
        }
    )
    print(forecast_df)

    # Fix issue 87
    # https://github.com/SalesforceAIResearch/uni2ts/issues/87
    # add 'pass' in torch.with_no_grad() to avoid error in file
    # /workspaces/project_data_analytics/.morai/lib/python3.12/site-packages/gluonts/torch/model/predictor.py

    plot_single(
        inp,
        label,
        forecast,
        context_length=200,
        name="pred",
        show_label=True,
    )
    plt.show()
    # plt.savefig("forecast_plot.png")


def test_train_llm_morai(
    df_timeseries_gold: pd.DataFrame, shards: list
) -> pd.DataFrame:
    df_forecats = pd.DataFrame()
    ts = df_timeseries_gold.copy()

    for shard in shards:
        test_frame = (
            shard[3].strftime("%Y-%m-%d") + " - " + shard[4].strftime("%Y-%m-%d")
        )
        print(
            "Train-Testing for",
            shard[3].strftime("%Y-%m-%d"),
            shard[4].strftime("%Y-%m-%d"),
        )

        # df_train_val_test: since we will use the gluonts split method, we need to provide the entire dataset in the given shard
        # df_test: to get the timestamps for the test set
        df_train_val_test, df_test = (
            ts[(ts["Timestamp"] <= shard[4])],
            ts[(ts["Timestamp"] >= shard[3]) & (ts["Timestamp"] <= shard[4])],
        )
        test_dates = df_test["Timestamp"].unique()

        df_forecast_shard = pd.DataFrame()

        for ts_len in df_train_val_test["ts_len"].unique():
            _df = (
                df_train_val_test[df_train_val_test["ts_len"] == ts_len]
                .set_index("Timestamp")
                .copy()
            )

            MODEL = "moirai-moe"  # model name: choose from {'moirai', 'moirai-moe'}
            SIZE = "small"  # model size: choose from {'small', 'base', 'large'}
            PDT = 4  # prediction length: any positive integer
            CTX = ts_len - PDT  # context length: any positive integer
            PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
            BSZ = 4  # batch size: any positive integer
            TEST = 4  # test set length: any positive integer

            patch_size = 4
            num_samples = 4
            target_dim = 1

            # Convert into GluonTS dataset
            # Ref: https://ts.gluon.ai/stable/tutorials/data_manipulation/pandasdataframes.html
            ds = PandasDataset.from_long_dataframe(
                _df, target="Vol/Prod_ratio_kg", item_id="ts_key"
            )

            # Split into train/test set
            train, test_template = split(
                ds, offset=-TEST
            )  # assign last TEST time steps as test set

            # Construct rolling window evaluation
            test_data = test_template.generate_instances(
                prediction_length=PDT,  # number of time steps for each prediction
                windows=TEST // PDT,  # number of windows in rolling window evaluation
                distance=PDT,  # number of time steps between each window - distance=PDT for non-overlapping windows
            )

            model = MoiraiMoEForecast(
                module=MoiraiMoEModule.from_pretrained(
                    f"Salesforce/moirai-moe-1.0-R-{SIZE}"
                ),
                prediction_length=PDT,
                context_length=CTX,
                patch_size=patch_size,
                num_samples=num_samples,
                target_dim=target_dim,
                feat_dynamic_real_dim=ds.num_feat_dynamic_real,
                past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
            )

            predictor = model.create_predictor(batch_size=BSZ)
            forecasts = predictor.predict(test_data.input)

            _dfs = []
            for forecast, label in zip(iter(forecasts), iter(test_data.label)):
                forecast_df = pd.DataFrame(
                    {
                        # "mean": forecast.mean,
                        "MORAI": forecast.quantile(0.5),
                        # "0.1_quantile": forecast.quantile(0.1),
                        # "0.9_quantile": forecast.quantile(0.9),
                    }
                )
                forecast_df["ts_key"] = label["item_id"]
                forecast_df["Timestamp"] = test_dates
                _dfs.append(forecast_df)

            df_forecast_ts_key = pd.concat(_dfs)

            df_forecast_shard = pd.concat([df_forecast_shard, df_forecast_ts_key])
            
            del _df, _dfs, df_forecast_ts_key

        print("     Finished Forecasting for test frame", test_frame)
        df_forecast_shard["test_frame"] = test_frame

    df_forecats = pd.concat([df_forecats, df_forecast_shard])

    return df_forecats


if __name__ == "__main__":
    # test_sample_forecast()

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

    df_timeseries_gold = pd.read_parquet("./data/gold/timeseries_gold.parquet")
    df_result_morai = test_train_llm_morai(
        df_timeseries_gold=df_timeseries_gold, shards=shards
    )
    df_result_morai.to_parquet("./data/forecast/morai_forecast.parquet")

    print("Test")