import torch
import pandas as pd 
import numpy as np
from datetime import datetime
from src.utils import read_config
from chronos import BaseChronosPipeline
import requests
from huggingface_hub import configure_http_backend

# Fix SSL Error 
# https://stackoverflow.com/questions/71692354/facing-ssl-error-with-huggingface-pretrained-models#:~:text=Downgrading%20requests%20to%202.27.1%20and%20adding%20the%20following,likely%20to%20want%20to%20use%20the%20datasets%20package.

# def backend_factory() -> requests.Session:
#     session = requests.Session()
#     session.verify = False
#     return session

# configure_http_backend(backend_factory=backend_factory)

pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-base",
    #device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
    torch_dtype=torch.bfloat16,
)

ts = pd.read_parquet("./data/gold/timeseries_gold.parquet")

prediction_length = 4 # 4-month ahead forecast

# Create the Timestamps splits for Train, Validation and Test
# Train = [: shard[0]]
# Validation = [shard[1] : shard[2]]
# Test = [shard[3] : shard[4]]
shards = [
    [datetime(2021,8,1), datetime(2021,9,1), datetime(2021,12,1), datetime(2022,1,1), datetime(2022,4,1)],
    [datetime(2021,12,1), datetime(2022,1,1), datetime(2022,4,1), datetime(2022,5,1), datetime(2022,8,1)],
    [datetime(2022,2,1), datetime(2022,3,1), datetime(2022,6,1), datetime(2022,7,1), datetime(2022,10,1) ],
]

df_forecats = pd.DataFrame()

for shard in shards:
    test_frame = shard[3].strftime("%Y-%m-%d") + " - " + shard[4].strftime("%Y-%m-%d")
    print("Train-Testing for", shard[3].strftime("%Y-%m-%d"), shard[4].strftime("%Y-%m-%d"))

    # Train and Validation Set are input together as one dataset
    # Prediction set is our 4-month window
    df_train_val, df_test = (
        ts[(ts["Timestamp"] <= shard[2])],
        ts[(ts["Timestamp"] >= shard[3]) & (ts["Timestamp"] <= shard[4])],
    )

    test_dates = df_test['Timestamp'].unique()

    df_forecast_shard = pd.DataFrame()
    for ts_len in ts['ts_len'].unique():

        _df = ts[ts['ts_len'] == ts_len].copy()

        grouped_data = _df.groupby('ts_key')['Vol/Prod_ratio_kg'].apply(list).tolist()

        context = torch.tensor(grouped_data)

        # predict using LLM model by passing context data
        forecasts = pipeline.predict(context, prediction_length)  

        df_forecast_ts_key = pd.DataFrame()
        for ts_key, forecast in zip(_df['ts_key'].unique(), forecasts):
            low, median, high = np.quantile(forecast.numpy(), [0.1, 0.5, 0.9], axis=0)

            df_forecast_ts_key = pd.DataFrame({'CHRONOS_lower': low,
                               'CHRONOS':median,
                               'CHRONOS_higher':high
                               })
            df_forecast_ts_key['ts_key'] = ts_key
            df_forecast_ts_key['Timestamp'] = test_dates

            df_forecast_shard = pd.concat([df_forecast_shard, df_forecast_ts_key])

    df_forecast_shard['test_frame']  = test_frame

    df_forecats = pd.concat([df_forecats, df_forecast_shard])
    

df_forecats.drop(columns=['CHRONOS_lower','CHRONOS_higher'], inplace=True)

df_forecats.to_parquet("./data/forecasts/chronos_bolt_forecast.parquet")


