import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import List, Dict


def seasonal_decomposition(
    ts_series: pd.Series, ts_key: str, period: int = 12, model: str = "additive"
) -> pd.DataFrame:
    """Generate Seasonal Decomposition Analysis as dataframe
    of the format:

    - ts_key (str)
    - trend (float64)
    - seasonality (float64)
    - residuals (float64)

    Args:
        ts_series (pd.Series): input timeseries
        ts_key (str): timeseries key
        period (int, optional): period. Defaults to 12.
        model (str, optional): model. Defaults to 'additive'.

    Returns:
        pd.DataFrame: sesonal decomposition dataframe
    """

    df_decomposition = None

    ts_len = len(ts_series)

    # adjust seasonality based on ts length
    if ts_len > 24:
        period = 12
    elif ts_len > 12:
        period = 6

    try:

        decomposition = sm.tsa.seasonal_decompose(ts_series, period=period, model=model)

        df_decomposition = pd.DataFrame(
            {
                "trend": decomposition.trend,
                "sesonality": decomposition.seasonal,
                "residuals": decomposition.resid,
            }
        )

        df_decomposition = df_decomposition.bfill().ffill()

        df_decomposition["ts_key"] = ts_key

    except Exception as e:
        print(e)

    return df_decomposition


def apply_feature_eng(
    df_ratio_gold: pd.DataFrame,
    df_ts_decomposition: pd.DataFrame,
    df_covid: pd.DataFrame,
    config: Dict,
) -> pd.DataFrame:
    """Apply feature engineering to ratio volume/production

    Args:
        df_ratio_gold (pd.DataFrame): dataframe with ratio volume /prod
        df_ts_decomposition (pd.DataFrame): timeseries decomposition
        df_covid (pd.DataFrame): monthly covid data
        config (Dict): feature engineering config

    Returns:
        pd.DataFrame: Timeseries Gold Tables with all features
    """

    lag_months = config["lag_months"]
    rolling_months = config["rolling_months"]
    target_col = config["target_col"]
    drop_cols = config["drop_cols"]
    expected_vol_col = config["expected_vol_col"]
    expected_vol_col_rename = config["expected_vol_col_rename"]
    config_seasonal_feat = config["seasonal_features"]
    config_covid_feat = config["covid_features"]

    df_ratio_gold.rename(
        columns={expected_vol_col: expected_vol_col_rename}, inplace=True
    )
    df_ratio_gold.drop(columns=drop_cols, inplace=True)
    df_ratio_gold = categorical_features(df_ratio_gold=df_ratio_gold)
    df_ratio_gold = time_features(df_ratio_gold=df_ratio_gold)

    # For target Column
    df_ratio_gold = lag_features(
        df_ratio_gold=df_ratio_gold, target_col=target_col, lag_months=lag_months
    )
    df_ratio_gold = rolling_features(
        df_ratio_gold=df_ratio_gold,
        target_col=target_col,
        rolling_months=rolling_months,
    )

    # For Expected Volume Columns
    df_ratio_gold = lag_features(
        df_ratio_gold=df_ratio_gold,
        target_col=expected_vol_col_rename,
        lag_months=lag_months,
    )
    df_ratio_gold = rolling_features(
        df_ratio_gold=df_ratio_gold,
        target_col=expected_vol_col_rename,
        rolling_months=rolling_months,
    )
    df_ratio_gold.drop(columns=[expected_vol_col_rename], inplace=True)

    df_ratio_gold = add_seasonal_features(
        df_ratio_gold=df_ratio_gold, df_ts_decomposition=df_ts_decomposition
    )
    
    for ref_col, col_config in config_seasonal_feat.items():

        print("Calculating lags and rolling features for ", ref_col)
        df_ratio_gold = lag_features(
            df_ratio_gold=df_ratio_gold,
            target_col=col_config['col_name'],
            lag_months=col_config['lag_months'],
        )
        df_ratio_gold = rolling_features(
            df_ratio_gold=df_ratio_gold,
            target_col=col_config['col_name'],
            rolling_months=col_config['rolling_months'],
        )

        # Drop Column to avoid data leakage
        df_ratio_gold.drop(columns=[col_config['col_name']], inplace=True)

    
    df_ratio_gold = add_covid_data(df_ratio_gold=df_ratio_gold, df_covid=df_covid)
   
    #Calculate Covid Features for each country
    for col_name in config_covid_feat['cols_names']:

        print("Calculating lags and rolling features for ", col_name)

        df_ratio_gold = lag_features(
            df_ratio_gold=df_ratio_gold,
            target_col=col_name,
            lag_months=config_covid_feat['lag_months'],
        )
        df_ratio_gold = rolling_features(
            df_ratio_gold=df_ratio_gold,
            target_col=col_name,
            rolling_months=config_covid_feat['rolling_months'],
        )

        # Drop Column to avoid data leakage
        df_ratio_gold.drop(columns=[col_name], inplace=True)

    return df_ratio_gold


def features_seasonal_decomposition(
    df_ratio_gold: pd.DataFrame, target_col: str
) -> pd.DataFrame:
    """Get the seasonal decomposition features of the timeseries

    Args:
        df_ratio_gold (pd.DataFrame): vol/production ratio dataframe
        target_col (str): target column

    Returns:
        pd.DataFrame: seasonal decomposition features of every timeseries
    """

    dfs = []
    for ts_key in df_ratio_gold["ts_key"].unique():

        ts_series = df_ratio_gold[df_ratio_gold["ts_key"] == ts_key][target_col].copy()

        _df = seasonal_decomposition(
            ts_series=ts_series, ts_key=ts_key, period=12, model="additive"
        )

        _df["Timestamp"] = df_ratio_gold[df_ratio_gold["ts_key"] == ts_key]["Timestamp"]

        if _df is not None:
            dfs.append(_df)
        del _df

    # join all the decompsitions values
    df_ts_decomposition = pd.concat(dfs)

    assert (
        df_ts_decomposition["ts_key"].nunique() == df_ratio_gold["ts_key"].nunique(),
        "There are missing timeseries",
    )

    return df_ts_decomposition


def categorical_features(df_ratio_gold: pd.DataFrame) -> pd.DataFrame:
    """Generate categorical Features

    Args:
        df_ratio_gold (pd.DataFrame): vol/production ratio dataframe

    Returns:
        pd.DataFrame: categorial features
    """
    df_ratio_gold["Provider"] = df_ratio_gold["ts_key"].apply(lambda x: x.split("-")[0])

    return df_ratio_gold


def time_features(df_ratio_gold: pd.DataFrame) -> pd.DataFrame:
    """Generate Time Features

    Args:
        df_ratio_gold (pd.DataFrame): vol/production ratio dataframe

    Returns:
        pd.DataFrame: time features
    """

    df_ratio_gold["Month"] = df_ratio_gold["Timestamp"].dt.month.astype(np.int8)
    df_ratio_gold["Year"] = df_ratio_gold["Timestamp"].dt.year

    return df_ratio_gold


def lag_features(
    df_ratio_gold: pd.DataFrame, target_col: str, lag_months: List[int]
) -> pd.DataFrame:
    """Generate Lag Features

    Args:
        df_ratio_gold (pd.DataFrame): vol/production ratio dataframe
        target_col (str): target column
        lag_months: List[int]: lag months

    Returns:
        pd.DataFrame: lag features
    """

    df_ratio_gold = df_ratio_gold.assign(
        **{
            f"{target_col}_Lag_{lag}": df_ratio_gold.groupby(["ts_key"])[
                target_col
            ].transform(lambda x: x.shift(lag))
            for lag in lag_months
        }
    )

    return df_ratio_gold


def rolling_features(
    df_ratio_gold: pd.DataFrame, target_col: str, rolling_months: List[int]
) -> pd.DataFrame:
    """Generate Rolling Features

    Args:
        df_ratio_gold (pd.DataFrame): vol/production ratio dataframe
        target_col (str): target column
        rolling_months (List[int]): rolling months

    Returns:
        pd.DataFrame: Rolling Features
    """
    START_LAG = 2

    for i in rolling_months:
        df_ratio_gold[f"{target_col}_Rolling_Mean_" + str(i)] = df_ratio_gold.groupby(["ts_key"])[
            target_col
        ].transform(lambda x: x.shift(START_LAG).rolling(i).mean())
        df_ratio_gold[f"{target_col}_Rolling_std_" + str(i)] = df_ratio_gold.groupby(["ts_key"])[
            target_col
        ].transform(lambda x: x.shift(START_LAG).rolling(i).std())

    df_ratio_gold = df_ratio_gold.backfill()

    return df_ratio_gold


def add_covid_data(df_ratio_gold: pd.DataFrame, df_covid: pd.DataFrame):
    """Add covid data data as columns.

    Args:
        df_ratio_gold (pd.DataFrame): vol/production ratio dataframe
        df_covid (pd.DataFrame): monthly covid data

    Returns:
        _type_: _description_
    """

    df_ratio_gold = pd.merge(df_ratio_gold, df_covid, on=["Timestamp"], how="left")

    df_ratio_gold = df_ratio_gold.fillna(0)

    return df_ratio_gold


def add_seasonal_features(
    df_ratio_gold: pd.DataFrame, df_ts_decomposition: pd.DataFrame
) -> pd.DataFrame:
    """Add Seasonal Features

    Args:
        df_ratio_gold (pd.DataFrame): vol/production ratio dataframe
        df_ts_decomposition (pd.DataFrame): seasonal decomposition features

    Returns:
        pd.DataFrame: _description_
    """

    df_ratio_gold = pd.merge(
        df_ratio_gold, df_ts_decomposition, on=["Timestamp", "ts_key"], how="left"
    )

    return df_ratio_gold
