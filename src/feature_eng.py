import pandas as pd
import statsmodels.api as sm

def seasonal_decomposition(ts_series: pd.Series,
                           ts_key: str, 
                           period: int=12,
                           model: str='additive') -> pd.DataFrame:
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

        decomposition = sm.tsa.seasonal_decompose(ts_series,
                                        period=period, 
                                        model=model) 
        
        df_decomposition = pd.DataFrame({'trend': decomposition.trend, 
                        'sesonality': decomposition.seasonal,
                        'residuals': decomposition.resid})
        
        df_decomposition = df_decomposition.bfill().ffill()

        df_decomposition['ts_key'] = ts_key  
    
    except Exception as e:
        print(e)

    return df_decomposition


def features_seasonal_decomposition(df_ratio_gold: pd.DataFrame) -> pd.DataFrame:
    """Get the seasonal decomposition features of the timeseries

    Args:
        df_ratio_gold (pd.DataFrame): vol/production ratio dataframe

    Returns:
        pd.DataFrame: seasonal decomposition features of every timeseries
    """

    dfs = []
    for ts_key in df_ratio_gold["ts_key"].unique():

        ts_series = df_ratio_gold[df_ratio_gold["ts_key"] == ts_key][col].copy()

        _df = seasonal_decomposition(ts_series=ts_series,
                            ts_key=ts_key, 
                            period=12,
                            model='additive')
        
        _df['Timestamp'] = df_ratio_gold[df_ratio_gold["ts_key"] == ts_key]['Timestamp']

        if _df is not None:
            dfs.append(_df)
        del _df

    # join all the decompsitions values
    df_ts_decomposition = pd.concat(dfs)

    assert (df_ts_decomposition['ts_key'].nunique()==df_ratio_gold['ts_key'].nunique(), 
            "There are missing timeseries")
    
    return df_ts_decomposition

