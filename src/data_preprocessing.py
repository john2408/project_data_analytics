import pandas as pd
import numpy as np
from datetime import datetime


def preprocessing_volume_data(df_vol: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing steps for volume data

    Args:
        df_vol (pd.DataFrame): historical volume data

    Returns:
        pd.DataFrame: preprocessed volume data
    """
    df_vol["Year"] = df_vol["Timestamp"].apply(lambda x: x.split("/")[0]).astype(int)
    df_vol["Month"] = df_vol["Timestamp"].apply(lambda x: x.split("/")[1]).astype(int)
    df_vol["Timestamp"] = df_vol[["Year", "Month"]].apply(
        lambda x: datetime(x["Year"], x["Month"], 1), axis=1

    )
    # Create Timeseries Key
    df_vol["ts_key"] = df_vol[["Provider", "Plant"]].apply(
        lambda x: x["Provider"] + "-" + x["Plant"], axis=1
    )

    df_vol["Actual Vol [Kg]"] = (
    df_vol["Actual Vol [Kg]"].str.replace(".", "").astype("float")
    )
    df_vol["Expected Vol [Kg]"] = df_vol["Expected Vol [Kg]"].str.replace(".", "")
    df_vol["Expected Vol [Kg]"] = (
        df_vol["Expected Vol [Kg]"].str.replace(",", ".").astype("float")
    )

    # We can transform the units to Tons for ease of manipulation and plotting
    df_vol["Actual Vol [Tons]"] = np.round(df_vol["Actual Vol [Kg]"] / 1000, 3)
    df_vol["Expected Vol [Tons]"] = np.round(df_vol["Expected Vol [Kg]"] / 1000, 3)

    df_vol.columns = df_vol.columns.str.replace(" ", "_")
    df_vol['Provider'] = df_vol['Provider'].astype('category')
    df_vol['Plant'] = df_vol['Plant'].astype('category')
    df_vol['ts_key'] = df_vol['ts_key'].astype('category')

    return df_vol


def preprocessing_production(df_prod: pd.DataFrame) -> pd.DataFrame:
    """preprocess production data

    Args:
        df_prod (pd.DataFrame): historical production data

    Returns:
        pd.DataFrame: preprocessed production data
    """
    # Convert the Timestamp to a Datetime Object
    df_prod["Year"] = df_prod["Timestamp"].apply(lambda x: x.split("/")[0]).astype(int)
    df_prod["Month"] = df_prod["Timestamp"].apply(lambda x: x.split("/")[1]).astype(int)
    df_prod["Timestamp"] = df_prod[["Year", "Month"]].apply(
        lambda x: datetime(x["Year"], x["Month"], 1), axis=1
    )

    # Unpivot columns to rows
    df_prod = pd.melt(
        df_prod.drop(columns=["Month", "Year"]),
        id_vars=["Timestamp"],
        var_name="Plant",
        value_name="Production",
    )

    return df_prod

def preprocesing_covid(df_covid: pd.DataFrame) -> pd.DataFrame:
    """Returns the Covid rate cases per country on monthly bases

    Args:
        df_covid (pd.DataFrame): input raw covid cases on biweekly basis

    Returns:
        pd.DataFrame: covid cases on monthly basis
    """

    df_covid['timestamp'] = pd.to_datetime(df_covid['year_week'] + '-1', format='%Y-%U-%w')

    # Some Facts avoid covid data before pivoting it
    print("The Covid data ranges from", df_covid['timestamp'].min(), " until ", df_covid['timestamp'].max())
    print("The file contains data for ", df_covid['country'].nunique(), " countries.")
    print("The file contains data for ", df_covid['age_group'].nunique(), " age groups ", df_covid['age_group'].unique())
    print("in Total it contains ", df_covid.shape[0], " rows.")
    print("in Total it contains ", df_covid.shape[1], " columns.")

    # Calculate the monthly covid rate to align it with the monthly data
    df_covid_rate =  df_covid[['timestamp','country','rate_14_day_per_100k']].drop_duplicates().copy()
    df_covid_rate["Year"] = df_covid_rate["timestamp"].dt.year
    df_covid_rate["Month"] = df_covid_rate["timestamp"].dt.month
    df_covid_rate = df_covid_rate.groupby(['Year','Month','country'], as_index=False)['rate_14_day_per_100k'].sum()
    df_covid_rate["Timestamp"] = df_covid_rate[["Year", "Month"]].apply(
        lambda x: datetime(x["Year"], x["Month"], 1), axis=1
    )
    df_covid_rate_country = df_covid_rate.pivot(index='Timestamp', columns='country', values='rate_14_day_per_100k')

    return df_covid_rate_country