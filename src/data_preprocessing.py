from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def rolling_zscore(series: pd.Series, window: int, threshold: int = 3) -> tuple:
    """
    Detect outliers in a time series using a rolling window Z-score method.

    Parameters:
    - series: pandas Series, the time series data.
    - window: int, the size of the rolling window.
    - threshold: float, the Z-score threshold to identify outliers.
    - return_all: bool, whether to return all intermediate calculations.

    Returns:
    - If return_all is True: Z-score, rolling mean, rolling std, mask for non-outliers.
    - If return_all is False: Series with outliers replaced by rolling mean.
    """
    # Calculate rolling statistics
    roll = series.rolling(window=window, min_periods=1, center=True)
    avg = roll.mean()
    std = roll.std(ddof=0)

    # Compute Z-score
    z = (series - avg) / std
    mask = z.abs() <= threshold

    # Return values based on return_all flag
    # return series.where(mask, avg)
    return z, avg, std, mask


def plot_outlier_detection(
    df: pd.DataFrame, ts_key: str, target_col: str, avg: pd.Series, mask: pd.Series
):
    """Plot the original time series data with rolling mean and detected outliers.

    Args:
        df (pd.DataFrame): time series data
        ts_key (str): time series key
        target_col (str): target column for outlier detection
        avg (pd.Series): rolling mean values
        mask (pd.Series): boolean array indicating outliers

    Returns:
        _type_: plot object
    """
    # Plot results
    plt.figure(figsize=(10, 6))
    df[target_col].plot(label="Original Data", alpha=0.6)
    avg.plot(label="Rolling Mean", linestyle="--")

    outliers = df.loc[~mask, target_col]
    if not outliers.empty:
        outliers.plot(label="Outliers", marker="o", linestyle="", color="red")
        avg[~mask].plot(label="Replacement", marker="o", linestyle="", color="orange")
    plt.legend()
    plt.title("Outlier Detection with Rolling Z-Score for Time Series: " + ts_key)

    return plt.gcf()


def apply_outlier_cleaning(
    input_path: str,
    output_pdf_path: str,
    output_parquet_path: str,
    target_col: str,
    window: int,
    verbose: bool = False,
) -> pd.DataFrame:
    """Apply outlier detection and cleaning to a time series dataset.

    Args:
        input_path (str): input path to the time series data
        output_pdf_path (str): output path for the PDF report
        output_parquet_path (str): output path for the cleaned time series data
        target_col (str): target column for outlier detection
        window (int): window size for rolling Z-score calculation

    Returns:
        pd.DataFrame: cleaned time series data with outliers replaced by rolling mean
    """

    df_ratio_gold = pd.read_parquet(input_path)

    with PdfPages(output_pdf_path) as pdf:
        dfs = []
        for ts_key in df_ratio_gold.ts_key.unique():
            if verbose:
                print(f"Processing ts_key: {ts_key}")

            ts = df_ratio_gold[df_ratio_gold.ts_key == ts_key].copy()
            ts.set_index("Timestamp", inplace=True)

            # Apply Z-score with rolling window
            z, avg, std, mask = rolling_zscore(ts[target_col], window=window)

            fig = plot_outlier_detection(
                df=ts, target_col=target_col, ts_key=ts_key, avg=avg, mask=mask
            )

            # Replace outliers with rolling mean
            adjusted_series = ts[target_col].where(mask, avg).fillna(0)
            ts[target_col] = adjusted_series

            # saves the current figure into a pdf page
            pdf.savefig(fig)
            # plt.savefig(f"./reports/outliers/outlier_detection_{ts_key}.png")
            plt.close(fig)

            dfs.append(ts)

            assert (
                ts.shape[0] == df_ratio_gold[df_ratio_gold.ts_key == ts_key].shape[0]
            )  # Check if we didn't change the size of
            assert (
                ts[target_col].isna().sum() == 0
            )  # Check if we didn't introduce any NaNs in the
            del ts

        # We can also set the file's metadata via the PdfPages object:
        d = pdf.infodict()
        d["Title"] = "Timeseries Vol-Prod-Ratio"
        d["Author"] = "John Torres"
        d["CreationDate"] = datetime.today()

        plt.close("all")

    df_ratio_gold_clean = pd.concat(dfs)
    df_ratio_gold_clean.reset_index(inplace=True)
    df_ratio_gold_clean.to_parquet(output_parquet_path)

    return df_ratio_gold_clean


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
    # df_vol['Provider'] = df_vol['Provider'].astype('category')
    # df_vol['Plant'] = df_vol['Plant'].astype('category')
    # df_vol['ts_key'] = df_vol['ts_key'].astype('category')

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

    df_covid["timestamp"] = pd.to_datetime(
        df_covid["year_week"] + "-1", format="%Y-%U-%w"
    )

    # Some Facts avoid covid data before pivoting it
    print(
        "The Covid data ranges from",
        df_covid["timestamp"].min(),
        " until ",
        df_covid["timestamp"].max(),
    )
    print("The file contains data for ", df_covid["country"].nunique(), " countries.")
    print(
        "The file contains data for ",
        df_covid["age_group"].nunique(),
        " age groups ",
        df_covid["age_group"].unique(),
    )
    print("in Total it contains ", df_covid.shape[0], " rows.")
    print("in Total it contains ", df_covid.shape[1], " columns.")

    # Calculate the monthly covid rate to align it with the monthly data
    df_covid_rate = (
        df_covid[["timestamp", "country", "rate_14_day_per_100k"]]
        .drop_duplicates()
        .copy()
    )
    df_covid_rate["Year"] = df_covid_rate["timestamp"].dt.year
    df_covid_rate["Month"] = df_covid_rate["timestamp"].dt.month
    df_covid_rate = df_covid_rate.groupby(["Year", "Month", "country"], as_index=False)[
        "rate_14_day_per_100k"
    ].sum()
    df_covid_rate["Timestamp"] = df_covid_rate[["Year", "Month"]].apply(
        lambda x: datetime(x["Year"], x["Month"], 1), axis=1
    )
    df_covid_rate_country = df_covid_rate.pivot(
        index="Timestamp", columns="country", values="rate_14_day_per_100k"
    )

    return df_covid_rate_country


def data_quality_vol_analysis(df_vol: pd.DataFrame) -> None:
    """Timeseries data quality checks for volume data

    Args:
        df_vol (pd.DataFrame): historical volume data in silver layer

    Returns:
        None
    """

    # Check the timeseries length
    df_vol["ts_len"] = df_vol.groupby("ts_key")["Timestamp"].transform(lambda x: len(x))

    # Create a Summary of all columns
    df_vol_summary = df_vol.groupby(["ts_key"]).agg(["min", "max"])
    df_vol_summary.columns = ["-".join(x) for x in df_vol_summary.columns]
    df_vol_summary = df_vol_summary.reset_index()

    print(
        " The min date available among all timeseries is: ",
        df_vol_summary["Timestamp-min"].min(),
    )
    print(
        " The max date available among all timeseries is: ",
        df_vol_summary["Timestamp-max"].max(),
    )

    print(" The min ts length is: ", df_vol_summary["ts_len-min"].min())
    print(" The max ts length is: ", df_vol_summary["ts_len-max"].max())

    ts_with_current_data = df_vol_summary[
        df_vol_summary["Timestamp-max"] == df_vol_summary["Timestamp-max"].max()
    ]["ts_key"].nunique()

    print(" Number of time series with data until October 2022: ", ts_with_current_data)
    print(
        " Number of Total Time Series Available: ", df_vol_summary["ts_key"].nunique()
    )
    print(
        " Number of Total Time Series Available for Prediction: ",
        np.round(ts_with_current_data / df_vol_summary["ts_key"].nunique(), 2) * 100,
        " %",
    )

    return df_vol_summary


def apply_data_quality_timeseries(
    df_vol: pd.DataFrame, df_vol_summary: pd.DataFrame, ts_len_threshold: int = 8
) -> pd.DataFrame:
    """Apply Data Quality to timeseries data. We apply the following data quality
    measures:
    - (1) Timeseries must have valid data at max date available
    - (2) Timeseries must have a lenght >= than ts_len_threshold
    - (3) Timeseries must not contain duplicates (No Duplicates)
    - (4) Timeseries must be contain values in all months (Completness)

    Args:
        df_vol (pd.DataFrame): volume data in silver layer
        df_vol_summary (pd.DataFrame): summary statistics for vol data
        ts_len_threshold (int): min numer of datapoints in a ts to be considered in the analysis.

    Returns:
        pd.DataFrame: gold volume data
    """

    # -----------------------------------------------------------------
    # (1) Timeseries must have valid data at max date available
    # -----------------------------------------------------------------

    # We keep only the timeseries which have data until the max date available
    max_date_valid_ts = df_vol_summary[
        (df_vol_summary["Timestamp-max"] == df_vol_summary["Timestamp-max"].max())
    ]["ts_key"].unique()

    # Now we just keep those valid timeseries from the data
    # and we overwrite our dataset
    df_vol = df_vol[df_vol["ts_key"].isin(max_date_valid_ts)].copy()

    # We just verify our filter was made correctly
    # We validate that all ts_key in our new df_vol
    # are the same as the one in the list valid_ts
    assert set(df_vol["ts_key"].unique()) == set(
        max_date_valid_ts
    ), "There are missing timeseries"

    print(
        "Number of available timeseries after first filtering:",
        df_vol["ts_key"].nunique(),
    )

    # After the filter we end up with 306 timeseries which can be forecast.
    # We analyze then some key metrics:
    print(" The min ts length is ", df_vol["ts_len"].min())
    print(" The max ts length is ", df_vol["ts_len"].max())
    print(" The mean ts length is ", df_vol["ts_len"].mean())

    # -----------------------------------------------------------------
    # (2) Timeseries must have a lenght >= than ts_len_threshold
    # -----------------------------------------------------------------
    remaining_ts = df_vol[df_vol["ts_len"] > ts_len_threshold]["ts_key"].unique()
    print(" TS to forecast with Models", len(remaining_ts))
    print(
        " TS to forecast with Models",
        np.round(len(remaining_ts) / len(max_date_valid_ts), 2) * 100,
        " %",
    )

    # Let`s now filter our data
    df_vol = df_vol[df_vol["ts_key"].isin(remaining_ts)].copy()

    # -----------------------------------------------------------------
    # (3) Timeseries must not contain duplicates
    # -----------------------------------------------------------------
    # The idea is the for every 'Timestamp','ts_key' combination
    # there should be only one entry for the column 'Actual Vol [Kg]'
    df_vol_c = (
        df_vol[["Timestamp", "ts_key", "Actual_Vol_[Tons]", "Expected_Vol_[Tons]"]]
        .groupby(["Timestamp", "ts_key"], group_keys=False)
        .agg(
            {
                "Actual_Vol_[Tons]": sum,
                "Expected_Vol_[Tons]": sum,
            }
        )
        .reset_index()
        .set_index("Timestamp")
    )

    # Verify that every Timestamp-Plant combination only contains one entry
    df_verify = (
        df_vol_c.groupby(["Timestamp", "ts_key"])["Actual_Vol_[Tons]"]
        .count()
        .to_frame()
        .rename(columns={"Actual_Vol_[Tons]": "n_values"})
    )
    assert df_verify["n_values"].unique() == np.array([1])

    # -----------------------------------------------------------------
    # (4) Timeseries must be contain values in all months (Completness)
    # -----------------------------------------------------------------
    ts = pd.DataFrame()
    # ref: https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases'
    for ts_key in df_vol_c["ts_key"].unique():
        # build custom date range for timeseries
        idx = pd.date_range(
            start=df_vol_c[df_vol_c["ts_key"] == ts_key].index.min(),
            end=df_vol_c.index.max(),
            freq="MS",  # Start of month
            name="Timestamp",
        )

        # fill holes in time series
        df = df_vol_c.loc[df_vol_c["ts_key"] == ts_key].reindex(idx)
        df.fillna(
            {
                "ts_key": ts_key,
                "Actual_Vol_[Tons]": 0,
                "Expected_Vol_[Tons]": 0,
            },
            inplace=True,
        )
        df["Actual_Vol_[Tons]"] = df["Actual_Vol_[Tons]"].astype(np.float32)
        df["Expected_Vol_[Tons]"] = df["Expected_Vol_[Tons]"].astype(np.float32)

        if ts.empty:
            ts = df
        else:
            ts = pd.concat([ts, df])

        del df

    ts.reset_index(inplace=True)

    # Create Column Plant
    ts["Plant"] = ts["ts_key"].apply(lambda x: x.split("-")[1])

    # We verify that all ts end at the max date
    max_date = ts["Timestamp"].max()
    max_date_all_ts = (
        ts[["ts_key", "Timestamp"]].groupby(["ts_key"])["Timestamp"].max().unique()[0]
    )
    assert max_date_all_ts == max_date, "There are timeseries ending at different dates"

    assert (
        ts[ts["Actual_Vol_[Tons]"].isna()].shape[0] == 0
    ), "There are NaN Values in Column Actual Vol Kgs"
    assert (
        ts[ts["Expected_Vol_[Tons]"].isna()].shape[0] == 0
    ), "There are NaN Values in Column Expected Vol Kgs"

    return ts
