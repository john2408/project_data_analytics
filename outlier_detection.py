from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def sample_outlier_detection():
    # Example time series data
    N = 1000
    np.random.seed(1)
    df = pd.DataFrame(
        {"MW": np.sin(np.linspace(0, 10, num=N)) + np.random.normal(scale=0.6, size=N)}
    )

    # Apply Z-score with rolling window
    z, avg, std, mask = rolling_zscore(df["MW"], window=50, return_all=True)


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
    plt.show()

    return plt.gcf()


def apply_outlier_cleaning(
    input_path: str,
    output_pdf_path: str,
    output_parquet_path: str,
    target_col: str,
    window: int,
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


if __name__ == "__main__":
    df_ratio_gold_clean = apply_outlier_cleaning(
        input_path="./data/gold/ratio_vol_prod_gold.parquet",
        output_pdf_path="./reports/outlier_detection.pdf",
        output_parquet_path="./data/gold/timeseries_gold_outlier_clean.parquet",
        target_col="Vol/Prod_ratio_kg",
        window=12,
    )
