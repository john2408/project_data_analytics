import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from typing import List, Dict
from src.utils import COLORS


def plot_f_system_smape_buckets_distribution(
    df_accuracy_smape: pd.DataFrame, figsize: tuple = (10, 6)
) -> Dict:
    """Generate buckets distribution plot for SMAPE

    Args:
        df_accuracy_smape (pd.DataFrame): accuracy values of models
        figsize (tuple, optional): figsize. Defaults to (10, 6).
    """
    # Define SMAPE buckets
    bins = [0, 10, 20, 30, 40, float("inf")]
    labels = ["0-10", "10-20", "20-30", "30-40", ">40"]
    model_names = ["best_model"]
    buckets_data = generate_smape_err_buckets(
        df_accuracy=df_accuracy_smape, model_names=model_names, bins=bins, labels=labels
    )

    colors = [
        "#7f7f7f",
        "#e377c2",
        "#8c564b",
        "#9467bd",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#1f77b4",
    ]

    x = np.arange(len(labels))  # x locations for the buckets
    width = 0.8 / len(model_names)  # Adjust width based on number of models

    fig, ax = plt.subplots(figsize=figsize)

    for i, (model_name, data) in enumerate(buckets_data.items()):
        ax.bar(
            x + i * width - width * len(model_names) / 2,
            data.values,
            width,
            label=model_name,
            color=COLORS.pop(),
            alpha=0.7,
        )

    # Add labels and title
    ax.set_xlabel("SMAPE Range")
    ax.set_ylabel("Count")
    ax.set_title("SMAPE Buckets Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.show()

    return buckets_data


def plot_smape_histogram(data: pd.DataFrame, figsize=(8, 6)):
    """Plot SMAPE histogram.

    Args:
        data (pd.DataFrame): data
        figsize (tuple, optional): figure size. Defaults to (8, 6).
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(data, bins=30, color="blue", alpha=0.7)
    ax.set_title("BEST SMAPE")
    ax.set_xlabel("SMAPE")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_model_distribution(
    df_model_per_ts: pd.DataFrame, figsize: tuple = (6, 6)
) -> None:
    """Generate best models per TS plot distribution

    Args:
        df_model_per_ts (pd.DataFrame): best models per timeseries
    """

    plt.figure(figsize=figsize)
    colors = sns.color_palette("pastel", len(df_model_per_ts))

    plt.pie(
        df_model_per_ts["count"],
        labels=df_model_per_ts["model_name"],
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops={"edgecolor": "gray"},
    )

    plt.title("Model Distribution")
    plt.show()


def plot_smape_buckets(
    df_accuracy_smape: pd.DataFrame, model_names: List[str], figsize: tuple = (10, 6)
) -> Dict:
    """Plot SMAPES Buckets

    Args:
        df_accuracy_smape (pd.DataFrame): accuracy values of models
        model_names (List[str]): model names
    """
    # Define SMAPE buckets
    bins = [0, 10, 20, 30, 40, float("inf")]
    labels = ["0-10", "10-20", "20-30", "30-40", ">40"]

    buckets_data = generate_smape_err_buckets(
        df_accuracy=df_accuracy_smape, model_names=model_names, bins=bins, labels=labels
    )

    x = np.arange(len(labels))  # x locations for the buckets
    width = 0.8 / len(model_names)  # Adjust width based on number of models

    fig, ax = plt.subplots(figsize=figsize)

    colors = sns.color_palette("Paired", len(buckets_data))
    for i, (model_name, data) in enumerate(buckets_data.items()):
        ax.bar(
            x + i * width - width * len(model_names) / 2,
            data.values,
            width,
            label=model_name,
            color=colors.pop(),
            alpha=0.7,
        )

    # Add labels and title
    ax.set_xlabel("SMAPE Range")
    ax.set_ylabel("Count")
    ax.set_title("SMAPE Buckets Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.show()

    return buckets_data


def generate_smape_err_buckets(
    df_accuracy: pd.DataFrame, model_names: List, bins: List, labels: List
) -> Dict:
    """Generate Error Buckets

    Args:
        df_accuracy (pd.DataFrame): input dataframe with errors
        bins (List): list of cufoff bins
        labels (List): list of labels

    Returns:
        Dict: error buckets
    """

    # Create Buckets
    buckets_data = {}

    df_acc_per_ts_key = (
        df_accuracy.groupby(["ts_key", "model_name"])["smape"]
        .median()
        .to_frame()
        .reset_index()
    )

    for model_name in model_names:
        model_data = pd.cut(
            df_acc_per_ts_key[df_acc_per_ts_key["model_name"] == model_name]["smape"],
            bins=bins,
            labels=labels,
        )
        buckets_data[model_name] = model_data.value_counts().sort_index()

    return buckets_data


def plot_err_less_20_SMAPE(
    buckets_data: dict, figsize: tuple = (8, 6), no_plot: bool = False
) -> None:
    """based on the SMAPE error intervals, plot the number
    of timeseries with an error of less than 20% SMAPE.
    This shows how robust the models are.

    Dict containing the number of timeseries in the
    error intervals per model, e.g:

    'CES': smape
        0-10     26
        10-20    77
        20-30    67
        30-40    41
        >40      55

    Args:
        buckets_data (dict): SMAPE error intervals
    """

    _models = []
    _forecast_less_20 = []
    for model_name, data in buckets_data.items():
        _models.append(model_name)
        _forecast_less_20.append(round(data.head(2).sum() / data.sum(), 2))

    df_acc_less_20 = pd.DataFrame(
        {"model": _models, "err_less_20_perc_ts_key": _forecast_less_20}
    ).sort_values(by="err_less_20_perc_ts_key", ascending=False)

    # Set up the Seaborn bar plot
    if not no_plot:
        plt.figure(figsize=figsize)
        sns.barplot(
            x="model",
            y="err_less_20_perc_ts_key",
            data=df_acc_less_20,
            palette="viridis",
        )

        # Label the plot
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.ylabel("Accuracy (Number of TS Key with SMAPE <20% ")
        plt.title("Model Accuracy with Less Than 20% SMAPE")

        # Show plot
        plt.show()

    return df_acc_less_20


def plot_ratio_vol_prod(ts_key: str, df_ratio: pd.DataFrame, figsize=(6, 10)) -> None:
    """Plot Ratio-Volume-Production values for a given
    timeseries key

    Args:
        ts_key (str): timeseries key
        df_ratio (pd.DataFrame): vol/production ratio dataframe
    """
    # Plot Parameters
    # ts_key = 'Provider_10-Plant_1'
    plant = ts_key.split("-")[1]
    x_axis = "Timestamp"
    vol1_axis = "Actual_Vol_[Tons]"
    vol2_axis = "Expected_Vol_[Tons]"
    prod_axis = "Production"
    ratio_axis = "Vol/Prod_ratio_kg"

    # Plot Variables
    _df = df_ratio.query(f" ts_key == '{ts_key}'")
    x = _df[x_axis]
    vol1 = _df[vol1_axis]
    vol2 = _df[vol2_axis]
    prod = _df[prod_axis]
    ratio = _df[ratio_axis]

    # Create a figure
    plt.rc("text", usetex=False)
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=figsize)

    line_colors = {
        vol1_axis: "#1f76b4",
        vol2_axis: "#ff7e0e",
        prod_axis: "#107824",
        ratio_axis: "#b41f49",
        "background_color": "#f0f0f0",
    }

    # Create Volume Plot
    axs[0].plot(x, vol1, label=vol1_axis, color=line_colors[vol1_axis])
    axs[0].plot(x, vol2, label=vol2_axis, color=line_colors[vol2_axis])
    axs[0].set_title(f"Incoming Volume {ts_key}")
    axs[0].legend()
    axs[0].tick_params(axis="x", rotation=80)
    axs[0].set_facecolor(line_colors["background_color"])
    axs[0].grid(True)

    # Production Plot
    axs[1].plot(x, prod, label=prod_axis, color=line_colors[prod_axis])
    axs[1].set_title(f"Production Level {plant}")
    axs[1].tick_params(axis="x", rotation=80)
    axs[1].legend()
    axs[1].set_facecolor(line_colors["background_color"])
    axs[1].grid(True)

    # Ratio Plot
    axs[2].plot(x, ratio, label=ratio_axis, color=line_colors[ratio_axis])
    axs[2].set_title(f"Volume/Production Ratio {ts_key}")
    axs[2].tick_params(axis="x", rotation=80)
    axs[2].legend()
    axs[2].set_facecolor(line_colors["background_color"])
    axs[2].grid(True)

    # Add labels and adjust layout
    plt.tight_layout()
    plt.show()
    plt.close("all")


def plot_ratio_all_ts(df_ratio: pd.DataFrame, path: str) -> None:
    """Plot Vol/Prod Ratio for all timeseries

    Args:
        df_ratio (pd.DataFrame): vol/production ratio dataframe
    """

    with PdfPages(path) as pdf:
        x_axis = "Timestamp"
        vol1_axis = "Actual_Vol_[Tons]"
        vol2_axis = "Expected_Vol_[Tons]"
        prod_axis = "Production"
        ratio_axis = "Vol/Prod_ratio_kg"
        light_gray = "#f0f0f0"

        line_colors = {
            vol1_axis: "#1f76b4",
            vol2_axis: "#ff7e0e",
            prod_axis: "#107824",
            ratio_axis: "#b41f49",
        }

        for ts_key in df_ratio["ts_key"].unique():  # [:10]:
            _df = df_ratio.query(f" ts_key == '{ts_key}'")
            plant = ts_key.split("-")[1]
            x = _df[x_axis]
            vol1 = _df[vol1_axis]
            vol2 = _df[vol2_axis]
            prod = _df[prod_axis]
            ratio = _df[ratio_axis]

            # Create a figure
            plt.rc("text", usetex=False)
            fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))

            # Create Volume Plot
            axs[0].plot(x, vol1, label=vol1_axis, color=line_colors[vol1_axis])
            axs[0].plot(x, vol2, label=vol2_axis, color=line_colors[vol2_axis])
            axs[0].set_title(f"Incoming Volume {ts_key}")
            axs[0].legend()
            axs[0].tick_params(axis="x", rotation=80)
            axs[0].set_facecolor(light_gray)
            axs[0].grid(True)

            # Production Plot
            axs[1].plot(x, prod, label=prod_axis, color=line_colors[prod_axis])
            axs[1].set_title(f"Production Level {plant}")
            axs[1].tick_params(axis="x", rotation=80)
            axs[1].legend()
            axs[1].set_facecolor(light_gray)
            axs[1].grid(True)

            # Ratio Plot
            axs[2].plot(x, ratio, label=ratio_axis, color=line_colors[ratio_axis])
            axs[2].set_title(f"Volume/Production Ratio {ts_key}")
            axs[2].tick_params(axis="x", rotation=80)
            axs[2].legend()
            axs[2].set_facecolor(light_gray)
            axs[2].grid(True)

            # Add labels and adjust layout
            plt.tight_layout()

            # saves the current figure into a pdf page
            pdf.savefig(fig)
            plt.close(fig)

            del _df

        # We can also set the file's metadata via the PdfPages object:
        d = pdf.infodict()
        d["Title"] = "Timeseries Vol-Prod-Ratio"
        d["Author"] = "John Torres"
        d["CreationDate"] = datetime.today()

        plt.close("all")
