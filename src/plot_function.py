
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Plot resolution type
def plot_res_type(
    data: pd.DataFrame
) -> None:
    
    """
    Plot resolution type.

    Description:
        Plot resolution type distribution.

    Args:
        data (pd.DataFrame): DataFrame of default population. 1 row per 1 default account.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A.
    """

    counts = data["resolution_type"].value_counts(dropna = False)
    labels = counts.index.fillna("Missing")

    plt.figure(figsize = (6, 6))
    plt.pie(
    counts.values,
    labels = labels,
    autopct = "%1.2f%%",
    pctdistance = 0.8,
    labeldistance = 1.1,
    colors = plt.cm.Dark2.colors,
    startangle = 90,
    wedgeprops = {"edgecolor": "white"}
    )
    plt.title("Resolution type distribution")

    return plt.show()

# Plot recovery by resolution type
def plot_recov_type(
    data: pd.DataFrame
) -> None:
    
    """
    Plot recovery by resolution type.

    Description:
        Plot recovery by resolution type by month since default.

    Args:
        data (pd.DataFrame): DataFrame of default cashflow. 1 row per 1 month per 1 default account.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A.
    """

    plot_df = (
        data.groupby(
            ["month_since_default", "resolution_type"],
            as_index = False
        ).agg(amount = ("amount", "mean"))
    )

    plt.figure(figsize = (10, 6))
    for res_type, g in plot_df.groupby("resolution_type"):
        plt.plot(
            g["month_since_default"],
            g["amount"],
            label = f"resolution_type: {res_type}"
        )
    plt.gca().set_yticklabels([f'{i:,.0f}' for i in plt.gca().get_yticks()])
    plt.xlabel("Month since default")
    plt.ylabel("Amount")
    plt.title("Recovery cashflow by resolution type")
    plt.legend()
    plt.tight_layout()

    return plt.show()

# Plot time to resolved
def plot_time_resolved(
    actual: pd.Series,
    predict: pd.Series,
    title: str
) -> None:
    
    """
    Plot time to resolved.

    Description:
        Plot time to resolved for unsolved case.

    Args:
        actual (pd.Series)   : DataFrame of actual time to resolution of unsolved case.
        predict (pd.Series)  : DataFrame of predict time to resolution of unsolved case.
        title (str)          : Name of the plot.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A.
    """

    fig, ax = plt.subplots(figsize = (10, 6))
    ax.hist(actual, bins = 50, alpha = 0.5, label = "Unsolved", density = True)
    ax.hist(predict, bins = 50, alpha = 0.5, label = "Prediction", density = True)
    ax.set_yticklabels([f"{y * 100:.2f}%" for y in ax.get_yticks()])
    ax.set_xlabel("Time to resolution")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(frameon = True, facecolor = 'white', loc = "upper right")
    plt.tight_layout()

    return fig

# Plot resolution type prediction
def plot_pred_res_type(
    predict: np.ndarray,
    type: np.ndarray,
    title: str
) -> None:
    
    """
    Plot predict resolution type.

    Description:
        Plot predict resolution type for unsolved case.

    Args:
        predict (np.ndarray) : Predict probability of each resolution case.
        type (np.ndarray)    : Type of resolution case label.
        title (str)          : Name of the plot.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A.
    """

    fig, ax = plt.subplots(figsize = (10, 6))
    for i, typ in enumerate(type):
        ax.hist(
            predict[:, i], bins = 50, alpha = 0.5,
            label = f"Resolved type {typ}", density = True
        )
    ax.set_xticklabels([f"{x * 100:.2f}%" for x in ax.get_xticks()])
    ax.set_yticklabels([f"{int(y):.0f}" for y in ax.get_yticks()])
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(frameon = True, facecolor = 'white', loc = "upper right")
    plt.tight_layout()

    return fig

# Plot probability of cashflow recieve
def plot_pred_cash_recieve(
    predict: pd.DataFrame,
    title: str
) -> None:
    
    """
    Plot probability of cashflow recieve.

    Description:
        Plot probability of cashflow recieve on training set.

    Args:
        predict (pd.DataFrame)  : DataFrame of summary of mean probability.
        title (str)             : Name of the plot.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A.
    """

    fig, ax = plt.subplots(figsize = (10, 6))
    ax.plot(
        predict["month_mid"],
        predict["mean_proba"],
        marker = "o",
        color = "tab:blue",
    )
    ax.set_yticklabels([f"{y * 100:.2f}%" for y in ax.get_yticks()])
    ax.set_xlabel("Month since default")
    ax.set_ylabel("Mean predicted probability")
    fig.suptitle(title)
    fig.tight_layout()

    return fig

# Plot prediction cashflow recieve amount
def plot_pred_cash_amount(
    predict: pd.DataFrame,
    title: str
) -> None:
    
    """
    Plot prediction cashflow recieve amount.

    Description:
        Plot prediction cashflow recieve amount on training set.

    Args:
        predict (pd.DataFrame)  : DataFrame of summary of mean cashflow recieve amount.
        title (str)             : Name of the plot.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A.
    """

    fig, ax = plt.subplots(figsize = (10, 6))
    for k, g in predict.groupby("model"):
        ax.plot(g["month_since_default"], g["pred"], label = f" Resolution type {k}",)
    ax.set_yticklabels([f"{y * 100:.2f}%" for y in ax.get_yticks()])
    ax.set_xlabel("Month since default")
    ax.set_ylabel("Cashflow amount rate")
    ax.set_title(title)
    ax.legend(frameon = True, facecolor = 'white', loc = "upper right")
    plt.tight_layout()

    return fig
