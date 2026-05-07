
import pandas as pd
import numpy as np
import re
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
