
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
