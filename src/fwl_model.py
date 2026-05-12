
import pandas as pd
import numpy as np
from scipy import stats

from scipy.stats import rankdata, chi2

# Helper function
def _mapping_mev(
    df_ori: pd.DataFrame,
    df_mev: pd.DataFrame,
    key_map: str
) -> pd.DataFrame:

    """
    Data mapping with MEV(s).

    Description:
        Mapping MEV(s) to the default account and cashflow data.

    Args:
        df_ori (pd.DataFrame)     : Input default data or cashflow data.
        df_mev (pd.DataFrame)     : Input MEV(s) data.
        key_map (str)             : Date key for mapping.
                                  If data is default date, key mapping is not "default_date".
                                  If data is cashflow date, key mapping is not "as_of_date".

    Returns:
        pd.DataFrame: Data with mapped MEV(s).

    Notes:
        - N/A.
    """

    df = (
        df_ori
        .set_index(key_map)
        .join(df_mev, how = "left")
    )

    return df

# Spearman rank correlation
def spearman_rank_corr(
    df_accounts: pd.DataFrame,
    df_cashflow: pd.DataFrame,
    df_mev: pd.DataFrame,
    target_col: str
) -> pd.DataFrame:
    
    """
    Calculates a Spearman rank-order correlation coefficient and p-value.

    Description:
        Spearman's correlation assesses monotonic relationships (whether
        linear or not). It operates on the ranks of the data rather than raw
        values, making it robust to outliers and suitable for non-normally
        distributed or ordinal data.

    Args:
        df_accounts (pd.DataFrame)  : Input default data.
        df_cashflow (pd.DataFrame)  : Input cashflow data.
        df_mev (pd.DataFrame)       : Input MEV(s) data.
        target_col (str)            : Target for calculation.

    Returns:
        pd.DataFrame: Spearman rank-order correlation coefficient and p-value.

    Notes:
        - If target_col is "time_to_resolution" default account level is calculated.
        - If target_col is "log_amount_rate" cashflow account level is calculated.
    """

    if target_col == "time_to_resolution":
        df = _mapping_mev(df_accounts, df_mev, "default_date")
        y = df[target_col].astype(float)
        n = y.shape[0]
        
    else:
        # Cashflow data preparation
        completed_ids = df_accounts.loc[df_accounts["resolved"] == 1, "acc_id"]
        account_feats = df_accounts[["acc_id", "ead"]].set_index("acc_id")
    
        df = _mapping_mev(df_cashflow, df_mev, "as_of_date")
        df = df[df["acc_id"].isin(completed_ids)]
        df = df.join(account_feats, on = "acc_id")
        df = df[df["amount"] > 0]  #Only cashflow recieve
        df[target_col] = np.log(df["amount"] / df["ead"] + 1e-6) #Fit model on log scale
        y = df[target_col].astype(float)
        n = y.shape[0]
    
    X = df.loc[:, df.columns.isin(df_mev.columns)]

    # Average rank
    yr = stats.rankdata(y)
    Xr = np.apply_along_axis(stats.rankdata, 0, X)

    # Center ranks
    y0 = yr - yr.mean()
    X0 = Xr - Xr.mean(axis = 0)

    # Spearman rank correlation
    r = (y0 @ X0) / np.sqrt((y0 @ y0) * (X0 ** 2).sum(axis = 0))

    # p-values
    df = n - 2
    tstat = r * np.sqrt(df / (1 - r**2))
    p = 2 * stats.t.sf(np.abs(tstat), df)
    spre_rnk_output = pd.DataFrame(
        {
            f"{target_col}_stat": r,
            f"{target_col}_p_value": p 
        },
        index = df_mev.columns
    )

    return spre_rnk_output

# Point-biserial correlation
def point_biserial_corr(
    df_accounts: pd.DataFrame,
    df_cashflow: pd.DataFrame,
    df_mev: pd.DataFrame,
    target_col: str
) -> pd.DataFrame:
    

    """
    Calculates a point-biserial correlation coefficient and p-value.

    Description:
        This is used to measure the relationship between a binary (dichotomous)
        variable and a continuous variable. It is mathematically equivalent to
        Pearson's correlation where one variable is coded as 0 or 1.

    Args:
        df_accounts (pd.DataFrame)  : Input default data.
        df_cashflow (pd.DataFrame)  : Input cashflow data.
        df_mev (pd.DataFrame)       : Input MEV(s) data.
        target_col (str)            : Target for calculation.

    Returns:
        pd.DataFrame: Point-biserial correlation coefficient and p-value.

    Notes:
        - N/A.
    """

    # Cashflow data preparation
    completed_ids = df_accounts.loc[df_accounts["resolved"] == 1, "acc_id"]

    df = _mapping_mev(df_cashflow, df_mev, "as_of_date")
    df = df[df["acc_id"].isin(completed_ids)]
    df[target_col] = (df["amount"] != 0).astype(int) #Cashflow >, < 0 --> 1, Cashflow = 0 --> 0
    y = df[target_col].astype(float)
    n = y.shape[0]
    X = df.loc[:, df.columns.isin(df_mev.columns)]

    # Center
    y0 = y - y.mean()
    X0 = X - X.mean(axis = 0)

    # Point-biserial correlation
    r = (y0 @ X0) / np.sqrt((y0 @ y0) * (X0 ** 2).sum(axis = 0))

    # p-values
    df = n - 2
    t = r * np.sqrt(df / (1 - r**2))
    p = 2 * stats.t.sf(np.abs(t), df)
    point_biserial_output = pd.DataFrame(
        {
            f"{target_col}_stat": r,
            f"{target_col}_p_value": p 
        },
        index = df_mev.columns
    )

    return point_biserial_output

# Kruskal
def kruskal_algor(
    df_accounts: pd.DataFrame,
    df_mev: pd.DataFrame,
    target_col: str
) -> pd.DataFrame:
    
    """
    Performs the Kruskal-Wallis H-test for independent samples.

    Description:
        This non-parametric method tests whether the median of two or more
        groups are significantly different. It is an alternative to One-Way
        ANOVA used when the data does not follow a normal distribution or
        is ordinal in nature.

    Args:
        df_mev (pd.DataFrame)       : Input MEV(s) data.
        target_col (str)            : Target for calculation.

    Returns:
        pd.DataFrame: Kruskal-Wallis H-test for independent samples.

    Notes:
        - N/A.
    """
    
    df = _mapping_mev(df_accounts, df_mev, "default_date")
    df = df[df["resolved"] == 1] #Resolved only

    g = np.asarray(df[target_col]) #g --> (n,) group labels
    X = np.asarray(df.loc[:, df.columns.isin(df_mev.columns)]) #X --> (n, p) features
    n, p = X.shape

    # Encode groups
    groups, g_idx = np.unique(g, return_inverse = True)
    k = len(groups)


    Xr = np.apply_along_axis(stats.rankdata, 0, X) #Rank X in column wise
    n_g = np.bincount(g_idx) #Group sizes

    # Sum of ranks per group (k × p)
    rank_sum = np.zeros((k, p))
    for i in range(k):
        rank_sum[i] = Xr[g_idx == i].sum(axis = 0)

    # Kruskal–Wallis H (vectorized)
    H = (
        12 / (n * (n + 1))
        * np.sum((rank_sum ** 2) / n_g[:, None], axis = 0)
        - 3 * (n + 1)
    )

    # Tie correction
    ties = np.apply_along_axis(
        lambda v: np.sum(np.bincount(v.astype(int))[2:] ** 3 -
                        np.bincount(v.astype(int))[2:]),
        0,
        Xr
    )

    C = 1 - ties / (n ** 3 - n)
    H_corrected = H / C
    eta2 = (H_corrected - k + 1) / (n - k)

    # p-values
    p = stats.chi2.sf(H, df = k - 1)
    kruskal_output = pd.DataFrame(
        {
            f"{target_col}_stat": eta2,
            f"{target_col}_p_value": p 
        },
        index = df_mev.columns
    )

    return kruskal_output
