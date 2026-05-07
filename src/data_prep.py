
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Default account preparation
def prepare_default_accounts(
    df_raw: pd.DataFrame,
    df_cashflows: pd.DataFrame,
    acc_id_col: str,
    default_col: str,
    default_flag: int,
    date_col: str,
    resolution_col: str,
    resolution_date: str,
    last_period: pd.Timestamp
) -> pd.DataFrame:

    """
    Define default population from dataset.

    Description:
        Finding the first default event of population. The resolution types are
        mapped from the latest information available from default population.
        The time to resolution for resolution cases is computed by the different
        between default date and resolution date. However, for censoring cases,
        the time to resolution is computed by the different between default date
        and latest period. The time to resolution is in a monthly basis.

    Args:
        df_raw (pd.DataFrame)       : Input transaction data.
        df_cashflow (pd.DataFrame)  : Input cashflow data.
        acc_id_col (str)            : Primary key.
        default_col (str)           : Default column for modeling.
        default_flag (int)          : Default value for event default identify.
        date_col (str)              : Date column in data.
        resolution_col (str)        : Account status for identify resolution types.
        resolution_date (str)       : Resolution date column in data.
        last_period (pd.Timestamp)  : The latest date in data for modeling.

    Returns:
        pd.DataFrame: DataFrame of default population. 1 row per 1 default account.

    Notes:
        - For on-going collection activities, the resolution date will be based on maximum date found of collection process.
        - For the time to resolution equal to 0, it is capped to 1 since it is better for modeling.
    """

    print("=== Processing ===\n[INFO]: Identify default population")

    df = df_raw.sort_values(by = [acc_id_col, date_col]).copy()

    # Excluded account that defaulted at the first observation date
    # Cannot ensure was it actually defaulted or the data can observe only that date
    mask = (
        df.groupby(acc_id_col).cumcount().eq(0)
        & df[default_col].eq(default_flag)
    )
    df = df[~mask]

    # Find first default per account
    df["default_date"] = df[acc_id_col].map(
        df.loc[df[default_col] == default_flag]
        .groupby(acc_id_col)[date_col]
        .min()
    )
    df = df[df["default_date"].notnull()] #Only default rows

    # Find resolution types
    print("[INFO]: Identify resolution types")

    last_status_map = (
        df.groupby(acc_id_col)[resolution_col]
        .last()
    )
    df["resolution_type"] = df[acc_id_col].map(last_status_map)
    df = df[df["default_date"] == df[date_col]].reset_index(drop = True) #Only first default rows

    # Time to resolution
    # If resolution normal cases --> find delta month between resolution_date and default_date
    # If resolution loss cases --> find delta month from maximum of active and collection resolution_date and default_date
    # If censoring cases --> find delta month between latest period and default_date
    print("[INFO]: Compute time to resolution")

    last_cf = df_cashflows.groupby(acc_id_col)[date_col].max().rename("last_cf_date")
    df = df.join(last_cf, on = acc_id_col)
 
    is_continue = df["resolution_type"].isin([202, 204, 205])
    df[resolution_date] = np.where(
        is_continue,
        df[[resolution_date, "last_cf_date"]].max(axis = 1), #Find maximum period
        df[resolution_date]
    )

    # Resolution cases
    df["resolved"] = df[resolution_date].notna().astype(int)

    df["time_to_resolution"] = np.where(
        df["resolved"] == 1,
        (
            (df[resolution_date].dt.year - df["default_date"].dt.year) * 12 +
            (df[resolution_date].dt.month - df["default_date"].dt.month)
        ),
        (
            (last_period.year - df[date_col].dt.year) * 12 +
            (last_period.month - df[date_col].dt.month)
        )
    )

    # Resolved and censored in the same date
    df["time_to_resolution"] = df["time_to_resolution"].astype(int).clip(lower = 1)

    print(f"[INFO]: Number of defualt: {df.shape[0]}")

    return df

# Building monthly panel
def build_monthly_panel(
    df_accounts: pd.DataFrame,
    df_cashflows: pd.DataFrame,
    acc_id_col: str,
    date_col: str,
    default_date: str,
    resolution_col: str,
    resolution_date: str,
    resolution_type: str,
    last_period: pd.Timestamp
) -> pd.DataFrame:
    
    """
    Building monthly cashflow panel.

    Description:
        To create monthly cashflow of recovery post default event.
        The cashflow data is not consecutively months. By running this
        function, the monthly panel is created for each default account.
        
    Args:
        df_raw (pd.DataFrame)       : Input transaction data.
        df_cashflow (pd.DataFrame)  : Input cashflow data.
        acc_id_col (str)            : Primary key.
        date_col (str)              : Date column in data.
        default_date (str)          : Default date column in data.
        resolution_col (str)        : Account status for identify resolution types.
        resolution_date (str)       : Resolution date column in data.
        resolution_date (str)       : Resolution type column in data.
        last_period (pd.Timestamp)  : The latest date in data for modeling.

    Returns:
        pd.DataFrame: DataFrame of default cashflow. 1 row per 1 month per 1 default account.

    Notes:
        - For at the date of default event, the cashflow is forced to zero.
    """

    print("=== Processing ===\n[INFO]: Panel monthly cashflow for default population")

    # Create all possible cashflow ranges
    months = pd.date_range(
            start = df_cashflows[date_col].min(),
            end = df_cashflows[date_col].max() + pd.offsets.MonthEnd(0),
            freq = "ME",
        )
    months = pd.DataFrame(months, columns = [date_col])
    
    # Prepare account for mapping
    account_map = df_cashflows[[acc_id_col]].drop_duplicates()
    panel = account_map.merge(months, how = "cross")
    
    # Map fully cashflow
    cashflow = (
        panel
        .merge(
            df_cashflows[[acc_id_col, date_col, "amount"]],
            how = "left",
            on = [acc_id_col, date_col]
        )
        .sort_values(by = [acc_id_col, date_col])
    )

    # Map default cashflow
    default_cashflow = pd.merge(
        cashflow,
        df_accounts[[acc_id_col, default_date, resolution_date, resolution_type, resolution_col]],
        how = "left",
        left_on = [acc_id_col],
        right_on = [acc_id_col]
    )
    
    # Select only valid cashflow
    default_cashflow = default_cashflow[
        (
            (default_cashflow[date_col] >= default_cashflow[default_date]) &
            (default_cashflow[date_col] <= default_cashflow[resolution_date])
        )
        |
        (
            (default_cashflow[resolution_type] == 100) &
            (default_cashflow[date_col] >= default_cashflow[default_date]) &
            (default_cashflow[date_col] <= last_period)
        )
    ]

    default_cashflow[[resolution_type, resolution_col]] = default_cashflow[[resolution_type, resolution_col]].astype(int)

    # Month since default (After default)
    default_cashflow["month_since_default"] = (
        (default_cashflow[date_col].dt.year - default_cashflow[default_date].dt.year) * 12 +
        (default_cashflow[date_col].dt.month - default_cashflow[default_date].dt.month)
    )
    
    # Clean recovery amount that first month of default should not have recovery
    default_cashflow["amount"] = np.where(
        default_cashflow["month_since_default"] == 0,
        0,
        default_cashflow["amount"]
    )

    # Resolution month flag only for resolved cases
    default_cashflow["is_resolution_month"] = np.where(
        (default_cashflow[resolution_col] == 1) &
        (default_cashflow[date_col] == default_cashflow[resolution_date]),
        1, 0
    )

    default_cashflow = default_cashflow[
        [
            date_col, acc_id_col, "month_since_default",
            "amount", resolution_type, "is_resolution_month",
            resolution_col
        ]
    ]

    return default_cashflow.reset_index(drop = True)
