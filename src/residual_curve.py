
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Residual cashflow
def residual_cashflow(
    df_accounts: pd.DataFrame,
    df_cashflow: pd.DataFrame,
    df_cf_incomplete: pd.DataFrame,
    type_proba_df: pd.DataFrame
) -> pd.DataFrame:

    """
    Compute residual cashflow.

    Description:
        Compute residual cashflow combine both of solved and unsolved cases.
        For solved cases, using the same logic calculation from compute_actual_lgd()
        and for unsolved cases, using the same logic calculation from
        build_unsolved_default_fwl_data(). These are because it needs to minimise
        the difference between unbias/FWL LGD and residual LGD.

    Args:
        df_accounts (pd.DataFrame)          : Input default data.
        df_cashflow (pd.DataFrame)          : Input actual cashflow data.
        df_cf_incomplete (pd.DataFrame)     : Input estimated cashflow data.
        type_proba_df (pd.DataFrame)        : Input probability of resolution types.

    Returns:
        pd.DataFrame: Data with final cashflow combined both of actual from resolved and estimated from unsolved.

    Notes:
        - The actual cashflow recieve amount is leveraged as much as possible from actual data.
        - The estimated resolution type is using the highest probability from resolution types model.
    """

    # Incompleted cashflow calculation
    df = df_cf_incomplete.copy()
    incomplete_ids = df["acc_id"].unique().tolist()
    resolved_type = type_proba_df.columns[1:]

    # PV of actual recovery
    # Actual recovery where Month since default at 0 is already flagged as 0 cashflow
    actual_cashflow = df_cashflow[df_cashflow["acc_id"].isin(incomplete_ids)]
    eir = df_accounts.set_index("acc_id")["eir"].to_dict()
    actual_cashflow["eir"] = actual_cashflow["acc_id"].map(eir)
    actual_cashflow["pv"] = actual_cashflow["amount"] / (1 + actual_cashflow["eir"] / 12) ** actual_cashflow["month_since_default"]

    # PV of estimate recovery
    # Month since default at 0 --> do not calculate recovery since it is default date
    df["p_cf"] = np.where(
        df["month_since_default"] == 0,
        0,
        df["p_cf"]
    )
    
    for tpy in resolved_type:
        df[f"pv_{tpy}"] = (df["p_cf"] * df["ead"] * df[f"{tpy}_cf_rate"]) / (1 + df["eir"] / 12) ** df["month_since_default"]

    # Using actual recovery with model to avoid underestimate LGD
    df = pd.merge(
        df,
        actual_cashflow[["acc_id", "month_since_default", "pv"]],
        how = "left",
        left_on = ["acc_id", "month_since_default"],
        right_on = ["acc_id", "month_since_default"]
    )
    
    # If there are actual recovery --> using actual recovery. If not --> using estimate recovery
    mask = df["pv"].notna() & (df["pv"] != 0)
    for tpy in resolved_type:
        df[f"pv_{tpy}"] = np.where(
            mask,
            df["pv"],
            df[f"pv_{tpy}"]
        )

    # Probability of cashflow weighted
    # Mapping resolution type
    df = pd.merge(
        df,
        type_proba_df,
        how = "left",
        left_on = ["acc_id"],
        right_on = ["acc_id"]
    )
    df['pv_amount'] = sum(df[f'pv_{c}'] * df[c] for c in resolved_type)

    # Find resolution type for unsolved cases
    unsolved_type = type_proba_df.copy()
    unsolved_type["est_res_type"] = (
        unsolved_type[resolved_type].idxmax(axis = 1)
    )
    unsolved_type = unsolved_type.set_index("acc_id")["est_res_type"]
    df["res_type_final"] = df["acc_id"].map(unsolved_type)

    # Completed cashflow calculation
    completed_ids = df_accounts.loc[df_accounts["resolved"] == 1, "acc_id"].tolist() #Only resolved cases
    panel = df_cashflow[df_cashflow["acc_id"].isin(completed_ids)]
    account_feats = df_accounts[["acc_id", "eir", "ead"]].set_index("acc_id")
    panel = panel.join(account_feats, on = "acc_id").rename(
        columns = {
            "resolution_type": "res_type_final"
        }
    )

    # Discount cashflow
    panel["pv_amount"] = panel["amount"] / (1 + panel["eir"] / 12) ** panel["month_since_default"]

    # Combine table
    keep_cols = ["acc_id", "month_since_default", "ead", "eir", "pv_amount", "res_type_final"]
    df_comb = pd.concat(
        [df[keep_cols], panel[keep_cols]],
        axis = 0,
        ignore_index = True
    )

    # Mapping default status
    acc_status = df_accounts.set_index("acc_id")["acc_status"].to_dict()
    df_comb["acc_status"] = df_comb["acc_id"].map(acc_status)

    return df_comb

# Residual LGD
def residual_lgd(
    df_res_cashflow: pd.DataFrame
) -> pd.DataFrame:
    
    """
    Compute residual LGD.

    Description:
        Residual LGD refers to the expected loss for customers who are already classified in Stage 3.
        The main distinction between LGD for Stage 1 and Stage 2 and Residual LGD for Stage 3 is as follows:
            - LGD for Stage 1 and Stage 2 applies to performing customers and estimates losses based on
              the total expected recovery after a potential default.
            - Residual LGD for Stage 3 focuses only on the remaining recoverable amount. It is influenced
              by how long a customer has been in Stage 3. The longer the customer stays in Stage 3,
              the lower the expected future recovery, resulting in a higher LGD.
        
    Args:
        df_res_cashflow (pd.DataFrame): Input final cashflow combined both of actual from resolved
                                        and estimated from unsolved.

    Returns:
        pd.DataFrame: Residual LGD table by default status and resolution type.

    Notes:
        - The input final cashflow is re-built by the equal range of month since default.
    """

    # Create equal range before aggregate to avoid negatively increase
    # Create all possible cashflow ranges
    months = pd.DataFrame(
        {
            "month_since_default": range(
                df_res_cashflow["month_since_default"].min(),
                df_res_cashflow["month_since_default"].max() + 1
       
            )
        }
    )
    
    # Prepare account for mapping
    account_map = df_res_cashflow.drop(["pv_amount", "month_since_default"], axis = 1).drop_duplicates()
    panel = account_map.merge(months, how = "cross")

    # Map fully cashflow
    cashflow = (
        panel
        .merge(
            df_res_cashflow[["acc_id", "month_since_default", "pv_amount"]],
            how = "left",
            on = ["acc_id", "month_since_default"]
        )
        .sort_values(by = ["acc_id", "month_since_default"])
    )
    cashflow["cum_pv_amount"] = (
        cashflow["pv_amount"].fillna(0)
        .groupby(cashflow["acc_id"])
        .cumsum()
    )
    cashflow["recovery_to_ead"] = cashflow["cum_pv_amount"] / cashflow["ead"]
    
    # Aggregated by default status, resolution type and month since default
    df = (
        cashflow
        .assign(w = cashflow["recovery_to_ead"] * cashflow["ead"])
        .groupby(["acc_status", "res_type_final", "month_since_default"], as_index = False)
        .agg(
            ead = ("ead", "sum"),
            w = ("w", "sum")
        )
    )

    # Average of %recovery
    df["recovery_to_ead"] = df["w"] / df["ead"]

    # Remaining EAD
    df["remaining_ead"] = 1 - df["recovery_to_ead"]

    # Ultimate recovery
    df["ultimate_recovery"] = (
        df
        .groupby(["acc_status", "res_type_final"])["recovery_to_ead"]
        .transform("max")
    )

    # Expected recovery
    df["expected_recovery"] = df["ultimate_recovery"] = df["recovery_to_ead"]

    # Residual LGD
    df["residual LGD"] = 1 - df["expected_recovery"] / df["remaining_ead"]

    return df
