
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Helper function
def build_default_fwl_data(
   df_accounts: pd.DataFrame,
   df_mev: pd.DataFrame,
   fwl_features: list
) -> pd.DataFrame:
    
    """
    Create training set for default account with MEV(s) Data.

    Description:
        Mapping MEV(s) Data at the date of default for FWL Consideration.

    Args:
        df_accounts (pd.DataFrame)   : Input default data.
        df_mev (pd.DataFrame)        : Input MEV(s) data.
        fwl_features (list)          : List of MEV(s) that incorrporating into the model. 

    Returns:
        pd.DataFrame: Data default account with MEV(s) mapped.

    Notes:
        - N/A.
    """
    
    mev_data = df_mev[fwl_features].reset_index(names = "default_date")
    df_train = pd.merge(
        df_accounts,
        mev_data[["default_date"] + fwl_features],
        how = "left",
        left_on = ["default_date"],
        right_on = ["default_date"]
        )
    
    return df_train

# Actual LGD (Resolved cases)
def compute_actual_lgd(
    df_account: pd.DataFrame,
    df_cashflow: pd.DataFrame,
    acc_id_col: str,
    resolution_col: str,
    eir_col: str,
    amount_col: str,
    month_s_col: str,
    ead_col: str
) -> pd.DataFrame:

    """
    Compute actual LGD for resolved cases.

    Description:
        The resolved cases can be computed the actual since the recovery process is done.
        The IFRS 9 is required to deduct to direct cost out from the actual recovery amount
        and the net recovery cashflow is discounted to present value (PV) by EIR.

        The LGD is computed by 1 - summation of PV(Recovery) divided EAD.
        PV = Σ CF_t / (1 + EIR/12)^t
        LGD = 1 - PV / EAD

    Args:
        df_account (pd.DataFrame)   : Input default data.
        df_cashflow (pd.DataFrame)  : Input cashflow data.
        acc_id_col (str)            : Primary key.
        resolution_col (str)        : Account status for identify resolution types.
        eir_col (str)               : EIR column for discount factor. (Percentage value in annual basis)
        amount_col (str)            : Cashflow amount column
        month_s_col (str)           : Month since default column for discount times
        ead_col (str)               : Exposure at default

    Returns:
        pd.DataFrame: DataFrame of default population. 1 row per 1 default account.

    Notes:
        - For resolved accounts that have no recovery, the 100% LGD will be assigned.
    """

    print("=== Processing ===\n[INFO]: Actual LGD for resolved cases")

    completed_ids = df_account.loc[df_account[resolution_col] == 1, acc_id_col].tolist() #Only resolved cases
    panel = df_cashflow[df_cashflow[acc_id_col].isin(completed_ids)]
    eir = df_account.set_index(acc_id_col)[eir_col].to_dict()
    panel[eir_col] = panel[acc_id_col].map(eir)
    
    # PV of recovery
    # Amount already deduct direct cost
    panel["pv"] = panel[amount_col] / (1 + panel[eir_col] / 12) ** panel[month_s_col]
    pv_cahflow = panel.groupby(acc_id_col)["pv"].sum()

    # Mapping back to default account
    df = df_account.copy()
    df["pv"] = df[acc_id_col].map(pv_cahflow)

    # Compute actual LGD by discounting cashflow
    df["lgd_actual"] = np.clip(
        1 - df["pv"] / df[ead_col],
        0.0, 1.0
    )

    # For resolved accounts but do not have cashflow --> LGD = 100%
    df["lgd_actual"] = np.where(
        (df[resolution_col] == 1) & (df["lgd_actual"].isnull()),
        1.0,
        df["lgd_actual"]
    )

    # Average (Show result)
    df_avg = df[df[resolution_col] == 1].groupby(
    "resolution_type", as_index = False
    ).apply(
        lambda x: np.average(
            x["lgd_actual"], weights = x[ead_col]
        )
    ).rename(columns = {None: 'lgd'})
    
    print("=== Result ===")
    for _, row in df_avg.iterrows():
        print(f"    resolution_type {int(row['resolution_type']):>3} - LGD: {row['lgd'] * 100:.2f}%")

    return df.drop("pv", axis = 1)

# Survival model for time to resolution
def fit_survival_model(
    df_accounts: pd.DataFrame,
    df_mev: pd.DataFrame = None,
    fwl_features: list = None
) -> WeibullAFTFitter:
    
    """
    Fitting the Accelerated Failure Time (AFT) Model for time to resolution.

    Description:
        Using both of resolved cases and non-resolved cases for fitting the model.
        The event target is used resolved (0, 1) and duration is time to resolution.
        The base features are only resolved and time to resolution while the FWL Features
        are using MEV(s) but it is an optional.

    Args:
        df_accounts (pd.DataFrame)   : Input default data.
        df_mev (pd.DataFrame)        : Input MEV(s) data.
        fwl_features (list)          : List of MEV(s) that incorrporating into the model. 

    Returns:
        callable: Model callable object from WeibullAFTFitter().

    Notes:
        - time_to_resolution for non-resolved cases are until latest data period.
    """
    
    df_tmp = df_accounts.copy()
    base_features = ["time_to_resolution", "resolved"]
    if fwl_features is None:
        df_train = df_tmp[base_features]
    else:
        df_tmp = build_default_fwl_data(df_accounts, df_mev, fwl_features)
        df_train = df_tmp[base_features + fwl_features]
    
    # AFT Model
    aft = WeibullAFTFitter()
    aft.fit(
        df_train,
        duration_col = "time_to_resolution",
        event_col = "resolved",
    )

    return aft

# Classification model for resoluation type
def fit_resolution_type_model(
    df_accounts: pd.DataFrame,
    df_mev: pd.DataFrame = None,
    fwl_features: list = None
) -> tuple[LogisticRegression, LabelEncoder]:
    
    """
    Fitting the multinomial logistic model resolution type.

    Description:
        Using only resolved cases for fitting the multinomial logistic model.
        The event target is used resolution type (0, 202, 204, 205). The base
        features are only time to resolution, EIR and EAD while the FWL Features
        are using MEV(s) but it is an optional.

    Args:
        df_accounts (pd.DataFrame)  : Input default data.
        df_mev (pd.DataFrame)       : Input MEV(s) data.
        fwl_features (list)         : List of MEV(s) that incorrporating into the model. 

    Returns:
        callable: Model callable object from LogisticRegression().
        callable: Encoder callable object from LabelEncoder().

    Notes:
        - If there are additional base features, the values of thoes cannot missing.
        - The class_weight = "balanced" is for equal probability to predict non-resolved cases.
    """
    
    # Only resolved cases
    df_completed = df_accounts[df_accounts['resolved'] == 1]

    # Get label encoder
    le_type = LabelEncoder()
    y = le_type.fit_transform(df_completed["resolution_type"])
    
    base_features = ["time_to_resolution", "eir", "ead"]
    
    if fwl_features is None:
        df_train = df_completed[base_features]
    else:
        df_tmp = build_default_fwl_data(df_completed, df_mev, fwl_features)
        df_train = df_tmp[base_features + fwl_features]
        
    # Classification model
    clf = LogisticRegression(
        solver = "lbfgs",
        class_weight = "balanced",
        max_iter = 1000
    )
    clf.fit(df_train, y)

    return clf, le_type
