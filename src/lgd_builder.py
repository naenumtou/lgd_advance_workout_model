
import numpy as np
import pandas as pd
import warnings

from lifelines import WeibullAFTFitter
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

from src.plot_function import plot_time_resolved, plot_pred_res_type, plot_pred_cash_recieve, plot_pred_cash_amount

warnings.filterwarnings("ignore")

# Helper function
# Training data for default account with FWL
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
   
# Training data for cashflow with FWL
def build_cashflow_fwl_data(
    df_accounts: pd.DataFrame,
    df_cashflow: pd.DataFrame,
    df_mev: pd.DataFrame = None,
    fwl_features: list = None
) -> pd.DataFrame:
    
    """
    Create training set for cashflow hazard model and cashflow amount model.

    Description:
        Using only resolved cases (0, 202, 204, 205) for fitting the models.
        The event target is binary (0, 1). The 0 means the zero that no cashflow
        receive in that months while 1 is means there are some cashflow in that
        month even it is negative due to the cost expense. The base features are
        only EIR and EAD while the FWL Features are using MEV(s) but it is an optional.

    Args:
        df_account (pd.DataFrame)           : Input default data.
        df_cashflow (pd.DataFrame)          : Input cashflow data.
        df_mev (pd.DataFrame, optional)     : Input MEV(s) data.
                                            If None, FWL MEV(s) is not considered.
        fwl_features (list, optional)       : List of MEV(s) that incorrporating into the model.
                                            If None, FWL MEV(s) is not considered.

    Returns:
        pd.DataFrame: Data monthly panel 1 row per month per account.

    Notes:
        - The negative cashflow is referred to cost amount that target 'has_cf' == 1.
    """

    completed_ids = df_accounts.loc[df_accounts["resolved"] == 1, "acc_id"]
    account_feats = df_accounts[["acc_id", "eir", "ead"]].set_index("acc_id")
    panel = df_cashflow[df_cashflow["acc_id"].isin(completed_ids)].copy()
    panel["has_cf"] = (panel["amount"] != 0).astype(int) #Cashflow >, < 0 --> 1, Cashflow = 0 --> 0  
    df_train = panel.join(account_feats, on = "acc_id")

    if fwl_features is None:
        return df_train
    
    else:
        mev_data = df_mev[fwl_features].reset_index(names = "as_of_date")
        df_train = pd.merge(
            df_train,
            mev_data[["as_of_date"] + fwl_features],
            how = "left",
            left_on = ["as_of_date"],
            right_on = ["as_of_date"]
            )

        return df_train
       
# Actual LGD (Resolved cases)
def compute_actual_lgd(
    df_accounts: pd.DataFrame,
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

    completed_ids = df_accounts.loc[df_accounts[resolution_col] == 1, acc_id_col].tolist() #Only resolved cases
    panel = df_cashflow[df_cashflow[acc_id_col].isin(completed_ids)]
    eir = df_accounts.set_index(acc_id_col)[eir_col].to_dict()
    panel[eir_col] = panel[acc_id_col].map(eir)
    
    # PV of recovery
    # Amount already deduct direct cost
    panel["pv"] = panel[amount_col] / (1 + panel[eir_col] / 12) ** panel[month_s_col]
    pv_cashflow = panel.groupby(acc_id_col)["pv"].sum()

    # Mapping back to default account
    df = df_accounts.copy()
    df["pv"] = df[acc_id_col].map(pv_cashflow)

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
    fwl_features: list = None,
    outplot: bool = True
) -> tuple[WeibullAFTFitter, None]:
    
    """
    Fitting the Accelerated Failure Time (AFT) Model for time to resolution.

    Description:
        Using both of resolved cases and unsolved cases for fitting the model.
        The event target is used resolved (0, 1) and duration is time to resolution.
        The base features are only EAD and EIR while the FWL Features are using MEV(s)
        but it is an optional.

    Args:
        df_accounts (pd.DataFrame)          : Input default data.
        df_mev (pd.DataFrame, optional)     : Input MEV(s) data.
                                            If None, FWL MEV(s) is not considered.
        fwl_features (list, optional)       : List of MEV(s) that incorrporating into the model.
                                            If None, FWL MEV(s) is not considered.
        outplot (bool)                      : Option for output plotting.

    Returns:
        callable   : Model callable object from WeibullAFTFitter().
        Figure     : Showing figure from matplotlib.

    Notes:
        - time_to_resolution for non-resolved cases are until latest data period.
        - If outplot = Ture --> output parameters will be 2.
        - If outplot = False --> output parameters will be 1.
    """
    
    df_tmp = df_accounts.copy()
    base_features = ["eir", "ead", "time_to_resolution", "resolved"]
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
   
    if outplot is False:
       return aft
    else:
       unsolved = df_train.loc[df_train["resolved"] == 0, "time_to_resolution"]
       pred = aft.predict_median(df_train[df_train["resolved"] == 0])
       fig = plot_time_resolved(unsolved, pred, "Median time to resolved for unsolved case")
       return aft, fig

# Classification model for resoluation type
def fit_resolution_type_model(
    df_accounts: pd.DataFrame,
    df_mev: pd.DataFrame = None,
    fwl_features: list = None,
    outplot: bool = True
) -> tuple[LogisticRegression, LabelEncoder, None]:
    
    """
    Fitting the multinomial logistic model resolution type.

    Description:
        Using only resolved cases for fitting the multinomial logistic model.
        The event target is used resolution type (0, 202, 204, 205). The base
        features are only time to resolution, EIR and EAD while the FWL Features
        are using MEV(s) but it is an optional.

    Args:
        df_accounts (pd.DataFrame)          : Input default data.
        df_mev (pd.DataFrame, optional)     : Input MEV(s) data.
                                            If None, FWL MEV(s) is not considered.
        fwl_features (list, optional)       : List of MEV(s) that incorrporating into the model.
                                            If None, FWL MEV(s) is not considered.
        outplot (bool)                      : Option for output plotting.

    Returns:
        callable    : Model callable object from LogisticRegression().
        callable    : Encoder callable object from LabelEncoder().
        Figure      : Showing figure from matplotlib.

    Notes:
        - If there are additional base features, the values of thoes cannot missing.
        - The class_weight = "balanced" is for equal probability to predict non-resolved cases.
        - If outplot = Ture --> output parameters will be 3.
        - If outplot = False --> output parameters will be 2.
    """

    # Base features
    base_features = ["eir", "ead", "time_to_resolution"]
    
    df_completed = df_accounts[df_accounts["resolved"] == 1] #Resolved cases
    df_incompleted = df_accounts[df_accounts["resolved"] == 0] #Unsolved cases

    # Get label encoder
    le_type = LabelEncoder()
    y = le_type.fit_transform(df_completed["resolution_type"])
    
    if fwl_features is None:
        df_train = df_completed[base_features]
        df_test = df_incompleted[base_features]
    else:
        df_tmp_completed = build_default_fwl_data(df_completed, df_mev, fwl_features)
        df_tmp_incompleted = build_default_fwl_data(df_incompleted, df_mev, fwl_features)
        df_train = df_tmp_completed[base_features + fwl_features]
        df_test = df_tmp_incompleted[base_features + fwl_features]
        
    # Classification model
    clf = LogisticRegression(
        solver = "lbfgs",
        class_weight = "balanced",
        max_iter = 1000
    )
    clf.fit(df_train, y)
    
    if outplot is False:
        return clf, le_type
    else:
        proba = clf.predict_proba(df_test)
        type = le_type.classes_
        fig = plot_pred_res_type(proba, type, "Predict resolution types for unsolved case")
        return clf, le_type, fig

# Classification model for cashflow recieve
def fit_cf_hazard_model(
    df_accounts: pd.DataFrame,
    df_cashflow: pd.DataFrame,
    df_mev: pd.DataFrame = None,
    fwl_features: list = None,
    outplot: bool = True
) -> tuple[LogisticRegression, None]:

    """
    Fitting the classification model for cashflow recieve.

    Description:
        Using only resolved cases for fitting the classification logistic model.
        The event target is used cashflow recieve (0, 1). The base features are
        month since default (after default), EIR and EAD. For the resolution type,
        the one-hot encoding is used to transform categorical to column based.     
        The FWL Features are using MEV(s) but it is an optional.

    Args:
        df_accounts (pd.DataFrame)          : Input default data.
        df_cashflow (pd.DataFrame)          : Input cashflow data.
        df_mev (pd.DataFrame, optional)     : Input MEV(s) data.
                                            If None, FWL MEV(s) is not considered.
        fwl_features (list, optional)       : List of MEV(s) that incorrporating into the model.
                                            If None, FWL MEV(s) is not considered.
        outplot (bool)                      : Option for output plotting.
        
    Returns:
        callable    : Model callable object from LogisticRegression().
        Figure      : Showing figure from matplotlib.
        
    Notes:
        - If outplot = Ture --> output parameters will be 2.
        - If outplot = False --> output parameters will be 1.
    """
    # Base features
    base_features = ["eir", "ead", "month_since_default"]

    if fwl_features is None:
        df_train = build_cashflow_fwl_data(df_accounts, df_cashflow)
        X = df_train[base_features]
        y_train = df_train["has_cf"]
    else:
        df_train = build_cashflow_fwl_data(df_accounts, df_cashflow, df_mev, fwl_features)
        X = df_train[base_features + fwl_features]
        y_train = df_train["has_cf"]

    # For resolution type features
    dummies = pd.get_dummies(
        df_train["resolution_type"].astype(str),
        prefix = "rtype",
        drop_first = True
    )

    X_train = pd.concat([X, dummies], axis = 1).fillna(0)
    clf = LogisticRegression(max_iter = 1000)
    clf.fit(X_train, y_train)
    clf.feature_names_ = X_train.columns.tolist() #Add features of training set
    
    if outplot is False:
        return clf
    else:
        df_plot = pd.DataFrame(
            {
                "month_since_default": X_train["month_since_default"],
                "actual": y_train,
                "proba": clf.predict_proba(X_train)[:, 1]
            }
        )

        df_plot["month_bin"] = pd.cut(df_plot["month_since_default"], bins = 20)
        summary = (
            df_plot
            .groupby("month_bin", observed = True)
            .agg(
                month_mid = ("month_since_default", "mean"),
                count = ("month_since_default", "size"),
                mean_proba = ("proba", "mean")
            )
            .reset_index(drop = True)
        )
        fig = plot_pred_cash_recieve(summary, "Mean predicted probability cashflow recieve")
        
        return clf, fig

# Regression model for cashflow amount recieve
def fit_cf_amount_models(
    df_accounts: pd.DataFrame,
    df_cashflow: pd.DataFrame,
    df_mev: pd.DataFrame = None,
    fwl_features: list = None,
    outplot: bool = True
) -> dict:
    
    """
    Fitting the regression model for cashflow amount recieve.

    Description:
        Using only resolved cases for fitting the linear regression model.
        The event target is used cashflow amount by EAD (Cashflow rate).
        The base feature is only month since default (after default).
        For the model fitting, it fits as overall pool level and it has been
        seperated by resolution type. The FWL Features are using MEV(s) but
        it is an optional.

    Args:
        df_accounts (pd.DataFrame)          : Input default data.
        df_cashflow (pd.DataFrame)          : Input cashflow data.
        df_mev (pd.DataFrame, optional)     : Input MEV(s) data.
                                            If None, FWL MEV(s) is not considered.
        fwl_features (list, optional)       : List of MEV(s) that incorrporating into the model.
                                            If None, FWL MEV(s) is not considered.
        outplot (bool)                      : Option for output plotting.
        
    Returns:
        Dictionary    : Keys are resolution type. Values are model callable object from smf.ols().
                        {keys: values} --> {resolution type (str): smf.ols() (callable)}
        Figure        : Showing figure from matplotlib.
        
    Notes:
        - If number of sample in particular resolution type less than 30, the model will be fitted (Skip).
        - If outplot = Ture --> output parameters will be 2.
        - If outplot = False --> output parameters will be 1.
    """

    if fwl_features is None:
        data = build_cashflow_fwl_data(df_accounts, df_cashflow)
        formula = f"log_amount_rate ~ log_month_since_default"
    else:
        data = build_cashflow_fwl_data(df_accounts, df_cashflow, df_mev, fwl_features)
        formula = f"log_amount_rate ~ log_month_since_default + " + " + ".join(fwl_features)

    # Training data
    data = data[data["has_cf"] == 1] #Only cashflow recieve
    data = data[data["month_since_default"] > 0] #Using log transform --> 0 cannot use
    data["log_amount_rate"] = np.log(data["amount"] / data["ead"] + 1e-6) #Fit model on log scale
    data["log_month_since_default"] = np.log(data["month_since_default"]) #Fit model on log scale

    models = {}

    for rtype in data["resolution_type"].unique():
        mask = data["resolution_type"] == rtype
        if mask.sum() < 30: #Minimum sample
            continue
        models[rtype] = smf.ols(formula, data = data[mask]).fit()
    
    if outplot is False:
        return models
    else:
        results = []
        for key, model in models.items():
            msd = np.exp(model.model.exog[:, 1])
            pred = np.exp(model.predict())
            df_tmp = pd.DataFrame(
                {
                    "model": key,
                    "month_since_default": msd,
                    "pred": pred
                }
            )
            results.append(df_tmp)

        df_plot = pd.concat(results, ignore_index = True)
        fig = plot_pred_cash_amount(df_plot, "Predicted cashflow recieve amount rate")

        return models, fig

# Data for default unsolved account with FWL
def build_unsolved_default_fwl_data(
    df_accounts: pd.DataFrame,
    aft: WeibullAFTFitter,
    clf_type: LogisticRegression,
    le_type: LabelEncoder,
    cf_hazard_model: LogisticRegression,
    cf_amount_models: dict,
    df_mev: pd.DataFrame = None,
    fwl_features: list = None,
) -> pd.DataFrame:
    
    """
    Building the data for unsolved cases.

    Description:
        Features preparation for unsolved cases to fit the pre-trained model.
        The data table contains model features that co-responding to the model.

    Args:
        df_accounts (pd.DataFrame)              : Input default data.
        aft (WeibullAFTFitter)                  : Time to resolved pre-trained model.
        clf_type (LogisticRegression)           : Resolution type pre-trained model.
        le_type (LabelEncoder)                  : Label encoder for target in resolution type pre-trained model.
        cf_hazard_model (LogisticRegression)    : Cashflow recieve pre-trained model.
        cf_amount_models (dict)                 : Dictionary cashflow amount model. Keys are resolution type.
                                                Values are model callable object from smf.ols().
                                                {keys: values} --> {resolution type (str): smf.ols() (callable)}
        df_mev (pd.DataFrame, optional)         : Input MEV(s) data.
                                                If None, FWL MEV(s) is not considered.
        fwl_features (list, optional)           : List of MEV(s) that incorrporating into the model.
                                                If None, FWL MEV(s) is not considered.

    Returns:
        pd.DataFrame: Data cashflow table of unsolved cases. The table contains expect time to resolved,
                    expect resolution type, probability of cashflow recieve, predicted cashflow amount rate
                    per resolution type.

    Notes:
        - N/A.
    """
    
    # Features of pre-trained model for prediction
    aft_idx = ["eir", "ead", "time_to_resolution", "resolved"] #AFT Model features
    clf_idx = ["eir", "ead", "time_to_resolution"] #CLF Model features 
    cf_haz_idx = ["eir", "ead", "month_since_default"] #Cashflow hazard model features
    rtype_cols = [c for c in cf_hazard_model.feature_names_ if c.startswith("rtype")] #Cashflow hazard model resolution features
    cf_amt_idx = ["month_since_default"] #Cashflow amount model features

    # Unsolved cases (Account level)
    df_unsolved = df_accounts[df_accounts["resolved"] == 0]
    
    if fwl_features is None:
        # Predict time to resolution
        df_unsolved["exp_time_to_resolution"] = aft.predict_median(df_unsolved[aft_idx]).astype(int)

        # Predict type of resolution (prob)
        type_proba_df = pd.DataFrame(
            clf_type.predict_proba(df_unsolved[clf_idx]),
            columns = le_type.classes_,
            index = df_unsolved["acc_id"]
        ).reset_index()

    else:
        # Mapping with MEV(s)
        df_unsolved = build_default_fwl_data(df_unsolved, df_mev, fwl_features)
        
        # Predict time to resolution
        df_unsolved["exp_time_to_resolution"] = aft.predict_median(df_unsolved[aft_idx + fwl_features]).astype(int)
        
        # Predict type of resolution (prob)
        type_proba_df = pd.DataFrame(
            clf_type.predict_proba(df_unsolved[clf_idx + fwl_features]),
            columns = le_type.classes_,
            index = df_unsolved["acc_id"]
        ).reset_index()
        
    # Create unsolved cases cashflow based on expected time to resolve (Monthly level)
    n_rep = df_unsolved["exp_time_to_resolution"] + 1 #For repeat rows
       
    if fwl_features is None:
        df_cf_haz = df_unsolved.loc[
            df_unsolved.index.repeat(n_rep)
        ].copy()
        df_cf_haz["month_since_default"] = df_cf_haz.groupby(level = 0).cumcount() #Start at 0

        # Create datetime range equal to "month_since_default"
        df_cf_haz["as_of_date"] = (
            pd.to_datetime(df_cf_haz["default_date"])
            + df_cf_haz["month_since_default"].apply(lambda x: pd.DateOffset(months = x))
            + pd.offsets.MonthEnd(0)
        )
        df_cf_haz["as_of_date"] = pd.to_datetime(df_cf_haz["as_of_date"])

        df_cf_haz = df_cf_haz.reset_index(drop = True)
        df_cf_haz[rtype_cols] = False #Pre-trained only resolved cases --> unsolved cases will be False
        
        # Predict probability of cashflow recieve
        df_cf_haz["p_cf"] = cf_hazard_model.predict_proba(df_cf_haz[cf_haz_idx + rtype_cols])[:, 1]
        
        # Predict cashflow amount recieve rate by resolution type model
        for tpy, model in cf_amount_models.items():
            df_cf_haz[f"{tpy}_cf_rate"] = model.predict(df_cf_haz[cf_amt_idx])
   
    else:
        # Drop mapped MEV(s)
        df_cf_haz = df_unsolved.drop(fwl_features, axis = 1).loc[
            df_unsolved.index.repeat(n_rep)
        ].copy()
        df_cf_haz["month_since_default"] = df_cf_haz.groupby(level = 0).cumcount() #Start at 0 

        # Create datetime range equal to "month_since_default" for MEV(s) mapping
        df_cf_haz["as_of_date"] = (
            pd.to_datetime(df_cf_haz["default_date"])
            + df_cf_haz["month_since_default"].apply(lambda x: pd.DateOffset(months = x))
            + pd.offsets.MonthEnd(0)
        )
        df_cf_haz["as_of_date"] = pd.to_datetime(df_cf_haz["as_of_date"])
           
        # Mapping with MEV(s)
        mev_data = df_mev[fwl_features].reset_index(names = "as_of_date")
        df_cf_haz = pd.merge(
            df_cf_haz,
            mev_data[["as_of_date"] + fwl_features],
            how = "left",
            left_on = ["as_of_date"],
            right_on = ["as_of_date"]
            ).reset_index(drop = True)
        df_cf_haz[rtype_cols] = False #Pre-trained only resolved cases --> unsolved cases will be False
        
        # Predict probability of cashflow recieve
        df_cf_haz["p_cf"] = cf_hazard_model.predict_proba(df_cf_haz[cf_haz_idx + fwl_features + rtype_cols])[:, 1]
        
        # Predict cashflow amount recieve rate by resolution type model
        for tpy, model in cf_amount_models.items():
            df_cf_haz[f"{tpy}_cf_rate"] = model.predict(df_cf_haz[cf_amt_idx + fwl_features])
        
    return df_cf_haz, type_proba_df
