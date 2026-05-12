
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