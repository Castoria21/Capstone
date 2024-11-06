"""
This script is used for generating DataFrames, with values of features, columns of cusip and index of time.
"""

import pandas as pd
import numpy as np
from utils import *


ffill_max = 12


def get_momentum(
        df_ret: pd.DataFrame, window: int=12, periods_skipped: int=1
) -> pd.DataFrame:
    """
    Get price momentum.

    Args:
        df_ret: a DataFrame with values of simple returns, columns of cusip and index of time.
        window: the number of most recent months to calculate momentum.
        periods_skipped: the number of most recent months skipped in calculation.

    Returns:
        a DataFrame with values of price momentum, columns of cusip and index of time.
    """
    if periods_skipped == 0:
        return (1 + df_ret).rolling(window=window).apply(np.prod)
    return (1 + df_ret).rolling(window=window).apply(np.prod) / (1 + df_ret).rolling(window=periods_skipped).apply(np.prod)


def get_size_df(df_crsp, holding_period):
    """
    Get market value as size.
    """
    df_size = df_crsp.copy(deep=True)
    df_size['size'] = df_size.eval("prc*shrout")
    return forward_fill(date_cusip_pivot(df_size, "size"), holding_period)


def get_eps_df(df_comp_fundq, df_ret, holding_period):
    """
    Get earnings per share.
    """
    df_eps = df_comp_fundq["date cusip epspxq consol".split()]
    df_eps['eps'] = df_eps.eval("epspxq")
    return get_formatted_feature(df_eps, df_ret, "eps", holding_period)


def get_pm_df(df_comp_fundq, df_ret, holding_period):
    """
    Get profit margin.
    """
    df_pm = df_comp_fundq["date cusip niq saleq consol".split()]
    df_pm['pm'] = df_pm.eval("niq/saleq")
    return get_formatted_feature(df_pm, df_ret, "pm", holding_period)