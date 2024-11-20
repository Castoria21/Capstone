"""
This script is used for generating DataFrames, with values of features, columns of cusip and index of time.
"""

import pandas as pd
import numpy as np
from .utils import *

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


def get_size_df(df_crsp: pd.DataFrame, holding_period: int) -> pd.DataFrame:
    """
    Get market value as size.

    Args:
        df_crsp: the DataFrame of CRSP raw data.
        holding_period: how long does the characteristic last.
    
    Returns:
        a DataFrame with values of size, columns of cusip and index of time.
    """
    df_size = df_crsp.copy(deep=True)
    df_size['size'] = df_size.eval("prc*shrout")
    return forward_fill(date_cusip_pivot(df_size, "size"), holding_period)


def get_eps_df(
    df_comp_fundq: pd.DataFrame, df_ret: pd.DataFrame, holding_period: int
) -> pd.DataFrame:
    """
    Get earnings per share.

    Args:
        df_comp_fundq: the DataFrame of compustat quarterly raw data.
    
    Returns:
        a DataFrame with values of eps, columns of cusip and index of time.
    """
    df_eps = df_comp_fundq["date cusip epspxq consol".split()]
    df_eps['eps'] = df_eps.eval("epspxq")
    return get_formatted_feature(df_eps, df_ret, "eps", holding_period)


def get_pm_df(
    df_comp_fundq: pd.DataFrame, df_ret: pd.DataFrame, holding_period: pd.DataFrame
) -> pd.DataFrame:
    """
    Get profit margin = net income quarterly / sales quarterly.
    """
    df_pm = df_comp_fundq["date cusip niq saleq consol".split()]
    df_pm['pm'] = df_pm.eval("niq/saleq")
    return get_formatted_feature(df_pm, df_ret, "pm", holding_period)


def get_dvx_df(
    df_comp_fundq: pd.DataFrame, df_ret: pd.DataFrame, holding_period: pd.DataFrame
) -> pd.DataFrame:
    """
    Get divident yield = dividend per share excluding extraordinary quarterly / price close quarterly.
    """
    df_pm = df_comp_fundq["date cusip dvpsxq prccq consol".split()]
    df_pm['dvpsxq'] = df_pm.eval("dvpsxq/prccq")
    df_pm = df_pm[df_pm['dvpsxq'] > 0]
    df_pm['dvpsxq'] = (df_pm['dvpsxq'] - df_pm['dvpsxq'].mean()) ** 2 / df_pm['dvpsxq'].var()
    return get_formatted_feature(df_pm, df_ret, "dvpsxq", holding_period)


def get_cr_df(
    df_comp_fundq: pd.DataFrame, df_ret: pd.DataFrame, holding_period: pd.DataFrame
) -> pd.DataFrame:
    """
    Get current ratio = current total assets quarterly / current total liabilities quarterly.
    """
    df_pm = df_comp_fundq["date cusip actq lctq consol".split()]
    df_pm['cr'] = df_pm.eval("actq/lctq")
    return get_formatted_feature(df_pm, df_ret, "cr", holding_period)


def get_bm_df(
    df_comp_fundq: pd.DataFrame, df_ret: pd.DataFrame, holding_period: pd.DataFrame
) -> pd.DataFrame:
    """
    Get bp = net assets / total market value
    """
    df_pm = df_comp_fundq["date cusip cstkeq prccq cshoq consol".split()]
    df_pm['bm'] = df_pm.eval("cstkeq/(prccq*cshoq)")
    return get_formatted_feature(df_pm, df_ret, "bm", holding_period)


def get_e2e_df(
    df_comp_fundq: pd.DataFrame, df_ret: pd.DataFrame, holding_period: pd.DataFrame
) -> pd.DataFrame:
    """
    Get ev/ebitda = (total market value - net liability + cash and cash equivalents) / (operating income + depreciation and amortization)
    """
    df_pm = df_comp_fundq["date cusip prccq cshoq lctq cheq oiadpq dpq consol".split()]
    df_pm['e2e'] = df_pm.eval("((prccq*cshoq)-lctq+cheq)/(oiadpq+dpq)")
    return get_formatted_feature(df_pm, df_ret, "e2e", holding_period)


def get_sm_df(
    df_comp_fundq: pd.DataFrame, df_ret: pd.DataFrame, holding_period: pd.DataFrame
) -> pd.DataFrame:
    """
    Get sales-to-market ratio = sales quarterly / total market value
    """
    df_pm = df_comp_fundq["date cusip saleq prccq cshoq consol".split()]
    df_pm['sm'] = df_pm.eval("saleq/(prccq*cshoq)")
    return get_formatted_feature(df_pm, df_ret, "sm", holding_period)


def get_peg_df(
    df_comp_fundq: pd.DataFrame, df_ret: pd.DataFrame, holding_period: pd.DataFrame
) -> pd.DataFrame:
    """
    Get peg ratio = p/e ratio / earnings growth rate
    """
    df_pm = df_comp_fundq["date cusip prccq epspxq cshoq consol".split()]
    df_pm['epspxq_pct'] = df_pm.groupby('cusip')['epspxq'].pct_change()
    df_pm = df_pm[df_pm['epspxq_pct'] > 0]
    df_pm['pe'] = df_pm.eval("prccq/epspxq")
    df_pm['peg'] = df_pm.eval("pe/epspxq_pct")
    return get_formatted_feature(df_pm, df_ret, "peg", holding_period)


if __name__ == "__main__":
    pass
