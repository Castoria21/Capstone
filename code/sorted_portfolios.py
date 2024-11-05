"""
This script contains functionalities to compute the sorted portfolio returns given characteristics and returns.
"""

from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from utils import create_empty_df, create_ew_df


def get_sorted_index(
        df: pd.DataFrame, holding_period: int=12, quantiles: np.ndarray=np.linspace(.1, 1, 9, endpoint=False)
) -> list[pd.DataFrame]:
    """
    This function is used to construct a list of DataFrames that indicates if the stocks contained in the sorted group.

    Args:
        df: the DataFrame with columns of cusip, index of time, and values of characteristics.
        holding_period: represents the period the portfolio holds.
        quantiles: the quantiles used to sort.

    Returns:
        a list containing the index DataFrame, from the lowest group to the highest group.
    """
    groups = quantiles.shape[0] + 1
    threshold = np.nanquantile(df, quantiles, axis=1).T
    df_idx_l = []
    for i in range(groups):
        idx_temp = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
        # The lowest group
        if i == 0:
            idx_temp[df < threshold[:, i:i+1]] = 1
        # The med groups
        elif i == groups-1:
            idx_temp[df > threshold[:, i-1:i]] = 1
        # The highest groups
        else:
            idx_temp[(df <= threshold[:, i:i+1]) & (df >= threshold[:, i-1:i])] = 1
        # If holding period > 1, we forward fill the index dataframe.
        if holding_period > 1:
            df_idx_l.append(idx_temp.ffill(limit=holding_period-1))
        else:
            df_idx_l.append(idx_temp)
    return df_idx_l


def get_sorted_portfolio(
        df_ret: pd.DataFrame, df_idx_l: list[pd.DataFrame], cost: float=0.0005, weights: Optional[list[pd.DataFrame], None]=None
) -> pd.DataFrame:
    """
    This function produces the sorted portfolio returns.

    Args:
        df_ret: the DataFrame with columns of cusip, index of time and values of returns.
        df_idx_l: the list of index DataFrame produced by 'get_sorted_index' function.
        cost: the percentage transaction cost.
        weights: If None, use equal-weighted. Else, specify a list of DataFrames represents the weights for each group.
    Returns:
        a DataFrame represents the sorted portfolio returns.
    """
    if weights == None:
        weights = [create_ew_df(df) for df in df_idx_l]
    groups = len(df_idx_l)
    portfolio = pd.concat(
        [
            (df_ret * (df_idx_l[i] * weights[i]).shift()).sum(axis=1) for i in range(groups)
        ], axis=1
    )
    return portfolio
    