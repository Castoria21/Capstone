"""
This script contains functionalities to evaluate the performance of the strategy.
"""

import pandas as pd
import numpy as np


def get_sharpe_ratio(
        ret: pd.Series, rf: float=0
) -> float:
    """
    Calculate the Sharpe ratio of the strategy.

    Args:
        ret: a Series of returns.
        rf: the risk-free rate.

    Returns:
        The Sharpe ratio.
    """
    return (ret.mean() - rf) / ret.std()


def get_maximum_drop_down(
        ret: pd.Series
) -> float:
    """
    Calculate the maximum drawdown of the strategy.

    Args:
        ret: a Series of returns.

    Returns:
        The maximum drawdown.
    """
    cum_ret = (1 + ret).cumprod()
    max_dd = (1 - cum_ret / cum_ret.cummax()).max()
    return max_dd


def get_aggregated_annual_metrics(
        monthly_ret: pd.Series, rf: float=0
) -> pd.Series:
    """
    Calculate the aggregated annual metrics of the strategy.

    Args:
        monthly_ret: a Series of monthly returns.
        rf: the risk-free rate.

    Returns:
        A Series of aggregated annual metrics.
    """
    ann_mean = monthly_ret.mean() * 12
    ann_std = monthly_ret.std() * np.sqrt(12)
    ann_sharpe = get_sharpe_ratio(monthly_ret, rf) * np.sqrt(12)
    ann_max_dd = get_maximum_drop_down(monthly_ret)
    return pd.Series({
        "Mean": f"{ann_mean:.2%}",
        "Std": f"{ann_std:.2%}",
        "SR": f"{ann_sharpe:.2f}",
        "MDD": f"{ann_max_dd:.2%}"
    })


if __name__ == "__main__":
    pass
