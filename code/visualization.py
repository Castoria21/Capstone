"""
This script contains functionalities for visualization.
"""

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("default")


def plot_solo_cum(
        ret: pd.Series, title: str='', log_scale: bool=True
):  
    """
    Plot the cumulative returns for one single portfolio.
    """
    plt.figure(figsize=(12,8))
    plt.plot((1+ret).cumprod())
    plt.xticks(ret.index[::50], rotation=45)
    if log_scale:
        plt.yscale('log')
    if title != None:
        plt.title(f"{title}")
    plt.show()


def plot_group_cum(
        portfolio_df: pd.DataFrame, feature_name: str, log_scale: bool=True
):  
    """
    Plot the cumulative returns for each sorted portfolio.

    Args:
        portfolio_df: a DataFrame with values of returns, 
            columns of sorted portfolio (order from low to high), and index of time.
        feature_name: the name of feature.
        log_scale: whether to use log scale on the y-axis or not.
    """
    labels = [f"Low {feature_name}"] + [f"{i+1}" for i in portfolio_df.columns[1:-1]] + [f"High {feature_name}"]
    plt.figure(figsize=(12,8))
    for i in portfolio_df:
        plt.plot((1+portfolio_df[i]).cumprod(), label=f"{labels[i]}")
    plt.legend()
    plt.xticks(portfolio_df.index[::50], rotation=45)
    plt.title(f"{feature_name} Sorted Portfolio")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    if log_scale:
        plt.yscale("log")
    plt.show()
