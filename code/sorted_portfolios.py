import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from utils import create_empty_df, create_ew_df
from visualization import plot_group_cum, plot_solo_cum


def get_sorted_index(df, holding_period=12, quantiles=np.linspace(.1, 1, 9, endpoint=False)) -> list[pd.DataFrame]:
    groups = quantiles.shape[0] + 1
    threshold = np.nanquantile(df, quantiles, axis=1).T
    df_idx_l = []
    for i in range(groups):
        idx_temp = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
        if i == 0:
            idx_temp[df < threshold[:, i:i+1]] = 1
        elif i == groups-1:
            idx_temp[df > threshold[:, i-1:i]] = 1
        else:
            idx_temp[(df <= threshold[:, i:i+1]) & (df >= threshold[:, i-1:i])] = 1
        if holding_period > 1:
            df_idx_l.append(idx_temp.ffill(limit=holding_period))
        else:
            df_idx_l.append(idx_temp)
    return df_idx_l


def get_sorted_portfolio(df_ret, df_idx_l, cost=0.0005, weights=None):
    if weights == None:
        weights = [create_ew_df(df) for df in df_idx_l]
    groups = len(df_idx_l)
    portfolio = pd.concat(
        [
            (df_ret * (df_idx_l[i] * weights[i]).shift()).sum(axis=1) for i in range(groups)
        ], axis=1
    )
    plot_group_cum(portfolio)
    return portfolio
    