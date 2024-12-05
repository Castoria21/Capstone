"""
This script contains matrix functionalities for computation.
"""

from datetime import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm


def count_overlap(
        df1: pd.DataFrame, df2: pd.DataFrame
) -> float:
    """
    Count the overlap between two strategies.
    """
    df3 = df1.fillna(0)
    df4 = df2.fillna(0)
    overlap = df3*df4
    # Ignore stocks that are not held in either strategy
    at_least_one = ((df3 == 1) | (df4 == 1))
    # Take average of every correlation
    return overlap.sum(axis=1) / at_least_one.sum(axis=1)


def lin_reg(
        y: pd.Series, dic: dict
) -> sm.regression.linear_model:
    """
    Regress fundemental momentum return on fundemental factor return and price momentum return
    """
    dfs = [pd.DataFrame(y.to_numpy(), columns=['y'], index=y.index.strftime('%Y-%m'))]
    for key, val in dic.items():
        dfi = pd.DataFrame(val.to_numpy(), columns=[key], index=val.index.strftime('%Y-%m'))
        dfs.append(dfi)
    merged_df = pd.concat(dfs, axis=1, join='outer')
    merged_df = merged_df.loc[merged_df[merged_df.iloc[:, 0] != 0].index[0]:].dropna()
    y = merged_df.iloc[:, 0]
    X = sm.add_constant(merged_df.iloc[:, 1:])
    model = sm.OLS(y, X)
    return model


def matrics(
        ret: pd.Series
) -> float:
    """
    Calculate some basic matrics for factor, like factor return and t-value
    """
    cmpnd_ret = (1 + ret).prod()
    n = len(ret)
    month_ret = cmpnd_ret**(1/n) - 1
    annual_ret = (1+month_ret)**12 - 1
    annual_std = ret.std(ddof=1) * np.sqrt(12)
    t_value = annual_ret / annual_std
    print(f'Annualized return for factor is {100*annual_ret:.2f}%')
    print(f't-value for factor is {t_value:.4f}')
    return



if __name__ == "__main__":
    pass