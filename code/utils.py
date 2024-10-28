import numpy as np
import pandas as pd


def create_empty_df(df):
    return pd.DataFrame(np.nan, index=df.index, columns=df.columns)


def create_ew_df(df):
    return df / df.sum(axis=1).to_numpy().reshape(-1, 1)


def date_cusip_pivot(df, val):
    temp = df.pivot(index='date', columns='cusip', values=val)
    temp.index = pd.to_datetime(temp.index)
    return temp.sort_index()


def get_simple_ret(df):
    return df.diff() / df.shift()


def cusip_val_pct(group, val):
    temp = pd.DataFrame()
    temp['cusip'] = group['cusip']
    temp['date'] = group['date']
    temp[f'{val}_pct'] = get_simple_ret(group[val])
    return temp


def get_time_merge(df_feature, df_ret):
    closest_indices = df_feature.index.searchsorted(df_ret.index, side="right") - 1
    closest_indices = closest_indices[closest_indices >= 0]
    aligned_df = df_feature.iloc[closest_indices].set_index(df_ret.index[:len(closest_indices)]).ffill()
    return aligned_df
