from datetime import datetime
import numpy as np
import pandas as pd


def forward_fill(df, holding_period):
    if holding_period > 1:
        return df.ffill(limit=holding_period-1)
    return df


def filter_datetime_df(df, start, end):
    return df[(df.index <= datetime(*end)) & (df.index >= datetime(*start))]


def create_empty_df(df):
    return pd.DataFrame(np.nan, index=df.index, columns=df.columns)


def create_ew_df(df):
    return df / df.sum(axis=1).to_numpy().reshape(-1, 1)


def date_cusip_pivot(df, val, index_conversion=True):
    temp = df.pivot(index='date', columns='cusip', values=val)
    return temp.sort_index()


def get_simple_ret(df):
    return df.diff() / df.shift()


def cusip_val_pct(group, val, std_scale_window=None):
    temp = pd.DataFrame()
    temp['cusip'] = group['cusip']
    temp['date'] = group['date']
    temp[f'{val}_pct'] = get_simple_ret(group[val])
    if std_scale_window != None:
        temp[f'{val}_pct'] = temp[f'{val}_pct'] / temp[f'{val}_pct'].rolling(window=std_scale_window).std()
    return temp


def get_time_merge(df_feature, df_ret):
    closest_indices = df_feature.index.searchsorted(df_ret.index, side="right") - 1
    closest_indices = closest_indices[closest_indices >= 0]
    aligned_df = df_feature.iloc[closest_indices].set_index(df_ret.index[:len(closest_indices)])
    return aligned_df


def get_formatted_feature(df, df_ret, feature_name, holding_period):
    temp = get_non_duplicated_df(df, feature_name)
    temp =  date_cusip_pivot(temp, feature_name).reindex(columns=df_ret.columns)
    temp = forward_fill(get_time_merge(temp, df_ret), holding_period)
    return temp


def get_non_duplicated_df(df, val: str):
    """
    Get rid of duplicated rows with identical date and cusip and empty vals
    """
    temp = df.copy(deep=True)
    temp = temp[~((temp.duplicated(subset=['date', 'cusip'], keep=False)) & (df[val].isna()))]
    # If there're still duplicates, it could be caused by different "consol".
    temp = temp[~((temp.duplicated(subset=['date', 'cusip'], keep=False)) & (df.consol == "C"))]
    return temp


def get_volume_threshold(df_vol, threshold=0.7, window=12):
    df_vol_ma = df_vol.rolling(window=window).mean()
    vol_q = np.nanquantile(df_vol, threshold, axis=1).reshape(-1, 1)
    df_vol_idx = create_empty_df(df_vol)
    df_vol_idx[df_vol_ma >= vol_q] = 1
    return df_vol_idx


def get_certain_month_idx(df, month=6):
    return df.index.str.endswith(f"{month:0>2}")


def count_non_cols(df, axis=1):
    return df.apply(lambda x: df.shape[axis] - x.isna().sum(), axis=1)
