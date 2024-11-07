"""
This script contains utility functionalities for computation.
"""

from datetime import datetime
import numpy as np
import pandas as pd


def forward_fill(
        df: pd.DataFrame, holding_period: int
) -> pd.DataFrame:
    """
    Make the DataFrame's value last for (holding_period - 1) rows.
    """
    if holding_period > 1:
        return df.ffill(limit=holding_period-1)
    # If holding period = 1, return the original df.
    return df


def filter_datetime_df(
        df: pd.DataFrame, start: tuple[int, int, int], end: tuple[int, int, int]
) -> pd.DataFrame:
    """
    Return the DataFrame during specified period.
    """
    return df[(df.index <= datetime(*end)) & (df.index >= datetime(*start))]


def create_empty_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a DataFrame with empty values and same columns and indexes as df.
    """
    return pd.DataFrame(np.nan, index=df.index, columns=df.columns)


def create_ew_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a DataFrame represents equal-weights.
    """
    return df / df.sum(axis=1).to_numpy().reshape(-1, 1)


def get_volume_threshold(
        df_vol: pd.DataFrame, threshold: float=0.3, window: int=12
) -> pd.DataFrame:
    """
    Exclude the stocks with lowest volume.

    Args:
        df_vol: a DataFrame with values of volume, columns of cusip and index of time.
        threshold: the lowest (threshold * 100%) volume will be excluded in sorting at time t.
        window: the number of most recent volume to be calculated.

    Returns:
        a DataFrame with values of either 1 or Nan. Element (i, j) equal to 1 means stock(i) will be included at time (j). Otherwise, excluded.
    """
    df_vol_ma = df_vol.rolling(window=window).mean()
    vol_q = np.nanquantile(df_vol, threshold, axis=1).reshape(-1, 1)
    df_vol_idx = create_empty_df(df_vol)
    df_vol_idx[df_vol_ma >= vol_q] = 1
    return df_vol_idx


def count_non_cols(df:pd.DataFrame, axis: int=1) -> pd.Series:
    """
    Count the number of non-empty values in each row.
    """
    return df.apply(lambda x: df.shape[axis] - x.isna().sum(), axis=1)


def date_cusip_pivot(df: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    """
    Return a pivot DataFrame with values of 'val', index of 'date', and columns of 'cusip'.
    """
    temp = df.pivot(index='date', columns='cusip', values=val)
    return temp.sort_index()


def get_simple_ret(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the simple return of a DataFrame.
    """
    return df.diff() / df.shift()


def get_time_merge(df_feature: pd.DataFrame, df_ret: pd.DataFrame) -> pd.DataFrame:
    """
    Match most recent values of df_feature based on df_ret.

    Args:
        df_feature: a DataFrame with values of feature, columns of cusip and index of time.
        df_ret: a DataFrame with values of return and same columns and index as df_feature.

    Returns:
        a DataFrame with values of most recent feature based on df_ret.
    """
    closest_indices = df_feature.index.searchsorted(df_ret.index, side="right") - 1
    closest_indices = closest_indices[closest_indices >= 0]
    aligned_df = df_feature.iloc[closest_indices].set_index(df_ret.index[:len(closest_indices)])
    return aligned_df


def get_non_duplicated_df(df: pd.DataFrame, val: str) -> pd.DataFrame:
    """
    Get rid of duplicated rows with identical date and cusip and empty vals.
    """
    temp = df.copy(deep=True)
    temp = temp[~((temp.duplicated(subset=['date', 'cusip'], keep=False)) & (df[val].isna()))]
    # If there're still duplicates, it could be caused by different "consol".
    temp = temp[~((temp.duplicated(subset=['date', 'cusip'], keep=False)) & (df.consol == "C"))]
    return temp


def get_formatted_feature(
        df_feature: pd.DataFrame, df_ret: pd.DataFrame, 
        feature_name: str, holding_period: int
) -> pd.DataFrame:
    """
    This function returns the formatted feature DataFrame used for sorting.

    Args:
        df_feature: a DataFrame with columns of date, cusip, 'feature_name', consol
        df_ret: a DataFrame with values of return, columns of cusip, and index of time.
        feature_name: the name of feature.
        holding_period: how long does the feature value last.
    
    Returns:
        a DataFrame with values of feature, and same columns and index as df_ret.
    """
    # Make sure there is no duplication in pivot.
    temp = get_non_duplicated_df(df_feature, feature_name)
    # Get the pivot DataFrame and make the columns the same as those of df_ret
    temp =  date_cusip_pivot(temp, feature_name).reindex(columns=df_ret.columns)
    # Forward fill
    temp = forward_fill(get_time_merge(temp, df_ret), holding_period)
    return temp


if __name__ == "__main__":
    pass
