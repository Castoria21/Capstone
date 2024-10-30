import pandas as pd
import numpy as np
from utils import get_formatted_feature, cusip_val_pct, get_non_duplicated_df


def get_momentum(df, window=12, periods_skipped=1):
    if periods_skipped == 0:
        return (1 + df).rolling(window=window).apply(np.prod)
    return (1 + df).rolling(window=window).apply(np.prod) / (1 + df).rolling(window=periods_skipped).apply(np.prod)


def get_eps_pct_df(df_comp, df_ret, ffill_limit=12, scale_window=None):
    """
    Get earnings per share percentage change.
    """
    df = df_comp.copy(deep=True)
    df = df[df.cshfd > 0]
    df['eps'] = df.eval("ni/cshfd")
    df = get_non_duplicated_df(df, "eps")
    df = df.groupby("cusip").apply(lambda x: cusip_val_pct(x, "eps", scale_window))
    return get_formatted_feature(df, df_ret, "eps_pct", ffill_limit, scale_window)


def get_pm_pct_df(df_comp, df_ret, ffill_limit=12, scale_window=None):
    """
    Get profit margin percentage change.
    """
    df = df_comp.copy(deep=True)
    df = df[df.indfmt == 'INDL']
    df['pm'] = df.eval("ni/sale")
    df = get_non_duplicated_df(df, "pm")
    df = df.groupby("cusip").apply(lambda x: cusip_val_pct(x, "pm", scale_window))
    return get_formatted_feature(df, df_ret, "pm_pct", ffill_limit, scale_window)


def get_size_df(df_comp, df_ret, ffill_limit=12, scale_window=None):
    """
    Get market value as size.
    """
    df = df_comp.copy(deep=True)
    df = get_non_duplicated_df(df, "mkvalt")
    return get_formatted_feature(df, df_ret, "mkvalt", ffill_limit, scale_window)


def get_bm_df(df_comp, df_ret, ffill_limit=12, scale_window=None):
    """
    Get bm.
    """
    df = df_comp.copy(deep=True)
    df["be"] = df.eval("ceq")
    df = get_non_duplicated_df(df, "be")
    return get_formatted_feature(df, df_ret, "be", ffill_limit, scale_window)