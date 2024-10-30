import pandas as pd
import numpy as np
from utils import get_formatted_feature, cusip_val_pct, get_non_duplicated_df, date_cusip_pivot


ffill_max = 12


def get_momentum(df, window=12, periods_skipped=1):
    if periods_skipped == 0:
        return (1 + df).rolling(window=window).apply(np.prod)
    return (1 + df).rolling(window=window).apply(np.prod) / (1 + df).rolling(window=periods_skipped).apply(np.prod)


def get_eps_pct_df(df_comp, df_ret, ffill_limit=ffill_max, scale_window=None):
    """
    Get earnings per share percentage change.
    """
    df = df_comp.copy(deep=True)
    df = df[df.cshfd > 0]
    df['eps'] = df.eval("ni/cshfd")
    df = get_non_duplicated_df(df, "eps")
    df = df.groupby("cusip").apply(lambda x: cusip_val_pct(x, "eps", scale_window))
    return get_formatted_feature(df, df_ret, "eps_pct", ffill_limit, scale_window)


def get_pm_pct_df(df_comp, df_ret, ffill_limit=ffill_max, scale_window=None):
    """
    Get profit margin percentage change.
    """
    df = df_comp.copy(deep=True)
    df = df[df.indfmt == 'INDL']
    df['pm'] = df.eval("ni/sale")
    df = get_non_duplicated_df(df, "pm")
    df = df.groupby("cusip").apply(lambda x: cusip_val_pct(x, "pm", scale_window))
    return get_formatted_feature(df, df_ret, "pm_pct", ffill_limit, scale_window)


def get_size_df(df_crsp, ffill_limit=ffill_max):
    """
    Get market value as size.
    """
    df = df_crsp.copy(deep=True)
    df['size'] = df.eval("prc*shrout")
    return date_cusip_pivot(df, "size").ffill(limit=ffill_limit)


def get_size_df_comp(df_comp, df_ret, ffill_limit=ffill_max):
    df = df_comp.copy(deep=True)
    df['size'] = df['mkvalt']
    df = get_non_duplicated_df(df, "size")
    return get_formatted_feature(df, df_ret, "size", ffill_limit=ffill_limit)



def get_bm_df(df_comp, df_ret, ffill_limit=ffill_max, scale_window=None):
    """
    Get bm.
    """
    df = df_comp.copy(deep=True)
    df = df[df.indfmt == 'INDL']
    df["bm"] = df.eval("ceq")
    df = get_non_duplicated_df(df, "bm")
    return get_formatted_feature(df, df_ret, "bm", ffill_limit=ffill_limit, scale_window=scale_window)
