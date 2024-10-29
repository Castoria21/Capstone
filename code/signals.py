import pandas as pd
import numpy as np
from utils import get_formatted_feature

def get_prc_mom(ret_df, window, periods_skipped=None):
    if periods_skipped == None:
        return (1 + ret_df).rolling(window=window).apply(np.prod)
    return (1 + ret_df).rolling(window=window).apply(np.prod) / (1 + ret_df).rolling(window=periods_skipped).apply(np.prod)


def get_eps_df(df_comp, df_ret, ffill_limit=12, scale_window=None):
    """
    Get earnings per share percentage change.
    """
    df = df_comp.copy(deep=True)
    df = df[df.cshfd > 0]
    df['eps'] = df.eval("ni/cshfd")
    return get_formatted_feature(df, df_ret, "eps", ffill_limit, scale_window)


def get_pm_df(df_comp, df_ret, ffill_limit=12, scale_window=None):
    """
    Get profit margin percentage change.
    """
    df = df_comp.copy(deep=True)
    df = df[df.indfmt == 'INDL']
    df = df[df.cusip != 'nan'].drop_duplicates(subset=['date', 'cusip']).sort_values(['cusip', 'date'])
    df['pm'] = df.eval("ni/sale")
    return get_formatted_feature(df, df_ret, "pm", ffill_limit, scale_window)
    
