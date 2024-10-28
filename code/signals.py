import pandas as pd
import numpy as np

def get_prc_mom(ret_df, window, periods_skipped):
    return (1 + ret_df).rolling(window=window).apply(np.prod) / (1 + ret_df).rolling(window=periods_skipped).apply(np.prod)
