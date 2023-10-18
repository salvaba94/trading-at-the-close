
import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path
from typing import List

try:
    from loguru import logger
except:
    import logging
    logger = logging.getLogger("__main__")



def aggregate(df: pd.DataFrame):
    """This function aggregates features."""

    aggregations = {
        "median_sizes": (df.groupby("stock_id")["bid_size"].median() + df.groupby("stock_id")["ask_size"].median()).to_dict(),
        "std_sizes": (df.groupby("stock_id")["bid_size"].std() + df.groupby("stock_id")["ask_size"].std()).to_dict(),
        "max_sizes": (df.groupby("stock_id")["bid_size"].max() + df.groupby("stock_id")["ask_size"].max()).to_dict(),
        "min_sizes": (df.groupby("stock_id")["bid_size"].min() + df.groupby("stock_id")["ask_size"].min()).to_dict(),
        "mean_sizes": (df.groupby("stock_id")["bid_size"].mean() + df.groupby("stock_id")["ask_size"].mean()).to_dict(),
        "first_sizes": (df.groupby("stock_id")["bid_size"].first() + df.groupby("stock_id")["ask_size"].first()).to_dict(),
        "last_sizes": (df.groupby("stock_id")["bid_size"].last() + df.groupby("stock_id")["ask_size"].last()).to_dict(),
    }

    return aggregations 
