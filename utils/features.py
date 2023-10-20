
import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path
from typing import List, Mapping, Any

try:
    from loguru import logger
except:
    import logging
    logger = logging.getLogger("__main__")

from .compression import downcast


# Realized return
def realized_return(wap):
    return np.sqrt(((np.log(wap).diff())**2).sum())

# Log return
def log_return(series):
    return np.log(series).diff()


def make_features(df: pd.DataFrame, aggregations: Mapping[str, Any], reduce_memory: bool = True, ) -> None:
    """This function creates new features for analysis. Works fully in memory"""

    logger.info("Creating additional features...")

    # Volumes
    df["ask_volume"] = df.eval("ask_size * ask_price")
    df["bid_volume"] = df.eval("bid_size * bid_price")
    df["spread_volume"] =  df.eval("ask_volume - bid_volume")
    #df["total_volume"] = df.eval("ask_volume + bid_volume")

    # Size features
    df["size_total"] = df["ask_size"] + df["bid_size"] 
    df["spread_size"] = df["ask_size"] - df["bid_size"] 
    df["ratio_size"] = df["bid_size"].div(df["ask_size"])

    # Imbalance features
    df["imb_ratio"] = df["imbalance_size"] / df["matched_size"]
    df["imb_s1"] = df.eval("(bid_size - ask_size)/(bid_size + ask_size)")
    df["imb_s2"] = df.eval("(imbalance_size - matched_size)/(matched_size + imbalance_size)")

    # Add 1 so as not to have negative values
    #df["imbalance_buy_sell_flag"] += 1

    # Perform aggregations and features dependent on these
    if aggregations is not None:
        for key, value in aggregations.items():
            df[key + "_aggr"] = df["stock_id"].map(value)
    
    if "median_size" in df.columns:
        df["high_volume"] = np.where(df["size"] > df["median_size"], 1, 0)

    # Date features
    #df["date_id_sin"] = np.sin(2. * np.pi * df["date_id"]/5.)
    #df["date_id_cos"] = np.cos(2. * np.pi * df["date_id"]/5.)

    # Combinations of price variables
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]

    df["mid_price"] = (df["ask_price"] + df["bid_price"])/ 2.

    """
    rolling_features = True
    if rolling_features:
        for price in prices:
            df[f"log_return_{price}"] = df.groupby(["time_id"])[price].apply(log_return).reset_index()[price]
    """

    # Price features
    for c in combinations(prices, 2):
        #df[f"{c[0]}_plus_{c[1]}"] = (df[f"{c[0]}"] + df[f"{c[1]}"])
        df[f"{c[0]}_minus_{c[1]}"] = (df[f"{c[0]}"] - df[f"{c[1]}"])
        #df[f"{c[0]}_times_{c[1]}"] = (df[f"{c[0]}"] * df[f"{c[1]}"])
        #df[f"{c[0]}_over_{c[1]}"] = (df[f"{c[0]}"] / df[f"{c[1]}"])
        df[f"{c[0]}_{c[1]}_imb1"] = df.eval(f"({c[0]}-{c[1]})/({c[0]}+{c[1]})")

    for c in combinations(prices, 3):
        max_ = df[list(c)].max(axis=1)
        min_ = df[list(c)].min(axis=1)
        mid_ = df[list(c)].sum(axis=1) - min_ - max_

        df[f"{c[0]}_{c[1]}_{c[2]}_imb2"] = (max_ - mid_)/(mid_ - min_)

    if reduce_memory:
        logger.info("Reducing data memory footprint...")
        downcast(df)



def select_features(df: pd.DataFrame, features: List[str] = None, reduce_memory: bool = True) -> None:
    """This function dorps features. Works fully in memory"""

    if features is not None:
        logger.info("Dropping unnecesary features...")
        df.drop(columns=[elem for elem in df.columns if elem not in features], inplace=True)

    if reduce_memory:
        logger.info("Reducing data memory footprint...")
        downcast(df)
