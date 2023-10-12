
import numpy as np
import csv
import pandas as pd
from itertools import combinations
from pathlib import Path
from typing import List
from loguru import logger

from .compression import downcast


def featurize(df: pd.DataFrame, features: List[str] = None, reduce_memory: bool = True) -> None:
    '''This function creates new features for analysis. Works fully in memory'''

    logger.info("Creating additional features...")
    # Separate a -1, 0, 1 variable into two
    df["imbalance_buy_flag"] = np.where(df["imbalance_buy_sell_flag"] == 1, 1, 0) 
    df["imbalance_sell_flag"] = np.where(df["imbalance_buy_sell_flag"] == -1, 1, 0)

    # Overall size
    df["volume"] = df.eval("ask_size * ask_price + bid_size * ask_size")

    '''
    # Median volume, low and high volumes flag
    median_vol = df.groupby("stock_id")["bid_size"].median() + \
                df.groupby("stock_id")["ask_size"].median()
    df["median_volume"] = df["stock_id"].map(median_vol.to_dict())
    '''

    df["imb_s1"] = df.eval("(bid_size-ask_size)/(bid_size+ask_size)")
    df["imb_s2"] = df.eval("(imbalance_size-matched_size)/(matched_size+imbalance_size)")
    df["price_spread"] = df.eval("ask_price - bid_price")
    df["imbalance_ratio"] = df.eval("imbalance_size / matched_size")
    
    df["ask_volume"] = df.eval("ask_size * ask_price")
    df["bid_volume"] = df.eval("bid_size * bid_price")
        
    df["ask_bid_volumes_diff"] = df["ask_volume"] - df["bid_volume"] 
    df["bid_size_over_ask_size"] = df["bid_size"].div(df["ask_size"])
    df["bid_price_over_ask_price"] = df["bid_price"].div(df["ask_price"])

    # Imbalance features
    df["imbalance_ratio"] = df["imbalance_size"] / df["matched_size"]
    df["imb_s1"] = df.eval("(bid_size - ask_size)/(bid_size + ask_size)")
    df["imb_s2"] = df.eval("(imbalance_size - matched_size)/(matched_size + imbalance_size)")

    # Combinations of price variables
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]

    for c in combinations(prices, 2):
        df[f"{c[0]}_minus_{c[1]}"] = (df[f"{c[0]}"] - df[f"{c[1]}"]).astype(np.float32)
        df[f"{c[0]}_times_{c[1]}"] = (df[f"{c[0]}"] * df[f"{c[1]}"]).astype(np.float32)
        df[f"{c[0]}_{c[1]}_imb1"] = df.eval(f"({c[0]}-{c[1]})/({c[0]}+{c[1]})")

    for c in combinations(prices, 3):
        max_ = df[list(c)].max(axis=1)
        min_ = df[list(c)].min(axis=1)
        mid_ = df[list(c)].sum(axis=1) - min_ - max_

        df[f"{c[0]}_{c[1]}_{c[2]}_imb2"] = (max_ - mid_)/(mid_ - min_)

    if features is not None:
        logger.info("Dropping unnecesary features...")
        df.drop(columns=[elem for elem in df.columns if elem not in features], inplace=True)

    if reduce_memory:
        logger.info("Reducing data memory footprint...")
        downcast(df)
