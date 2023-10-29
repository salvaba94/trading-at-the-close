
import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Mapping, Any
from numba import njit

try:
    from loguru import logger
except:
    import logging
    logger = logging.getLogger("__main__")

from .compression import downcast
from .utils import merge_df


# Log return
@njit
def log_return(series):
    return np.log(series).diff()


@njit
def moving_avg(arr: np.ndarray, window: int, min_periods: int = 1) -> np.ndarray:
    result = np.full(arr.shape, np.nan)  # Fill with nan
    if min_periods is None:
        min_periods = window

    for i in range(arr.shape[0]):
        windowed_data = arr[max(0, i - window + 1) : i + 1]
        valid_count = np.sum(~np.isnan(windowed_data))
        if valid_count >= min_periods:
            result[i] = np.nanmean(windowed_data)  # Compute mean considering possible NaN values

    return result

def get_moving_avg(x, lookback=10):
    x = x.to_numpy()
    return moving_avg(x, lookback)


@njit
def weighted_avg(values: np.ndarray, weights: np.ndarray):
    """
    Return the weighted average.

    values, weights -- NumPy ndarrays with the same shape.
    """
    masked_values = values[~np.isnan(values)]
    masked_weights = weights[~np.isnan(values)]

    return np.average(masked_values, weights=masked_weights)


def get_weighted_avg(x: pd.Series, weights: np.ndarray):

    stock_id = x["stock_id"]
    values = x.drop(columns=["stock_id"]).to_numpy().ravel()
    weights = weights[stock_id]

    return weighted_avg(values, weights)

@njit
def weighted_std(values, weights):
    """
    Return the weighted standard deviation.

    values, weights -- NumPy ndarrays with the same shape.
    """

    masked_values = values[~np.isnan(values)]
    masked_weights = weights[~np.isnan(values)]

    average = np.average(masked_values, weights=masked_weights)

    # Fast and numerically precise:
    variance = np.average((masked_values - average)**2, weights=masked_weights)
    return np.sqrt(variance)


def get_weighted_std(x: pd.Series, weights: np.ndarray):

    stock_id = x["stock_id"]
    values = x.drop(columns=["stock_id"]).to_numpy().ravel()
    weights = weights[stock_id]

    return weighted_std(values, weights)


def get_rsi(x: pd.Series, length: int = 14):

    change = x.diff()
    gain = change.mask(change < 0, 0.0)
    loss = -change.mask(change > 0, -0.0)

    avg_gain = gain.ewm(alpha=1. / length, ignore_na=True).mean()
    avg_loss = loss.ewm(alpha=1. / length, ignore_na=True).mean()

    rsi = 100. - (100. / (1. + avg_gain / avg_loss))

    return rsi


def make_features(df: pd.DataFrame, reduce_memory: bool = True) -> None:
    """This function creates new features for analysis. Works fully in memory"""

    logger.info("Creating additional features...")

    eval_engine = "numexpr"
    # Volumes
    #df["ask_volume"] = df.eval("ask_size * ask_price", engine=eval_engine)
    #df["bid_volume"] = df.eval("bid_size * bid_price", engine=eval_engine)
    #df["spread_volume"] =  df.eval("ask_volume - bid_volume", engine=eval_engine)
    #df["total_volume"] = df.eval("ask_volume + bid_volume", engine=eval_engine)
    #df["ratio_volume"] = df.eval("bid_volume / ask_volume", engine=eval_engine)

    # Size features
    df["total_size"] = df.eval("ask_size + bid_size", engine=eval_engine) 
    df["spread_size"] = df.eval("ask_size - bid_size", engine=eval_engine) 
    df["ratio_size"] = df.eval("bid_size / ask_size", engine=eval_engine)

    # Imbalance features
    df["imb_ratio"] = df.eval("imbalance_size / matched_size", engine=eval_engine)
    df["imb_s1"] = df.eval("(bid_size - ask_size)/(bid_size + ask_size)", engine=eval_engine)
    df["imb_s2"] = df.eval("(imbalance_size - matched_size)/(matched_size + imbalance_size)", engine=eval_engine)

    # Encode imbalance flag as separate features
    df_encoded = pd.get_dummies(df["imbalance_buy_sell_flag"])
    df_encoded = df_encoded.rename(columns={
        -1: "sell_side_imbalance", 
        0 : "neutral_imbalance", 
        1: "buy_side_imbalance"
    }).astype(np.int8).drop(columns=["neutral_imbalance"])

    df = pd.concat([df, df_encoded], axis=1)

    # Price features
    df["mid_price"] = df.eval("(ask_price + bid_price) / 2.", engine=eval_engine)

    # Combinations of price variables
    prices = ["reference_price", "ask_price", "bid_price", "wap", "far_price", "near_price"]
    # Price features
    for c in combinations(prices, 2):
        df[f"{c[0]}_{c[1]}_spread"] = df.eval(f"({c[0]} - {c[1]})", engine=eval_engine)
        df[f"{c[0]}_{c[1]}_imb1"] = df.eval(f"({c[0]}-{c[1]})/({c[0]}+{c[1]})", engine=eval_engine)
        #df[f"{c[0]}_{c[1]}_hmean"] = df.eval(f"(2. / ( 1. / {c[0]} + 1. / {c[1]}))", engine=eval_engine) # Harmonic mean


    for c in combinations(prices, 3):
        max_ = df[list(c)].max(axis=1)
        min_ = df[list(c)].min(axis=1)
        mid_ = df[list(c)].sum(axis=1) - min_ - max_

        df[f"{c[0]}_{c[1]}_{c[2]}_imb2"] = (max_ - mid_)/(mid_ - min_)

    if reduce_memory:
        logger.info("Reducing data memory footprint...")
        downcast(df)

    return df


def make_aggregated_features(df: pd.DataFrame, weights: List[float] = None, reduce_memory: bool = False):

    if not isinstance(weights, np.ndarray):
        weights = np.array(weights)

    apply_engine = "numba"
    sample = df.groupby(["date_id", "seconds_in_bucket"], group_keys=False, sort=False)


    date_aggregations = {}

    sizes = ["total_size", "spread_size", "ratio_size", "imb_s1", "imb_s2", "imb_ratio", "matched_size", "imbalance_size"]
    sizes = []
    for size in sizes:
        date_aggregations["mean_" + size] = sample[size].mean(engine=apply_engine).to_dict()
        date_aggregations["std_" + size] = sample[size].std(engine=apply_engine).to_dict()

    prices = ["wap", "reference_price", "ask_price", "bid_price"]
    prices = []
    for price in prices:
        if weights is not None:
            date_aggregations["weighted_mean_" + price] = sample[[price, "stock_id"]].apply(lambda x: get_weighted_avg(x, weights=weights)).to_dict()
            date_aggregations["weighted_std_" + price] = sample[[price, "stock_id"]].apply(lambda x: get_weighted_std(x, weights=weights)).to_dict()

        else:
            date_aggregations["mean_" + price] = sample[price].mean(engine=apply_engine).to_dict()
            date_aggregations["std_" + price] = sample[price].std(engine=apply_engine).to_dict()


    # Perform aggregations and features dependent on these
    logger.info("Applying date-wise aggregations...")
    for key, value in date_aggregations.items():
        aggr_index = ["date_id", "seconds_in_bucket"]
        aggr_colum = pd.Series(value, name=key + "_date_aggr")
        aggr_colum.index = aggr_colum.index.set_names(aggr_index)
        df = pd.merge(df, aggr_colum, on=aggr_index)

    if reduce_memory:
        logger.info("Reducing data memory footprint...")
        downcast(df)

    return df



def make_rolling_features(df: pd.DataFrame, windows: List[float] = [12, 24, 36, 48], reduce_memory: bool = True) -> None:

    apply_engine = "numba"

    sample_date = df.groupby(["stock_id", "seconds_in_bucket"])
    # Add lagged target
    df[f"revealed_target"] = sample_date["target"].shift(1)

    sample_stock = df.groupby(["stock_id"])
    prices = ["reference_price", "ask_price", "bid_price", "wap", "far_price", "near_price"]
    sizes = ["total_size", "spread_size", "ratio_size", "imb_s1", "imb_s2", "imb_ratio", "matched_size", "imbalance_size"]
    sizes = []
    prices = []
    # Adding previous date target as feature
    for window in windows:
        for elem in prices + sizes:
            df[f"{elem}_SMA{window * 10}"] = sample_stock[elem].rolling(window, min_periods=1).mean(engine=apply_engine)
            df[f"{elem}_std{window * 10}"] = sample_stock[elem].rolling(window, min_periods=1).std(engine=apply_engine)

    prices = ["wap"]
    for elem in prices + sizes:
        df[f"{elem}_MACD"] = (sample_stock[elem].ewm(12, ignore_na=True).mean() - \
                              sample_stock[elem].ewm(26, ignore_na=True).mean()).reset_index()[elem]
        
        df[f"{elem}_RSI"] = sample_stock[elem].apply(get_rsi).reset_index()[elem]


    if reduce_memory:
        logger.info("Reducing data memory footprint...")
        downcast(df)

    return df


def select_features(df: pd.DataFrame, features: List[str] = None, reduce_memory: bool = True) -> None:
    """This function dorps features. Works fully in memory"""

    if features is not None:
        logger.info("Dropping unnecesary features...")
        df.drop(columns=[elem for elem in df.columns if elem not in features], inplace=True)

    if reduce_memory:
        logger.info("Reducing data memory footprint...")
        downcast(df)

    return df
