
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


#==============================================================================

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


#==============================================================================

def get_moving_avg(x, lookback=10):
    x = x.to_numpy()
    return moving_avg(x, lookback)

#==============================================================================

@njit
def weighted_avg(values: np.ndarray, weights: np.ndarray):
    """
    Return the weighted average.

    values, weights -- NumPy ndarrays with the same shape.
    """
    masked_values = values[~np.isnan(values)]
    masked_weights = weights[~np.isnan(values)]

    return np.average(masked_values, weights=masked_weights)


#==============================================================================

def get_weighted_avg(x: pd.Series, weights: np.ndarray):

    stock_id = x["stock_id"]
    values = x.drop(columns=["stock_id"]).to_numpy().ravel()
    weights = weights[stock_id]

    return weighted_avg(values, weights)

#==============================================================================

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

#==============================================================================

def get_weighted_std(x: pd.Series, weights: np.ndarray):

    stock_id = x["stock_id"]
    values = x.drop(columns=["stock_id"]).to_numpy().ravel()
    weights = weights[stock_id]

    return weighted_std(values, weights)


"""

#==============================================================================

def get_rsi(x: pd.Series, length: int = 14):

    change = x.diff()
    gain = change.mask(change < 0, 0.0)
    loss = -change.mask(change > 0, -0.0)

    avg_gain = gain.ewm(alpha=1. / length, ignore_na=True).mean()
    avg_loss = loss.ewm(alpha=1. / length, ignore_na=True).mean()

    rsi = 100. - (100. / (1. + avg_gain / avg_loss))

    return rsi

#==============================================================================

def make_size_combs(df, sizes, total=True):
    for c in combinations(sizes, 2):
        print(c)
        df[f"{c[0]}.{c[1]}.spread"] = df[c[0]] - df[c[1]]
        df[f"{c[0]}.{c[1]}.imb1"] = (df[c[0]] - df[c[1]]) / (df[c[0]] + df[c[1]])
        df[f"{c[0]}.{c[1]}.ratio"] = df[c[0]] / df[c[1]]
        #df[f"{c[0]}.{c[1]}.hmean"] = (2. / ( 1. / df[c[0]] + 1. / df[c[1]]))# Harmonic mean
        if total:
            df[f"{c[0]}.{c[1]}.total"] = df[c[0]] + df[c[1]]

#==============================================================================

def make_price_combs(df, prices, mid_price=False):
    for c in combinations(prices, 2):
        df[f"{c[0]}.{c[1]}.spread"] = df[c[0]] - df[c[1]]
        df[f"{c[0]}.{c[1]}.imb1"] = (df[c[0]] - df[c[1]]) / (df[c[0]] + df[c[1]])
        #df[f"{c[0]}.{c[1]}.ratio"] = df[c[0]] / df[c[1]]

    for c in combinations(prices, 3):
        max_ = df[list(c)].max(axis=1)
        min_ = df[list(c)].min(axis=1)
        mid_ = df[list(c)].sum(axis=1) - min_ - max_

        df[f"{c[0]}.{c[1]}.{c[2]}.imb2"] = (max_ - mid_)/(mid_ - min_)
"""

#==============================================================================

def generate_time_features(df):

    pass
    #df["dow"] = df["date_id"] % 5  # Day of the week
    #df["seconds"] = df["seconds_in_bucket"] % 60  # Seconds
    #df["minute"] = df["seconds_in_bucket"] // 60  # Minutes


#==============================================================================

def generate_size_features(df):
    # Calculate various features
    df["bid_size.ask_size.total"] = df["ask_size"] + df["bid_size"]
    df["bid_size.ask_size.imbalance"] = (df["bid_size"] - df["ask_size"]) / (df["bid_size"] + df["ask_size"])
    df["imbalance_size.matched_size.imbalance"] = (df["imbalance_size"] - df["matched_size"]) / (df["matched_size"] + df["imbalance_size"])
    df["bid_size.ask_size.ratio"] = df["bid_size"] / df["ask_size"]


#==============================================================================

def generate_price_features(df):
    df["ask_price.bid_price.mean"] = (df["ask_price"] + df["bid_price"]) / 2
    df["ask_price.bid_price.spread"] = df["ask_price"] - df["bid_price"]
    df["far_price.near_price.spread"] = df["far_price"] - df["near_price"]

    # Create features for pairwise price imbalances
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    for c in combinations(prices, 2):
        df[f"{c[0]}.{c[1]}.imbalance"] = (df[c[0]] - df[c[1]])/(df[c[0]] + df[c[1]])

    # Calculate triplet imbalance features using the Numba-optimized function
    for c in combinations(prices, 3):
        max_ = df[list(c)].max(axis=1)
        min_ = df[list(c)].min(axis=1)
        mid_ = df[list(c)].sum(axis=1) - min_ - max_

        df[f"{c[0]}.{c[1]}.{c[2]}.imbalance"] = (max_ - mid_)/(mid_ - min_)

#==============================================================================

def generate_mixed_features(df: pd.DataFrame) -> None:

    # Calculate additional features
    df["prices.spread.imbalance_size.bid_price.product"] = df["ask_price.bid_price.spread"] * df["imbalance_size"] 
    df["prices.spread.sizes.imbalances.product"] = df["ask_price.bid_price.spread"] * df["bid_size.ask_size.imbalance"]
    df["prices.spread.product"] = df["ask_price.bid_price.spread"] * df["far_price.near_price.spread"]

#==============================================================================

def generate_rolling_features(df: pd.DataFrame, revealed_target: bool = True) -> None:

    df["imbalance_size.diff.matched_size.ratio"] = df.groupby("stock_id")["imbalance_size"].diff(periods=1) / df["matched_size"]
    df["ask_price.bid_price.spread.diff"] = df.groupby("stock_id")["ask_price.bid_price.spread"].diff()

    # Calculate shifted and return features for specific columns
    for col in ["matched_size", "imbalance_size", "reference_price", "imbalance_buy_sell_flag"]:
        for window in [1, 2, 3, 10]:
            df[f"{col}.shift.{window}"] = df.groupby("stock_id")[col].shift(window)
            df[f"{col}.pct_change.{window}"] = df.groupby("stock_id")[col].pct_change(window)
    
    # Calculate diff features for specific columns
    for col in ["ask_price", "bid_price", "ask_size", "bid_size"]:
        for window in [1, 2, 3, 10]:
            df[f"{col}.diff.{window}"] = df.groupby("stock_id")[col].diff(window)    

    # Add revealed target
    df[f"revealed_target"] = df.groupby(["stock_id", "seconds_in_bucket"])["target"].shift(1)


#==============================================================================


def generate_aggregated_features(df: pd.DataFrame, global_stock_aggregations: Mapping[int, Any] = None, weights: List[float] = None) -> None:

    apply_engine = "numba"
    sample = df.groupby(["date_id", "seconds_in_bucket"])

    # Calculate various statistical aggregation features
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    for func in ["mean", "std", "skew", "kurt"]:
        df[f"all_prices.{func}.aggr"] = df[prices].aggregate(func, axis=1)
        df[f"all_sizes.{func}.aggr"] = df[sizes].aggregate(func, axis=1)

    # Map global features to the DataFrame
    if global_stock_aggregations is not None:
        for key, value in global_stock_aggregations.items():
            df[f"{key}.global_aggr"] = df["stock_id"].map(value.to_dict())


    date_aggregations = {}

    for size in sizes:
        date_aggregations[size + ".mean"] = sample[size].mean(engine=apply_engine).to_dict()
        date_aggregations[size + ".std" ] = sample[size].std(engine=apply_engine).to_dict()

    for price in prices:
        if weights is not None:
            date_aggregations[price + ".mean.weighted"] = sample[[price, "stock_id"]].apply(lambda x: get_weighted_avg(x, weights=weights)).to_dict()
            date_aggregations[price + ".std.weighted"] = sample[[price, "stock_id"]].apply(lambda x: get_weighted_std(x, weights=weights)).to_dict()

        else:
            date_aggregations[price + ".mean"] = sample[price].mean(engine=apply_engine).to_dict()
            date_aggregations[price + ".std"] = sample[price].std(engine=apply_engine).to_dict()


    # Perform aggregations and features dependent on these
    logger.info("Applying date-wise aggregations...")
    for key, value in date_aggregations.items():
        aggr_index = ["date_id", "seconds_in_bucket"]
        aggr_colum = pd.Series(value, name=key + ".date_aggr")
        aggr_colum.index = aggr_colum.index.set_names(aggr_index)
        df = merge_df(df, aggr_colum, on=aggr_index)



#==============================================================================

def generate_global_features(df: pd.DataFrame) -> None:

    apply_engine = "numba"

    global_stock_id_feats = {
        "size.median": df.groupby("stock_id")["bid_size"].mean(engine=apply_engine) + df.groupby("stock_id")["ask_size"].mean(engine=apply_engine),
        "size.std": df.groupby("stock_id")["bid_size"].std(engine=apply_engine) + df.groupby("stock_id")["ask_size"].std(engine=apply_engine),
        "size.ptp": df.groupby("stock_id")["bid_size"].max(engine=apply_engine) - df.groupby("stock_id")["bid_size"].min(engine=apply_engine),
        "price.median": df.groupby("stock_id")["bid_price"].mean(engine=apply_engine) + df.groupby("stock_id")["ask_price"].mean(engine=apply_engine),
        "price.std": df.groupby("stock_id")["bid_price"].std(engine=apply_engine) + df.groupby("stock_id")["ask_price"].std(engine=apply_engine),
        "price.ptp": df.groupby("stock_id")["bid_price"].max(engine=apply_engine) - df.groupby("stock_id")["ask_price"].min(engine=apply_engine),
    }

    return global_stock_id_feats

#==============================================================================

# Function to do the feature engineering
def feature_engineering(
    df: pd.DataFrame, 
    global_stock_aggregations: Mapping[int, Any] = None, 
    weights: List[float] = None,
    revealed_target: bool = True, 
    reduce_memory: bool = True
) -> None:

    ########################
    # Time features
    ########################
    logger.info("Generating time features...")
    generate_time_features(df)

    ########################
    # Size-only features
    ########################
    logger.info("Generating size features...")
    generate_size_features(df)

    ########################
    # Price-only features
    ########################
    logger.info("Generating price features...")
    generate_price_features(df)

    ########################
    # Mixed features
    ########################
    logger.info("Generating mixed features...")
    generate_mixed_features(df)

    ########################
    # Aggregated features
    ########################
    logger.info("Generating aggregated features...")
    generate_aggregated_features(df, global_stock_aggregations, weights)

    ########################
    # Rolling features
    ########################
    logger.info("Generating rolling features...")
    generate_rolling_features(df, revealed_target)

    ########################
    # A bit of cleaning
    ########################
    # Replace infinite values with nulls
    df.replace([np.inf, -np.inf], 0)

    if reduce_memory:
        logger.info("Reducing data memory footprint...")
        downcast(df)

    return df


#==============================================================================

"""
def make_features(df: pd.DataFrame, reduce_memory: bool = True) -> None:

    logger.info("Creating additional features...")

    # Volumes
    #df["ask_volume"] = df["ask_size"] * df["ask_price"]
    #df["bid_volume"] = df["bid_size"] * df["bid_price"]
    #df["spread_volume"] =  df.eval("ask_volume - bid_volume", engine=eval_engine)
    #df["total_volume"] = df.eval("ask_volume + bid_volume", engine=eval_engine)
    #df["ratio_volume"] = df.eval("bid_volume / ask_volume", engine=eval_engine)

    # Size features
    df["total_size"] = df["ask_size"] + df["bid_size"]
    df["ratio_size"] = df["bid_size"] / df["ask_size"]
    df["imb_s1"] = (df["bid_size"] - df["ask_size"])/(df["bid_size"] + df["ask_size"])

    # Imbalance features
    df["imb_ratio"] = df["imbalance_size"] / df["matched_size"]
    df["imb_spread"] = df["imbalance_size"] - df["matched_size"]
    df["imb_s2"] = (df["imbalance_size"] - df["matched_size"])/(df["matched_size"] + df["imbalance_size"])

    #make_size_combs(df, ["ask_size", "bid_size"], total=True)
    #make_size_combs(df, ["imbalance_size", "matched_size"])
    make_price_combs(df, ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"])

    # Encode imbalance flag as separate features
    df_encoded = pd.get_dummies(df["imbalance_buy_sell_flag"])
    df_encoded = df_encoded.rename(columns={
        -1: "sell_side_imbalance", 
        0 : "neutral_imbalance", 
        1: "buy_side_imbalance"
    }).astype(np.int8).drop(columns=["neutral_imbalance"])

    df = pd.concat([df, df_encoded], axis=1)

    if reduce_memory:
        logger.info("Reducing data memory footprint...")
        downcast(df)

    return df

#==============================================================================

def make_aggregated_features(df: pd.DataFrame, weights: List[float] = None, reduce_memory: bool = False):

    if not isinstance(weights, np.ndarray):
        weights = np.array(weights)

    apply_engine = "numba"
    sample = df.groupby(["date_id", "seconds_in_bucket"], group_keys=False, sort=False)

    date_aggregations = {}

    sizes = ["ask_size.bid_size.imb1", "imbalance_size.matched_size.imb1"]
    for size in sizes:
        date_aggregations[size + ".mean"] = sample[size].mean(engine=apply_engine).to_dict()
        date_aggregations[size + ".std" ] = sample[size].std(engine=apply_engine).to_dict()

    prices = ["wap", "reference_price", "ask_price", "bid_price"]
    for price in prices:
        if weights is not None:
            date_aggregations[price + ".mean.weighted"] = sample[[price, "stock_id"]].apply(lambda x: get_weighted_avg(x, weights=weights)).to_dict()
            date_aggregations[price + ".std.weighted"] = sample[[price, "stock_id"]].apply(lambda x: get_weighted_std(x, weights=weights)).to_dict()

        else:
            date_aggregations[price + ".mean"] = sample[price].mean(engine=apply_engine).to_dict()
            date_aggregations[price + ".std"] = sample[price].std(engine=apply_engine).to_dict()


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

#==============================================================================


def make_rolling_features(df: pd.DataFrame, windows: List[float] = [12, 24, 36, 48], reduce_memory: bool = True) -> None:

    apply_engine = "numba"

    sample_date = df.groupby(["stock_id", "seconds_in_bucket"])
    # Add lagged target
    df[f"revealed_target"] = sample_date["target"].shift(1)

    sample_stock = df.groupby(["stock_id"])
    prices = ["reference_price", "ask_price", "bid_price", "wap"]
    
    # Adding previous date target as feature
    for window in windows:
        for elem in prices:
            df[f"{elem}.SMA.window{window * 10}"] = sample_stock[elem].rolling(window, min_periods=1).mean(engine=apply_engine)
            df[f"{elem}.std.window{window * 10}"] = sample_stock[elem].rolling(window, min_periods=1).std(engine=apply_engine)

    for elem in prices:
        df[f"{elem}.MACD"] = (sample_stock[elem].ewm(12, ignore_na=True).mean() - \
                              sample_stock[elem].ewm(26, ignore_na=True).mean()).reset_index()[elem]
        
        df[f"{elem}.RSI"] = sample_stock[elem].apply(get_rsi).reset_index()[elem]

    if reduce_memory:
        logger.info("Reducing data memory footprint...")
        downcast(df)

    return df
"""

#==============================================================================

def select_features(df: pd.DataFrame, features: List[str] = None, reduce_memory: bool = True) -> None:
    """This function dorps features. Works fully in memory"""

    if features is not None:
        logger.info("Dropping unnecesary features...")
        df.drop(columns=[elem for elem in df.columns if elem not in features], inplace=True)

    if reduce_memory:
        logger.info("Reducing data memory footprint...")
        downcast(df)

    return df

#==============================================================================

def clean_dataset(df: pd.DataFrame) -> None:
    """This function cleans the dataset row-wise. Works fully in memory"""

    logger.info("Cleaning dataset...")

    # Drop all nans rows in target 
    df.dropna(subset="target", inplace=True)

    # Reset infex in-place
    df.reset_index(drop=True, inplace=True)

#==============================================================================
