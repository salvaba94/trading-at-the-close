
import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path
from typing import List, Mapping, Any
from numba import njit, prange

try:
    from loguru import logger
except:
    import logging
    logger = logging.getLogger("__main__")

from .compression import downcast


#==============================================================================

# 📊 Function to compute triplet imbalance in parallel using Numba
@njit(parallel=True)
def compute_triplet_imbalance(df_values, comb_indices):
    num_rows = df_values.shape[0]
    num_combinations = len(comb_indices)
    imbalance_features = np.empty((num_rows, num_combinations))

    # 🔁 Loop through all combinations of triplets
    for i in prange(num_combinations):
        a, b, c = comb_indices[i]
        
        # 🔁 Loop through rows of the DataFrame
        for j in range(num_rows):
            max_val = max(df_values[j, a], df_values[j, b], df_values[j, c])
            min_val = min(df_values[j, a], df_values[j, b], df_values[j, c])
            mid_val = df_values[j, a] + df_values[j, b] + df_values[j, c] - min_val - max_val
            
            # 🚫 Prevent division by zero
            if mid_val == min_val:
                imbalance_features[j, i] = np.nan
            else:
                imbalance_features[j, i] = (max_val - mid_val) / (mid_val - min_val)

    return imbalance_features

# 📈 Function to calculate triplet imbalance for given price data and a DataFrame
def calculate_triplet_imbalance_numba(price, df):
    # Convert DataFrame to numpy array for Numba compatibility
    df_values = df[price].values
    comb_indices = [(price.index(a), price.index(b), price.index(c)) for a, b, c in combinations(price, 3)]

    # Calculate the triplet imbalance using the Numba-optimized function
    features_array = compute_triplet_imbalance(df_values, comb_indices)

    # Create a DataFrame from the results
    columns = [f"{a}.{b}.{c}.imbalance" for a, b, c in combinations(price, 3)]
    features = pd.DataFrame(features_array, columns=columns)

    return features


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

def get_weighted_avg(x: pd.Series):

    weights = x["stock_weights"].to_numpy().ravel()
    values = x.drop(columns=["stock_weights"]).to_numpy().ravel()

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

def get_weighted_std(x: pd.Series):

    weights = x["stock_weights"].to_numpy().ravel()
    values = x.drop(columns=["stock_weights"]).to_numpy().ravel()

    return weighted_std(values, weights)

#==============================================================================

# Function to do the feature engineering
def feature_engineering(
    df: pd.DataFrame, 
    weights: Mapping[int, Any] = None,
    clusters: Mapping[int, Any] = None,
    revealed_target: bool = True, 
    reduce_memory: bool = True
) -> pd.DataFrame:

    ########################
    # Time features
    ########################
    logger.info("Generating time features...")
    df = generate_time_features(df)

    ########################
    # Size-only features
    ########################
    logger.info("Generating size features...")
    df = generate_size_features(df)

    ########################
    # Price-only features
    ########################
    logger.info("Generating price features...")
    df = generate_price_features(df)

    ########################
    # Mixed features
    ########################
    logger.info("Generating mixed features...")
    df = generate_mixed_features(df)

    if reduce_memory:
        logger.info("Reducing data memory footprint...")
        downcast(df)

    ########################
    # Rolling features
    ########################
    logger.info("Generating rolling features...")
    df = generate_rolling_features(df, revealed_target)

    if reduce_memory:
        logger.info("Reducing data memory footprint...")
        downcast(df)

    ########################
    # Aggregated features
    ########################
    logger.info("Generating aggregated features...")
    df = generate_aggregated_features(df, weights, clusters)


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

def generate_size_features(df: pd.DataFrame) -> pd.DataFrame:
    # Calculate size-related features
    """
    df["ask_size.bid_size.total"] = df["ask_size"] + df["bid_size"]
    df["ask_size.bid_size.spread"] = df["ask_size"] - df["bid_size"]
    df["ask_size.bid_size.imbalance"] = (df["ask_size"] - df["bid_size"]) / (df["bid_size"] + df["ask_size"])
    df["imbalance_size.matched_size.imbalance"] = (df["imbalance_size"] - df["matched_size"]) / (df["matched_size"] + df["imbalance_size"])
    df["bid_size.ask_size.ratio"] = df["bid_size"] / df["ask_size"]
    df["imbalance_size.matched_size.ratio"] = df["imbalance_size"] / df["matched_size"]
    """

    for c in combinations(["ask_size", "bid_size"], 2):
        df[f"{c[0]}.{c[1]}.spread"] = df[c[0]] - df[c[1]]
        df[f"{c[0]}.{c[1]}.total"] = df[c[0]] + df[c[1]]
        df[f"{c[0]}.{c[1]}.ratio"] = df[c[0]] / df[c[1]]
        df[f"{c[0]}.{c[1]}.imbalance"] = (df[c[0]] - df[c[1]])/(df[c[0]] + df[c[1]])

    for c in combinations(["imbalance_size", "matched_size"], 2):
        df[f"{c[0]}.{c[1]}.spread"] = df[c[0]] - df[c[1]]
        df[f"{c[0]}.{c[1]}.total"] = df[c[0]] + df[c[1]]
        df[f"{c[0]}.{c[1]}.ratio"] = df[c[0]] / df[c[1]]
        df[f"{c[0]}.{c[1]}.imbalance"] = (df[c[0]] - df[c[1]])/(df[c[0]] + df[c[1]])

    # Calculate triplet imbalance features using the Numba-optimized function
    sizes = ["ask_size", "bid_size", "imbalance_size", "matched_size"]
    for c in [sizes]:
        triplet_feature = calculate_triplet_imbalance_numba(c, df)
        df[triplet_feature.columns] = triplet_feature.values

    return df

#==============================================================================

def generate_price_features(df: pd.DataFrame) -> pd.DataFrame:

    # Calculate price-related features
    df["ask_price.bid_price.mean"] = (df["ask_price"] + df["bid_price"]) / 2

    # Create features for pairwise price imbalances
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    for c in combinations(prices, 2):
        df[f"{c[0]}.{c[1]}.spread"] = df[c[0]] - df[c[1]]
        df[f"{c[0]}.{c[1]}.imbalance"] = (df[c[0]] - df[c[1]])/(df[c[0]] + df[c[1]])

    # Calculate triplet imbalance features using the Numba-optimized function
    prices = ["ask_price", "bid_price", "wap", "reference_price"]
    for c in [prices]:
        triplet_feature = calculate_triplet_imbalance_numba(c, df)
        df[triplet_feature.columns] = triplet_feature.values

    return df

#==============================================================================

def generate_time_features(df: pd.DataFrame) -> pd.DataFrame:

    #df["dow"] = df["date_id"] % 5  # Day of the week
    #df["seconds"] = df["seconds_in_bucket"] % 60  # Seconds
    #df["minute"] = df["seconds_in_bucket"] // 60  # Minutes

    return df

#==============================================================================

def generate_mixed_features(df: pd.DataFrame) -> pd.DataFrame:


    rolling_sample = df.groupby(["stock_id", "date_id"])

    # Calculate additional features
    df["prices.spread.imbalance_size.bid_price.product"] = df["ask_price.bid_price.spread"] * df["imbalance_size"] 
    df["prices.spread.sizes.imbalance.product"] = df["ask_price.bid_price.spread"] * df["ask_size.bid_size.imbalance"]
    df["prices.imbalance.sizes.spread.product"] = df["ask_size.bid_size.imbalance"] * df["ask_price.bid_price.spread"]
    df["prices.spread.product"] = df["ask_price.bid_price.spread"] * df["far_price.near_price.spread"]

    #df["imbalance_buy_sell_flag"] += 1

    #df["ask_price.ask_size.product"] = df["ask_price"] * df["ask_size"]
    #df["bid_price.bid_size.product"] = df["bid_price"] * df["bid_size"]
    #df["wap.size.product"] = df["wap"] * df["ask_size.bid_size.total"]
    #df["reference_price.size.product"] = df["reference_price"] * df["ask_size.bid_size.total"]
    #df["far_price.ask_size.product"] = df["far_price"] * df["ask_size"]
    #df["near_price.bid_size.product"] = df["near_price"] * df["bid_size"]


    return df

#==============================================================================

def generate_rolling_features(df: pd.DataFrame, revealed_target: bool = True) -> pd.DataFrame:


    engine = "numba"
    rolling_sample = df.groupby(["stock_id", "date_id"])



    # Rolling averages
    """
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    for price in prices:
        for window in [4, 8, 10]:
            df[f"{price}.sma.{window}"] = rolling_sample[col].rolling(window).mean(engine=engine)
    """

    # Add revealed target
    if revealed_target:
        target_sample = df.groupby(["stock_id", "seconds_in_bucket"])
        df[f"revealed_target"] = target_sample["target"].shift(1)

    for col in ["matched_size", "imbalance_size", "reference_price", "wap", "imbalance_buy_sell_flag"]:
        for window in [1, 2, 3, 5, 10]:
            df[f"{col}.shift.{window}"] = rolling_sample[col].shift(window)
            df[f"{col}.pct_change.{window}"] = rolling_sample[col].pct_change(window)

    for col in ["ask_price", "bid_price", "ask_size", "bid_size", "ask_size.bid_size.ratio",
                "prices.imbalance.sizes.spread.product", "ask_price.bid_price.spread", "reference_price.wap.spread"]:
        for window in [1, 2, 3, 5, 10]:
            df[f"{col}.diff.{window}"] = rolling_sample[col].diff(window)

    df["imbalance_size.diff.matched_size.ratio"] = rolling_sample["imbalance_size"].diff() / df["matched_size"]
    #df[f"wap.pct_change.1"] = rolling_sample["wap"].pct_change(1)

    return df

#==============================================================================

def apply_aggregations(
    aggr_index: List[str],
    aggr_mappings: Mapping[str, Any],
    df: pd.DataFrame,
    suffix: str = "aggr"
) -> pd.DataFrame:

    aggr_df = None
    # Perform aggregations and features dependent on these
    for key, value in aggr_mappings.items():
        aggr_colum = pd.Series(value, name=key + "." + suffix)
        aggr_colum.index = aggr_colum.index.set_names(aggr_index)
    
        if aggr_df is None:
            aggr_df = pd.DataFrame(aggr_colum)
        else:
            aggr_df[key + "." + suffix] = aggr_colum

    df = pd.merge(df, aggr_df, on=aggr_index, how="left")

    return df

#==============================================================================


def generate_aggregated_features(df: pd.DataFrame, weights: Mapping[int, Any] = None, clusters: Mapping[int, Any] = None) -> None:

    apply_engine = "numba"

    ###########################################
    # Prize and size aggregations
    ###########################################

    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]
    # Calculate various statistical aggregation features
    for func in ["mean", "std", "skew", "kurt"]:
        df[f"all_prices.{func}"] = df[prices].agg(func, axis=1)
        df[f"all_sizes.{func}"] = df[sizes].agg(func, axis=1)

    sizes = ["ask_size", "bid_size", "imbalance_size", "matched_size", 
        "prices.imbalance.sizes.spread.product"]
    prices = ["reference_price", "ask_price", "bid_price", "wap", 
        "ask_price.bid_price.spread", "reference_price.wap.spread"]

    if weights is not None:
        df["stock_weights"] = df["stock_id"].map(weights)
    if clusters is not None:
        df["stock_clusters"] = df["stock_id"].map(clusters)

    ###########################################
    # Cluster aggregations (by industries)
    ###########################################
    if clusters is not None:
        cluster_aggregations = {}
        cluster_indx = ["date_id", "seconds_in_bucket", "stock_clusters"]
        cluster_sample = df.groupby(cluster_indx)


    ###########################################
    # Cluster aggregations (by industries)
    ###########################################
    stock_aggregations = {}
    stock_indx = ["date_id", "seconds_in_bucket"]
    stock_sample = df.groupby(stock_indx)


    for elem in prices + sizes:
        if weights is not None:
            stock_aggregations[elem + ".mean.weighted"] = stock_sample[[elem, "stock_weights"]].apply(get_weighted_avg).to_dict()
            stock_aggregations[elem + ".std.weighted"] = stock_sample[[elem, "stock_weights"]].apply(get_weighted_std).to_dict()
            df[elem + ".weighted"] = df["stock_weights"] * df[elem]

        else:
            stock_aggregations[elem + ".mean"] = stock_sample[elem].mean(engine=apply_engine).to_dict()
            stock_aggregations[elem + ".std"] = stock_sample[elem].std(engine=apply_engine).to_dict()

        if clusters is not None:
            cluster_aggregations[elem + ".mean"] = cluster_sample[elem].mean(engine=apply_engine).to_dict()
            cluster_aggregations[elem + ".std" ] = cluster_sample[elem].std(engine=apply_engine).to_dict()

    if weights is not None:
        df["wap.weighted.momentum"] = df.groupby(["stock_id", "date_id"])["wap.weighted"].pct_change(periods=6)
    else:
        df["wap.momentum"] = df.groupby(["stock_id", "date_id"])["wap"].pct_change(periods=6)

    #if clusters is not None:
    #    df = apply_aggregations(cluster_indx, cluster_aggregations, df, suffix="cluster_aggr")
    df = apply_aggregations(stock_indx, stock_aggregations, df, suffix="stock_aggr")

    return df


#==============================================================================

def generate_global_features(df: pd.DataFrame) -> None:

    apply_engine = "numba"

    global_stock_id_feats = {
        "size.median": df.groupby("stock_id")["bid_size"].median() + df.groupby("stock_id")["ask_size"].median(),
        "size.std": df.groupby("stock_id")["bid_size"].std(engine=apply_engine) + df.groupby("stock_id")["ask_size"].std(engine=apply_engine),
        "size.ptp": df.groupby("stock_id")["bid_size"].max(engine=apply_engine) - df.groupby("stock_id")["bid_size"].min(engine=apply_engine),
        "price.median": df.groupby("stock_id")["bid_price"].median() + df.groupby("stock_id")["ask_price"].median(),
        "price.std": df.groupby("stock_id")["bid_price"].std(engine=apply_engine) + df.groupby("stock_id")["ask_price"].std(engine=apply_engine),
        "price.ptp": df.groupby("stock_id")["bid_price"].max(engine=apply_engine) - df.groupby("stock_id")["ask_price"].min(engine=apply_engine),
    }

    return global_stock_id_feats


#==============================================================================

def apply_global_features(df: pd.DataFrame, global_stock_id_feats: Mapping[str, Any]) -> None:

    logger.info("Applying global features...")

    for key, value in global_stock_id_feats.items():
        df[f"{key}.global"] = df["stock_id"].map(value.to_dict())

    return df

#==============================================================================

def select_features(df: pd.DataFrame, features: List[str] = None, reduce_memory: bool = True) -> None:

    if features is not None:
        logger.info("Dropping unnecesary features...")
        df.drop(columns=[elem for elem in df.columns if elem not in features], inplace=True)

    if reduce_memory:
        logger.info("Reducing data memory footprint...")
        downcast(df)

    return df

#==============================================================================

def clean_dataset(df: pd.DataFrame) -> None:

    logger.info("Cleaning dataset...")

    # Drop all nans rows in target 
    df.dropna(subset="target", inplace=True)

    # Reset infex in-place
    df.reset_index(drop=True, inplace=True)

#==============================================================================
