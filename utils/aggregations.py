
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


def realized_volatility(series):
    return np.sqrt(np.sum(series**2))


def weighted_avg(values, weights):
    """
    Return the weighted average.

    values, weights -- NumPy ndarrays with the same shape.
    """

    if not isinstance(weights, np.ndarray):
        weights = np.array(weights)

    stock_id = values["stock_id"]
    values = values.drop(columns=["stock_id"])
    weights = weights[stock_id].reshape((-1, 1))

    average = np.average(values, weights=weights)

    return average


def weighted_std(values, weights):
    """
    Return the weighted standard deviation.

    values, weights -- NumPy ndarrays with the same shape.
    """

    if not isinstance(weights, np.ndarray):
        weights = np.array(weights)

    stock_id = values["stock_id"]
    values = values.drop(columns=["stock_id"])
    weights = weights[stock_id].reshape((-1, 1))

    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average)**2, weights=weights)
    return np.sqrt(variance)



def aggregate(df: pd.DataFrame, weights: List[float] = None):
    """This function aggregates features."""

    stock_aggregations = {
        #"median_total_size": df.groupby("stock_id")["total_size"].median().to_dict(),
        #"std_total_size": df.groupby("stock_id")["total_size"].std().to_dict(),
        #"max_total_size": df.groupby("stock_id")["total_size"].max().to_dict(),
        #"min_total_size": df.groupby("stock_id")["total_size"].min().to_dict(),
        #"mean_total_size": df.groupby("stock_id")["total_size"].mean().to_dict(),
        #"first_total_size": df.groupby("stock_id")["total_size"].first().to_dict(),
        #"last_total_size": df.groupby("stock_id")["total_size"].last().to_dict(),
    }

    date_aggregations = {
        #"sum_log_return_wap": df.groupby(["date_id", "seconds_in_bucket"])["log_return_wap"].sum().to_dict(),
        #"median_log_return_wap": df.groupby(["date_id", "seconds_in_bucket"])["log_return_wap"].median().to_dict(),
        #"std_log_return_wap": df.groupby(["date_id", "seconds_in_bucket"])["log_return_wap"].std().to_dict(),
        #"realized_return_wap": df.groupby(["date_id", "seconds_in_bucket"])["log_return_wap"].apply(realized_volatility).to_dict(),
        #"sum_log_return_ask_price": df.groupby(["date_id", "seconds_in_bucket"])["log_return_ask_price"].sum().to_dict(),
        #"median_log_return_ask_price": df.groupby(["date_id", "seconds_in_bucket"])["log_return_ask_price"].median().to_dict(),
        #"std_log_return_ask_price": df.groupby(["date_id", "seconds_in_bucket"])["log_return_ask_price"].std().to_dict(),
        #"realized_return_ask_price": df.groupby(["date_id", "seconds_in_bucket"])["log_return_ask_price"].apply(realized_volatility).to_dict(),
        #"sum_log_return_bid_price": df.groupby(["date_id", "seconds_in_bucket"])["log_return_bid_price"].sum().to_dict(),
        #"median_log_return_bid_price": df.groupby(["date_id", "seconds_in_bucket"])["log_return_bid_price"].median().to_dict(),
        #"std_log_return_bid_price": df.groupby(["date_id", "seconds_in_bucket"])["log_return_bid_price"].std().to_dict(),
        #"realized_return_bid_price": df.groupby(["date_id", "seconds_in_bucket"])["log_return_bid_price"].apply(realized_volatility).to_dict(),
        "sum_total_size": df.groupby(["date_id", "seconds_in_bucket"])["total_size"].sum().to_dict(),
        "median_total_size": df.groupby(["date_id", "seconds_in_bucket"])["total_size"].median().to_dict(),
        "std_total_size": df.groupby(["date_id", "seconds_in_bucket"])["total_size"].std().to_dict(),
        "sum_imb_s1": df.groupby(["date_id", "seconds_in_bucket"])["imb_s1"].sum().to_dict(),
        "median_imb_s1": df.groupby(["date_id", "seconds_in_bucket"])["imb_s1"].median().to_dict(),
        "std_total_imb_s1": df.groupby(["date_id", "seconds_in_bucket"])["imb_s1"].std().to_dict(),
        "sum_imb_s2": df.groupby(["date_id", "seconds_in_bucket"])["imb_s2"].sum().to_dict(),
        "median_imb_s2": df.groupby(["date_id", "seconds_in_bucket"])["imb_s2"].median().to_dict(),
        "std_total_imb_s2": df.groupby(["date_id", "seconds_in_bucket"])["imb_s2"].std().to_dict(),
    }

    if weights is not None:
        date_aggregations["weighted_mean_wap"] = df.groupby(["date_id", "seconds_in_bucket"])[["wap", "stock_id"]].apply(lambda x: weighted_avg(x, weights=weights)).to_dict()
        date_aggregations["weighted_std_wap"] = df.groupby(["date_id", "seconds_in_bucket"])[["wap", "stock_id"]].apply(lambda x: weighted_std(x, weights=weights)).to_dict()
        date_aggregations["weighted_mean_ask_price"] = df.groupby(["date_id", "seconds_in_bucket"])[["ask_price", "stock_id"]].apply(lambda x: weighted_avg(x, weights=weights)).to_dict()
        date_aggregations["weighted_std_ask_price"] = df.groupby(["date_id", "seconds_in_bucket"])[["ask_price", "stock_id"]].apply(lambda x: weighted_std(x, weights=weights)).to_dict()
        date_aggregations["weighted_mean_bid_price"] = df.groupby(["date_id", "seconds_in_bucket"])[["bid_price", "stock_id"]].apply(lambda x: weighted_avg(x, weights=weights)).to_dict()
        date_aggregations["weighted_std_bid_price"] = df.groupby(["date_id", "seconds_in_bucket"])[["bid_price", "stock_id"]].apply(lambda x: weighted_std(x, weights=weights)).to_dict()
    else:
        date_aggregations["mean_wap"] =  df.groupby(["date_id", "seconds_in_bucket"])["wap"].mean().to_dict()
        date_aggregations["std_wap"] = df.groupby(["date_id", "seconds_in_bucket"])["wap"].std().to_dict()
        date_aggregations["mean_ask_price"] = df.groupby(["date_id", "seconds_in_bucket"])["ask_price"].mean().to_dict()
        date_aggregations["std_ask_price"] = df.groupby(["date_id", "seconds_in_bucket"])["ask_price"].std().to_dict()
        date_aggregations["mean_bid_price"] = df.groupby(["date_id", "seconds_in_bucket"])["bid_price"].mean().to_dict()
        date_aggregations["std_bid_price"] = df.groupby(["date_id", "seconds_in_bucket"])["bid_price"].std().to_dict()

    return stock_aggregations, date_aggregations



def apply_aggregations(df: pd.DataFrame, aggregations_stock: Mapping[str, Any] = None, aggregations_date: Mapping[str, Any] = None, reduce_memory: bool = False):

    # Perform aggregations and features dependent on these
    if aggregations_stock is not None:
        logger.info("Applying stock-wise aggregations...")
        for key, value in aggregations_stock.items():
            df[key + "_stock_aggr"] = df["stock_id"].map(value)

    # Perform aggregations and features dependent on these
    if aggregations_date is not None:
        logger.info("Applying date-wise aggregations...")
        for key, value in aggregations_date.items():
            aggr_index = ["date_id", "seconds_in_bucket"]
            aggr_colum = pd.Series(value, name=key + "_date_aggr")
            aggr_colum.index = aggr_colum.index.set_names(aggr_index)
            df = df.join(aggr_colum, on=aggr_index)

    if reduce_memory:
        logger.info("Reducing data memory footprint...")
        downcast(df)

    return df
