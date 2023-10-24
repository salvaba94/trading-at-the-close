
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


def aggregate(df: pd.DataFrame, weights: List[float] = None):
    """This function aggregates features."""

    df_slice = df[[
        "stock_id",
        "date_id", 
        "seconds_in_bucket", 
        "wap", 
        "ask_price",
        "bid_price",
        #"log_return_wap",
        #"log_return_ask_price",
        #"log_return_bid_price",
        "total_size",
        "imb_s1",
        "imb_s2"
    ]]
    key = ""

    # Apply only to prices
    if weights is not None:
        key = "weighted_"
        df_slice["stock_weight"] = df_slice["stock_id"].map(pd.Series(weights).to_dict())
        df_slice["wap"] = df_slice["wap"].mul(df_slice["stock_weight"])
        df_slice["ask_price"] = df_slice["ask_price"].mul(df_slice["stock_weight"])
        df_slice["bid_price"] = df_slice["bid_price"].mul(df_slice["stock_weight"])


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
        key + "mean_wap": df_slice.groupby(["date_id", "seconds_in_bucket"])["wap"].mean().to_dict(),
        key + "std_wap": df_slice.groupby(["date_id", "seconds_in_bucket"])["wap"].std().to_dict(),
        key + "mean_ask_price": df_slice.groupby(["date_id", "seconds_in_bucket"])["ask_price"].mean().to_dict(),
        key + "std_ask_price": df_slice.groupby(["date_id", "seconds_in_bucket"])["ask_price"].std().to_dict(),
        key + "mean_bid_price": df_slice.groupby(["date_id", "seconds_in_bucket"])["bid_price"].mean().to_dict(),
        key + "std_bid_price": df_slice.groupby(["date_id", "seconds_in_bucket"])["bid_price"].std().to_dict(),
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
        "sum_total_size": df_slice.groupby(["date_id", "seconds_in_bucket"])["total_size"].sum().to_dict(),
        "median_total_size": df_slice.groupby(["date_id", "seconds_in_bucket"])["total_size"].median().to_dict(),
        "std_total_size": df_slice.groupby(["date_id", "seconds_in_bucket"])["total_size"].std().to_dict(),
        "sum_imb_s1": df_slice.groupby(["date_id", "seconds_in_bucket"])["imb_s1"].sum().to_dict(),
        "median_imb_s1": df_slice.groupby(["date_id", "seconds_in_bucket"])["imb_s1"].median().to_dict(),
        "std_total_imb_s1": df.groupby(["date_id", "seconds_in_bucket"])["imb_s1"].std().to_dict(),
        "sum_imb_s2": df_slice.groupby(["date_id", "seconds_in_bucket"])["imb_s2"].sum().to_dict(),
        "median_imb_s2": df_slice.groupby(["date_id", "seconds_in_bucket"])["imb_s2"].median().to_dict(),
        "std_total_imb_s2": df_slice.groupby(["date_id", "seconds_in_bucket"])["imb_s2"].std().to_dict(),
    }

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
            df.join(aggr_colum, on=aggr_index)

    if reduce_memory:
        logger.info("Reducing data memory footprint...")
        downcast(df)
