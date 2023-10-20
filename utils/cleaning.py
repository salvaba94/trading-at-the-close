
import numpy as np
import pandas as pd
from typing import List

try:
    from loguru import logger
except:
    import logging
    logger = logging.getLogger("__main__")


def clean(df: pd.DataFrame) -> None:
    """This function cleans the dataset row-wise. Works fully in memory"""

    logger.info("Cleaning dataset...")

    df.dropna(subset="target", inplace=True)
    #df.replace([np.inf, -np.inf], np.nan, inplace=True)
    #df.fillna(method="ffill").fillna(0)
