import pandas as pd

try:
    from loguru import logger
except:
    import logging
    logger = logging.getLogger("__main__")


#==============================================================================

def downcast(df):
    '''This function downcasts the elements in a dataframe to reduce the memory footprint'''

    start_mem = df.memory_usage().sum() / 1024**2
    logger.info(f"Memory usage of dataframe is {start_mem:.2f} MB")

    fcols = df.select_dtypes("float").columns
    icols = df.select_dtypes("integer").columns

    df[fcols] = df[fcols].apply(pd.to_numeric, downcast="float")
    df[icols] = df[icols].apply(pd.to_numeric, downcast="integer")

    end_mem = df.memory_usage().sum() / 1024**2
    logger.info(f"Memory usage after optimization is: {end_mem:.2f} MB")
    decrease = 100 * (start_mem - end_mem) / start_mem
    logger.info(f"Decreased by {decrease:.2f}%")


#==============================================================================