import numpy as np
import pandas as pd
from numba import njit

#==============================================================================

@njit
def cube(x):
    if x >= 0:
        return x**(1/3)
    elif x < 0:
        return -(abs(x)**(1/3))


#==============================================================================

@njit
def pow_order(x,n):
    if x >= 0:
        return x**(n)
    elif x < 0:
        return -(abs(x)**(n))
    
    
#==============================================================================


def cubic_mean(array_results):
    array_results = array_results.copy()
    columns = [i for i in range(len(array_results))]
    df_results=pd.DataFrame(np.vstack(array_results).T)
    
    df_results.columns = columns.copy()
    cubic_cols=[]
    for col in columns:
        df_results[f'{col}_3']=df_results[col]**3
        cubic_cols.append(f'{col}_3')

    return df_results[cubic_cols].mean(axis=1).apply(cube).values

#==============================================================================

def pow_mean(array_results, order=0):
    array_results = array_results.copy()
    columns = [i for i in range(len(array_results))]
    df_results=pd.DataFrame(np.vstack(array_results).T)

    df_results.columns = columns.copy()
    pow_cols = []
    for col in columns:
        df_results[f"{col}_{order}"] = df_results[col].apply(lambda x: pow_order(x, order))
        pow_cols.append(f"{col}_{order}")
        
    return df_results[pow_cols].mean(axis=1).apply(lambda x: pow_order(x, 1./order)).values

#==============================================================================