import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

#==============================================================================

def symmetric_mean_absolute_percentage_error(A, F):
    '''Calculate symmetric MAPE metric'''
    tmp = 2 * np.abs(F - A) / (np.abs(A) + np.abs(F))
    len_ = np.count_nonzero(~np.isnan(tmp))
    if len_ == 0 and np.nansum(tmp) == 0: # Deals with a special case
        return 100
    return 100 / len_ * np.nansum(tmp)

#==============================================================================

def calculate_metrics(model, x, y_gt):
    '''Get model evaluation metrics on the test set.'''

    # Get model predictions
    y_pred= model.predict(x)
    
    # Calculate evaluation metrics for assesing performance of the model.
    mae = mean_absolute_error(y_gt, y_pred)
    mse = mean_squared_error(y_gt, y_pred)
    mape = mean_absolute_percentage_error(y_gt, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_gt, y_pred)
    
    return mae, mse, mape, smape

#==============================================================================