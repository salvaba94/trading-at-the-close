import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from timeit import default_timer as timer

try:
    from loguru import logger
except:
    import logging
    logger = logging.getLogger("__main__")

from .files import save_model



class TrainTestSplit(object):

    def __init__(
        self,
        test_size: float,
        *args,
        **kwargs
    ):

        self._test_size = test_size
        self.n_splits = 1


    def split(
        self,
        data, 
        *args, 
        **kwargs
    ):

        n_samples = data.shape[0]
        n_train = n_samples - int(self._test_size * n_samples)
        n_test = n_samples - n_train

        train = np.arange(n_train)
        test = np.arange(n_train, n_train + n_test)
    
        yield train, test 




def cross_validate(model, x, y, cv=TimeSeriesSplit(), scorer=mean_absolute_error, categorical_feature="auto", groups=None, callbacks=None, job_path=None):

    scores = np.zeros(cv.n_splits)
    
    models = []
    logger.info(f"Starting evaluation...")
    logger.info("=" * 30)
    for i, (train_index, val_index) in enumerate(cv.split(x, groups=groups)):
        
        x_train, x_val = x.iloc[train_index], x.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        start = timer()
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], categorical_feature=categorical_feature, callbacks=callbacks)
        end = timer()
        
        models.append(model)

        y_pred = model.predict(x_val)
        scores[i] = scorer(y_pred, y_val)

        logger.info(f"Fold {i + 1}: {scores[i]:.4f} (took {end - start:.2f}s)")

        if job_path is not None:
            model_path = job_path.joinpath("models")
            model_path.mkdir(exist_ok=True, parents=True)
            save_model(model, file=model_path.joinpath("model-" + str(i + 1).zfill(2) + ".pkl"))

    logger.info("-" * 30)
    logger.success(f"Average MAE = {scores.mean():.4f} ± {scores.std():.2f}")
    logger.info("=" * 30)
    
    return scores, models