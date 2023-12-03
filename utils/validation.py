import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from timeit import default_timer as timer
import random

try:
    from loguru import logger
except:
    import logging
    logger = logging.getLogger("__main__")

from .files import save_model
from .ensembling import pow_mean

#==============================================================================

class PurgedEmbargoSplit(object):

    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 5,
        *args,
        **kwargs
    ):
    
        self.n_splits = n_splits
        self._gap = gap


    def split(
        self,
        date_id,
        *args, 
        **kwargs
    ):

        fold_size = date_id.unique().shape[0] // self.n_splits

        for i in range(1):
            i = self.n_splits - 1
            start = i * fold_size
            end = start + fold_size
            if i < self.n_splits - 1:  # No need to purge after the last fold
                purged_start = end - 2
                purged_end = end + self._gap + 2
                train_indices = (date_id >= start) & (date_id < purged_start) | (date_id > purged_end)
            else:
                train_indices = (date_id >= start) & (date_id < end)
            
            test_indices = (date_id >= end) & (date_id < end + fold_size)

            train_indices = train_indices.index[train_indices]
            test_indices = test_indices.index[test_indices]

            yield train_indices, test_indices


#==============================================================================

class TrainTestSplit(object):

    def __init__(
        self,
        test_size: int,
        by_date_mode: bool = True,
        n_splits: int = 1,
        *args,
        **kwargs
    ):
    
        self._test_size = test_size
        self.n_splits = n_splits

        self._by_date_mode = by_date_mode


    def split(
        self,
        date_id,
        *args, 
        **kwargs
    ):

        n_samples = date_id.shape[0]

        if self._by_date_mode:
            n_train = date_id.loc[date_id <= date_id.max() - self._test_size].shape[0]
        else:
            n_train = n_samples - int(self._test_size * n_samples)

        n_test = n_samples - n_train

        train = np.arange(n_train)
        test = np.arange(n_train, n_train + n_test)
    
        for _ in range(self.n_splits):
            yield train, test 


#==============================================================================


def cross_validate(
        model_type, 
        model_params, 
        x, 
        y, 
        date_id=None, 
        cv=TimeSeriesSplit(), 
        scorer=mean_absolute_error, 
        groups=None, 
        job_path=None, 
        power=None,
        *args, 
        **kwargs
    ):

    scores = np.zeros(cv.n_splits)
    seed = model_params.get("random_seed", 1020)
    random.seed(seed)

    models = []
    logger.info(f"Starting evaluation...")
    logger.info("=" * 30)

    y_preds = []
    for i, (train_index, val_index) in enumerate(cv.split(date_id=date_id, groups=groups)):

        x_train, x_val = x.iloc[train_index], x.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        if "sample_weight" in kwargs.keys():
            kwargs["sample_weight"] = kwargs["sample_weight"][train_index]

        model = model_type(**model_params)
        random_state = random.randint(1, 9999)
        model.set_params(random_state=random_state)

        logger.info(f"Training model with seed {random_state}")

        eval_set = None
        if x_val.shape[0] > 0:
            eval_set = [(x_val, y_val)]

        start = timer()

        model.fit(x_train, y_train, eval_set=eval_set, *args, **kwargs)
        end = timer()
        
        models.append(model)

        if eval_set is not None:
            y_pred = model.predict(x_val)
            scores[i] = scorer(y_pred, y_val)
            y_preds.append(y_pred)

            logger.info(f"Fold {i + 1}: {scores[i]:.4f} (took {end - start:.2f}s)")

        if job_path is not None:
            model_path = job_path.joinpath("models")
            model_path.mkdir(exist_ok=True, parents=True)
            save_model(model, file=model_path.joinpath("model-" + str(i + 1).zfill(2) + ".pkl"))

    logger.info("-" * 30)
    logger.success(f"Average MAE = {scores.mean():.4f} ± {scores.std():.4f}")
    if power is not None:
        y_pred = pow_mean(y_preds, power)
        score = scorer(y_pred, y_val)
        logger.success(f"Ensembling MAE = {score:.4f}")
    logger.info("=" * 30)
    
    return scores, models

#==============================================================================