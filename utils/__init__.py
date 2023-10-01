
import os

if os.environ["LOCAL_MOCKAPI"]:
    from .public_timeseries_testing_util_local import MockApi
else:
    from .public_timeseries_testing_util import MockApi

__all__ = ["MockApi"]
