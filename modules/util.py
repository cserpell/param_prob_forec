# coding=utf-8
"""Util module with helper methods for time series handling."""
import contextlib
import functools
import logging
import operator
import os
import time

import joblib
from tensorflow.keras import backend
from tensorflow.keras import losses


_GLOBAL_TIMES = {}


def prod(iterable):
    """Gets product of all items in an iterable."""
    return functools.reduce(operator.mul, iterable, 1)


def mean_squared_loss_with_sigma(y_true, y_pred):
    """Mean squared error plus log(sigma^2), assuming shape x 2 width."""
    return backend.mean(
        y_pred[:, 1] + backend.square(y_pred[:, 0] - y_true[:, 0]) /
        backend.exp(y_pred[:, 1]), axis=-1)


def sigma_mean_squared_error(y_true, y_pred):
    """Mean squared error for variable sigma output."""
    return losses.mean_squared_error(y_true[:, 0], y_pred[:, 0])


def sigma_mean_absolute_error(y_true, y_pred):
    """Mean squared error for variable sigma output."""
    return losses.mean_absolute_error(y_true[:, 0], y_pred[:, 0])


def sigma_mean_absolute_percentage_error(y_true, y_pred):
    """Mean squared error for variable sigma output."""
    return losses.mean_absolute_percentage_error(y_true[:, 0], y_pred[:, 0])


@contextlib.contextmanager
def time_context(name):
    """Measure time in code in the context manager, and print given name."""
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    logging.info('%s finished in %s s', name, elapsed_time)
    _GLOBAL_TIMES[name] = elapsed_time


def load(directory, file_name):
    """Store all processed data set."""
    return joblib.load(os.path.join(directory, file_name))


def store(stored_object, file_name):
    """Store all processed data set."""
    joblib.dump(stored_object, file_name)


def get_all_times():
    """Return current times."""
    return _GLOBAL_TIMES.copy()
