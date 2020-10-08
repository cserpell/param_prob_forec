# coding=utf-8
"""Persistence model."""
import logging
import sys

import numpy as np

from models import model


class Persistence(model.Model):
    """Persistence model to forecast time series."""

    def _check_sizes(self, x_data):
        """Check that input and output sizes match."""
        if self._output_shape[1] != x_data.shape[-1]:
            logging.error('Persistence model needs same number of input and '
                          'output time series, and no date features!')
            sys.exit(1)

    def _internal_predict(self, x_test):
        """Predict from given data using model internally."""
        # Remember input x shape is (samples, steps, lags, series), and the
        # output has shape (samples, steps, series). We just repeat last value.
        # Note we assume same preprocess in output and input. That is not true,
        # but we will asume so. Also we assume predicting same input series
        self._check_sizes(x_test)
        return np.stack([x_test[:, -1, -1, :]] * self._output_shape[0], axis=1)

    def _internal_fit(self, x_train, y_train, validation_x=None,
                      validation_y=None):
        """Fit internal model with given data."""


class StochasticPersistence(Persistence):
    """Stochastic persistence model to forecast time series."""

    def __init__(self, model_options=None):
        """Constructor of StochasticPersistence."""
        super(StochasticPersistence, self).__init__(
            model_options=model_options)
        self._std = None

    def _internal_predict(self, x_test):
        """Predict from given data using model internally."""
        # Remember input x shape is (samples, steps, lags, series), and the
        # output has shape (samples, steps, series). We just repeat last value.
        # Note we assume same preprocess in output and input. That is not true,
        # but we will asume so. Also we assume predicting same input series
        self._check_sizes(x_test)
        # Here we sample from estimated normal distributions
        prediction = None
        for _ in range(self._std.shape[0]):
            last = (x_test[:, -1, -1, :] if prediction is None else
                    prediction[:, -1, :])
            next_pred = np.random.normal(loc=last, scale=self._std[0, :])
            next_pred.shape = (next_pred.shape[0], 1, next_pred.shape[1])
            prediction = (next_pred if prediction is None else
                          np.concatenate([prediction, next_pred], axis=1))
        return prediction

    def _internal_fit(self, x_train, y_train, validation_x=None,
                      validation_y=None):
        """Fit internal model with given data."""
        # Shapes are:
        # x_train (samples, steps, lags, series)
        # y_train (samples, steps, series)
        self._check_sizes(x_train)
        # We perform prediction without uncertainty and get std from residuals
        prediction = np.stack([x_train[:, -1, -1, :]] * self._output_shape[0],
                              axis=1)
        # Note we assume same preprocess in output and input. That is not true,
        # but we will asume so. Also we assume predicting same input series
        self._std = np.std(prediction - y_train, axis=0)
