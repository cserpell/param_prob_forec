# coding=utf-8
"""Base abstract time series forecast model."""
import abc
import collections
import datetime
import inspect

import joblib
import numpy as np

from modules import data
from modules import preprocessor
from modules import util


ModelOptions = collections.namedtuple('ModelOptions', [
    'preprocess',
    'sequential_mini_step',
    'arima_p',
    'arima_d',
    'arima_q',
    'nn_batch_size',
    'nn_dropout_no_mc',
    'nn_dropout_output',
    'nn_dropout_recurrence',
    'nn_epochs',
    'nn_l2_regularizer',
    'nn_learning_rate',
    'nn_optimizer',
    'nn_output_distribution',
    'nn_patience',
    'nn_tensorboard',
    'nn_use_variable_sigma',
    'flow_steps',
    'flow_temperature_max',
    'flow_temperature_min',
    'flow_temperature_steps',
    'flow_use_temperature',
    'lstm_layers',
    'lstm_nodes',
    'lstm_stateful',
    'use_dates',
])


def build_options(model_options):
    """Builds a ModelOptions object, filtering correct parameters."""
    valid_keys = inspect.getfullargspec(ModelOptions).args
    return ModelOptions(**{key: value for key, value in model_options.items()
                           if key in valid_keys})


def get_date_features(all_datetimes):
    """Get date features from numpy datetime64 objects."""
    return [[one_datetime.year,
             one_datetime.month,
             one_datetime.day,
             one_datetime.hour,
             one_datetime.weekday(),
             one_datetime.weekday() in data.WEEKEND_DAYS]
            for one_datetime in all_datetimes]


class Model(abc.ABC):
    """Base abstract class with models to forecast time series."""

    def __init__(self, model_options=None):
        """Constructor of Model."""
        super(Model, self).__init__()
        self._options = model_options
        self._output_shape = None
        self._preprocess_in = None
        self._preprocess_kind = (
            preprocessor.Preprocess[self._options.preprocess.upper()]
            if self._options else None)
        self._preprocess_out = None

    def _preprocess_active(self):
        """Returns whether preprocess is needed or not."""
        return bool(self._preprocess_kind != preprocessor.Preprocess.NONE)

    def _train_preprocess_one(self, one_array, use_dates=False):
        """Train preprocessing of one array."""
        preprocess = None
        if self._preprocess_active():
            dates_array = None
            if use_dates:
                # If dates are active, do not preprocess them
                dates_array = one_array[:, :, :, -6:]
                one_array = one_array[:, :, :, :-6]
            # Preprocess requires (samples, features) input shape.
            # Transform inplace, instead of using reshape, for RAM usage
            original_array = one_array  # Pointer copy, not deep copy
            original_shape = one_array.shape
            original_array.shape = (original_shape[0],
                                    util.prod(original_shape[1:]))
            preprocess = preprocessor.build_preprocess(self._preprocess_kind)
            one_array = preprocess.fit_transform(original_array)
            # Note that input one_array has to be reshaped back to what it was
            # when received by the method, because the reshape was inplace
            original_array.shape = original_shape
            one_array.shape = original_shape
            if use_dates:
                one_array = np.concatenate([one_array, dates_array], axis=-1)
        return one_array, preprocess

    def _train_preprocess(self, x_train, y_train):
        """Train preprocessing step of models."""
        x_train, self._preprocess_in = self._train_preprocess_one(
            x_train, use_dates=self._options.use_dates)
        y_train, self._preprocess_out = self._train_preprocess_one(y_train)
        return x_train, y_train

    def _apply_preprocess(self, x_test, preprocessor_inst, use_dates=False):
        """Apply preprocessing to input when forecasting."""
        if self._preprocess_active():
            dates_array = None
            if use_dates:
                # If dates are active, do not preprocess them
                dates_array = x_test[:, :, :, -6:]
                x_test = x_test[:, :, :, :-6]
            # Transform inplace, instead of using reshape, for RAM usage
            original_x_test = x_test  # Pointer copy, not deep copy
            original_shape = x_test.shape
            original_x_test.shape = (original_shape[0],
                                     util.prod(original_shape[1:]))
            x_test = preprocessor_inst.transform(original_x_test)
            # Note that input x_test has to be reshaped back to what it was
            # when received by the method, because the reshape was inplace
            original_x_test.shape = original_shape
            x_test.shape = original_shape
            if use_dates:
                x_test = np.concatenate([x_test, dates_array], axis=-1)
        return x_test

    def _inverse_preprocess(self, y_test):
        """Apply inverse preprocessing for output."""
        if self._preprocess_active():
            # Transform inplace, instead of using reshape, for RAM usage
            original_y_test = y_test  # Pointer copy, not deep copy
            original_shape = y_test.shape
            original_y_test.shape = (original_shape[0],
                                     util.prod(original_shape[1:]))
            y_test = self._preprocess_out.inverse_transform(original_y_test)
            # Note that input y_test has to be reshaped back to what it was
            # when received by the method, because the reshape was inplace
            original_y_test.shape = original_shape
            y_test.shape = original_shape
        return y_test

    @abc.abstractmethod
    def _internal_predict(self, x_test):
        """Predict from given data using model internally."""

    @abc.abstractmethod
    def _internal_fit(self, x_train, y_train, validation_x=None,
                      validation_y=None):
        """Fit internal model with given data."""

    def save(self, file_name):
        """Save model in given file name."""
        joblib.dump(
            [self._options, self._output_shape,
             self._preprocess_in, self._preprocess_out],
            '{}_meta'.format(file_name))

    def load(self, file_name):
        """Load model from given file name."""
        read_list = joblib.load('{}_meta'.format(file_name))
        self._options = build_options(read_list[0])
        self._output_shape = read_list[1]
        self._preprocess_in = read_list[2]
        self._preprocess_kind = preprocessor.Preprocess[
            self._options.preprocess.upper()]
        self._preprocess_out = read_list[3]

    def predict(self, x_test, samples=1):
        """Predict from given data using model."""
        x_test_shape = x_test.shape
        pre_x_test = self._apply_preprocess(
            x_test, self._preprocess_in, use_dates=self._options.use_dates)
        return_value = np.stack([
            self._inverse_preprocess(self._fill_multi_step(pre_x_test))
            for _ in range(samples)])
        x_test.shape = x_test_shape  # In case pre_x_test is the same x_test
        return return_value

    def fit(self, x_train, y_train, validation_x=None, validation_y=None):
        """Main preprocessing and fit procedure, with given data."""
        validation_x_shape = None
        validation_y_shape = None
        x_train_shape = x_train.shape
        pre_x_train, pre_y_train = self._train_preprocess(x_train, y_train)
        if validation_x is not None and validation_y is not None:
            # If validation set is available, apply preprocessing too
            validation_x_shape = validation_x.shape
            validation_x = self._apply_preprocess(
                validation_x, self._preprocess_in,
                use_dates=self._options.use_dates)
            validation_y_shape = validation_y.shape
            validation_y = self._apply_preprocess(
                validation_y, self._preprocess_out)
        # Here we are modelling previous lags as input features:
        # (number of samples, time steps, number of series)
        self._output_shape = (pre_y_train.shape[1], pre_y_train.shape[2])
        self._internal_fit(pre_x_train, pre_y_train, validation_x=validation_x,
                           validation_y=validation_y)
        if validation_x is not None:
            validation_x.shape = validation_x_shape
            validation_y.shape = validation_y_shape
        x_train.shape = x_train_shape  # In case pre_x_train is same x_train

    def _reset_internal_states(self, x_test):
        """Called before any prediction."""

    def _fill_multi_step(self, x_test):
        """Predict multi steps using current model."""
        # Multi step predictions are performed repeating some step predictions
        # using intermediate outputs as inputs for following predictions
        final_out = None
        self._reset_internal_states(x_test)
        # Total steps that are needed in output
        total_steps = self._output_shape[0]
        timedelta_step = datetime.timedelta(
            hours=1) if self._options.use_dates else None
        while final_out is None or final_out.shape[1] < total_steps:
            next_out = self._internal_predict(x_test)
            final_out = (next_out if final_out is None else
                         np.concatenate([final_out, next_out], axis=1))
            if final_out.shape[1] < total_steps:
                # Mini steps predictions are active, so output is injected as
                # input again. We modify x_test with new data.
                # Note there is a big assumption here: preprocessing of input
                # is same as preprocessing of output (for example standard).
                # This is not actually true, as we are using a different
                # scaler object for each
                # if next_out.shape[-1] != x_test.shape[-1]:
                #     logging.error('Output time step injecting works only '
                #                   'for same number of input and output '
                #                   'time series and no date features!')
                #     sys.exit(1)
                last_datetime = None
                if self._options.use_dates:
                    # Date features have to be readded when reinjecting
                    # There are 6 date features year, month, day, hour, ...
                    last_date = x_test[:, -1, -1, -6:]
                    last_datetime = [datetime.datetime(
                        int(row[0]), int(row[1]), int(row[2]),
                        hour=int(row[3])) for row in last_date]
                # Iterate over each predicted step.
                # next_out shape is (items, steps, output variables)
                for pos in range(next_out.shape[1]):
                    step = next_out[:, pos:(pos + 1), :]
                    step = step.reshape((step.shape[0], step.shape[1], 1,
                                         step.shape[2]))
                    if self._options.use_dates:
                        # Get next datetimes
                        last_datetime = [one_datetime + timedelta_step
                                         for one_datetime in last_datetime]
                        date_features = get_date_features(last_datetime)
                        dates = np.array(date_features).reshape(
                            (len(date_features), 1, 1, len(date_features[0])))
                        # TODO: CHECK: Next line should not be necessary...
                        dates = dates.repeat(step.shape[1], axis=1)
                        step = np.concatenate([step, dates], axis=3)
                    step = np.concatenate([x_test[:, -1:, 1:, :],
                                           step], axis=2)
                    x_test = np.concatenate([x_test[:, 1:, :, :],
                                             step], axis=1)
        if final_out.shape[1] > total_steps:
            final_out = final_out[:, :total_steps, :]
        return final_out

    def stopped_epoch(self):
        """Get number of epochs elapsed until end of training."""
        return 0
