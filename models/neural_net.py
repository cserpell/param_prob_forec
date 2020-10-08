# coding=utf-8
"""Neural net base model."""
import logging
import sys

import joblib
import numpy as np
from tensorflow.keras import callbacks
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers

from models import model
from modules import util


class CustomStopper(callbacks.EarlyStopping):
    """Early stop that forces stopping only after number of epochs."""
    def __init__(self, monitor='val_loss', patience=100,
                 restore_best_weights=True, start_epoch=100):
        super(CustomStopper, self).__init__(
            monitor=monitor, patience=patience,
            restore_best_weights=restore_best_weights)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super(CustomStopper, self).on_epoch_end(epoch, logs=logs)


class NeuralNet(model.Model):
    """Neural net base model to forecast time series."""

    def __init__(self, model_options=None):
        """Constructor of NeuralNet."""
        super(NeuralNet, self).__init__(model_options=model_options)
        self._inference_batch_size = None
        self._inference_model = None
        self._model = None
        self._stateful_model = None
        self._stateful_model_samples = None
        self._stopped_epoch = None

    def _l2_regularization(self):
        """Gets L2 regularization, if needed. Otherwise, it returns None."""
        if self._options.nn_l2_regularizer > 0.0:
            return regularizers.l2(self._options.nn_l2_regularizer)
        return None

    def _mc_dropout(self):
        """Gets correct training argument depending on MC dropout kind."""
        if self._options.nn_dropout_no_mc:
            return None
        return True

    def _internal_predict(self, x_test):
        """Predict from given data using model internally."""
        if self._options.lstm_stateful:
            x_test = x_test[:, -1:, :, :]
        prediction = self._inference_model.predict(
            x_test, batch_size=self._inference_batch_size)
        if self._options.nn_use_variable_sigma:
            # Here we sample from estimated normal distributions.
            # First: Transform estimated log(sigma^2) into sigma
            sigma = np.sqrt(np.exp(prediction[:, 1]))
            # Second, sample normal distributions
            prediction = np.random.normal(loc=prediction[:, 0], scale=sigma)
        return prediction

    def _reset_internal_states(self, x_test):
        """Called before any prediction."""
        if self._options.lstm_stateful:
            # In stateful mode, we assume sequential_mini_step is 1
            if self._options.sequential_mini_step != 1:
                logging.error('Stateful mode assumes sequential mini step 1!')
                sys.exit(1)
            # In this case, x data shape is (samples, 1, lags, series)
            x_shape = x_test.shape
            # If there is no stateful model already, or has different test size
            if (self._inference_model is None or
                    self._inference_batch_size != x_shape[0]):
                input_layer = layers.Input(
                    batch_shape=([x_shape[0], 1] + list(x_shape[2:])))
                last_layer = self._create_net(input_layer, stateful=True)
                self._inference_model = models.Model(
                    inputs=input_layer, outputs=last_layer)
                # The model is identical to the trained one, but with stateful
                # LSTMs. We copy weights from the original model to the new one
                self._inference_model.set_weights(self._model.get_weights())
                self._inference_batch_size = x_shape[0]
            # We reset states and then pass data for times we already know.
            self._inference_model.reset_states()
            # Note that in predict calls below we get slices of length 1 step
            for step in range(x_shape[1] - 1):
                # We pass batch size equal to all samples
                self._inference_model.predict(
                    x_test[:, step:step + 1, :, :],
                    batch_size=self._inference_batch_size)

    def _loss(self):
        """Correct final loss for this model. Default mean squared error."""
        return (util.mean_squared_loss_with_sigma
                if self._options.nn_use_variable_sigma else
                losses.mean_squared_error)

    def _callbacks(self):
        """Return additional callbacks to run during training."""

    def _internal_fit(self, x_train, y_train, validation_x=None,
                      validation_y=None):
        """Fit internal model with given data."""
        # As a new model will be trained, remove old stateful model, if any
        self._inference_model = None
        self._inference_batch_size = None
        # x_train data shape is (samples, steps, lags, series)
        if self._options.sequential_mini_step > 0:
            # In this case, we always perform one step prediction, so ignore
            # other steps of y_train for training, and use them only for eval.
            # Note we duplicate y_train in memory and don't overwrite it
            y_train = y_train[:, :self._options.sequential_mini_step, :]
            if validation_y is not None:
                validation_y = validation_y[
                    :, :self._options.sequential_mini_step, :]
        if self._options.nn_use_variable_sigma:
            # Note we add a dummy output that is ignored by metrics. It is
            # because metrics need same input size in prediction and y_train.
            # Sigma predictions are ignored (except for the loss).
            # This duplicates y_train in memory, but it does not overwrite it
            y_train = np.stack([y_train, np.zeros(y_train.shape)], axis=1)
            if validation_y is not None:
                validation_y = np.stack(
                    [validation_y, np.zeros(validation_y.shape)], axis=1)
            metrics = [util.sigma_mean_squared_error,
                       util.sigma_mean_absolute_error,
                       util.sigma_mean_absolute_percentage_error]
        else:
            metrics = [losses.mean_squared_error,
                       losses.mean_absolute_error,
                       losses.mean_absolute_percentage_error]
        # We create model here
        input_layer = layers.Input(shape=x_train.shape[1:])
        last_layer = self._create_net(input_layer)
        self._model = models.Model(inputs=input_layer, outputs=last_layer)
        optimizer = getattr(optimizers, self._options.nn_optimizer)(
            lr=self._options.nn_learning_rate)
        self._model.compile(
            loss=self._loss(), optimizer=optimizer, metrics=metrics)
        logging.info(self._model.summary())
        validation_data = None
        calls = None
        if validation_x is not None:
            validation_data = (validation_x, validation_y)
            if self._options.nn_patience >= 0:
                if self._options.flow_use_temperature:
                    calls = [CustomStopper(
                        monitor='val_loss', patience=self._options.nn_patience,
                        restore_best_weights=True,
                        start_epoch=self._options.flow_temperature_steps)]
                else:
                    calls = [callbacks.EarlyStopping(
                        monitor='val_loss', patience=self._options.nn_patience,
                        restore_best_weights=True)]
        additional_calls = self._callbacks()
        if self._options.nn_tensorboard:
            tb_call = callbacks.TensorBoard(log_dir="./logs")
            if additional_calls is None:
                additional_calls = [tb_call]
            else:
                additional_calls.append(tb_call)
        if calls is None:
            calls = additional_calls
        elif additional_calls is not None:
            calls += additional_calls
        self._model.fit(
            x=x_train, y=y_train, validation_data=validation_data,
            epochs=self._options.nn_epochs, callbacks=calls,
            batch_size=self._options.nn_batch_size)
        # Store real number of epochs where it stopped
        self._stopped_epoch = None
        if self._options.nn_patience and calls:
            for one_call in calls:
                if hasattr(one_call, 'stopped_epoch'):
                    self._stopped_epoch = (one_call.stopped_epoch -
                                           self._options.nn_patience)
        if self._stopped_epoch is None:
            self._stopped_epoch = self._options.nn_epochs
        if not self._options.lstm_stateful:
            # If not stateful prediction, then use same model for inference
            self._inference_model = self._model
            self._inference_batch_size = self._options.nn_batch_size

    def stopped_epoch(self):
        """Get number of epochs elapsed until end of training."""
        return self._stopped_epoch

    def _number_out_steps(self):
        """Returns correct number of steps that are output of regression."""
        # Number outputs is number of mini steps (for multi step regression)
        return (self._options.sequential_mini_step
                if self._options.sequential_mini_step > 0 else
                self._output_shape[0])

    def _number_out_series(self):
        """Returns correct number of series that are output of regression."""
        return self._output_shape[1]

    def _get_number_outputs(self):
        """Returns correct number of outputs for the neural network."""
        # Number of outputs is multiplied by number of output series
        number_outputs = self._number_out_steps() * self._number_out_series()
        if self._options.nn_use_variable_sigma:
            # In this case, output has two values: mean and variance
            number_outputs *= 2
        return number_outputs

    def _create_net(self, last_layer, stateful=False):
        """Create neural net model. To be called at the end by subclasses."""
        # Output shape is (time steps, number of series)
        output_shape = (self._number_out_steps(), self._number_out_series())
        if self._options.nn_use_variable_sigma:
            output_shape = tuple([2] + list(output_shape))
        # Reshape output to be exactly the same as y shapes
        return layers.Reshape(output_shape)(last_layer)

    def save(self, file_name):
        """Save model in given file name."""
        self._model.save(file_name)
        joblib.dump([self._stopped_epoch], '{}_nn'.format(file_name))
        super(NeuralNet, self).save(file_name)

    def load(self, file_name):
        """Load model from given file name."""
        super(NeuralNet, self).load(file_name)
        self._model = models.load_model(file_name)
        read_list = joblib.load('{}_nn'.format(file_name))
        self._stopped_epoch = read_list[0]
        self._inference_model = None
        self._inference_batch_size = None
