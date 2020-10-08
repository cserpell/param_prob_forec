# coding=utf-8
"""LSTM recurrent network models with internal sampling."""
import tensorflow as tf
from tensorflow import math
from tensorflow import nn
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.python.keras.layers import merge

from models import neural_net
from modules import distributions
from modules import util


class PaperLSTMSampled(neural_net.NeuralNet):
    """Neural net model to forecast time series with a LSTM network."""

    def __init__(self, model_options=None):
        """Constructor of PaperLSTMSampled."""
        super(PaperLSTMSampled, self).__init__(model_options=model_options)
        self._distribution_layer = None
        self._output_distribution = (
            distributions.Distribution[
                self._options.nn_output_distribution.upper()]
            if self._options else None)

    def load(self, file_name):
        """Load model from given file name."""
        super(PaperLSTMSampled, self).load(file_name)
        self._output_distribution = distributions.Distribution[
            self._options.nn_output_distribution.upper()]

    def _loss(self):
        """Correct final loss for this model. Default mean squared error."""

        def likelihood_loss(y_true, _):
            """Adding negative log likelihood loss."""
            y_true = tf.reshape(
                y_true, (tf.shape(y_true)[0], self._get_number_outputs()))
            return -self._distribution_layer.log_prob(y_true +
                                                      distributions.EPSILON)

        return likelihood_loss

    def _create_net(self, last_layer, stateful=False):
        """Create neural net model. To be called at the end by subclasses."""
        # X input data shape is (samples, steps, lags, series), so it is
        # transformed here to what LSTM input expects:
        # (samples, steps, lags * series)
        last_layer = layers.Reshape((
            -1, util.prod(val for val in last_layer.shape[2:])))(last_layer)
        # LSTM will learn to add state information to the input features.
        recurrent_dropout = (  # The same for each LSTM layer
            self._options.nn_dropout_recurrence
            if self._options.nn_dropout_recurrence > 0.0 else 0.0)
        for number in range(self._options.lstm_layers):
            # Each LSTM layer
            # Note we are not adding dropout to the input layer.
            # The training keyword is necessary to use dropout when testing
            last_layer = layers.LSTM(
                self._options.lstm_nodes,
                kernel_regularizer=self._l2_regularization(),
                return_sequences=bool(number != self._options.lstm_layers - 1),
                recurrent_regularizer=self._l2_regularization(),
                recurrent_dropout=recurrent_dropout,
                stateful=stateful)(last_layer, training=self._mc_dropout())
        # LSTM output is already flatten here
        # Add dropout to output layer
        if self._options.nn_dropout_output > 0.0:
            last_layer = layers.Dropout(self._options.nn_dropout_output)(
                last_layer, training=self._mc_dropout())
        # Output dense layer with correct number of outputs.
        # Using default activation for regression: linear
        last_layer = layers.Dense(
            distributions.number_parameters(self._output_distribution) *
            self._get_number_outputs(),
            kernel_regularizer=self._l2_regularization())(last_layer)
        # Reshape output to put mean and variance first
        last_layer = layers.Reshape(
            (distributions.number_parameters(self._output_distribution),
             self._get_number_outputs()))(last_layer)
        last_layer = distributions.get_sampler(
            self._output_distribution)(last_layer)
        self._distribution_layer = last_layer
        # As we added epsilon to avoid the zero value problem, we subtract it
        last_layer = layers.Lambda(
            lambda param: param - distributions.EPSILON)(last_layer)
        # Base NeuralNet class adds a reshape layer with correct shape
        return super(PaperLSTMSampled, self)._create_net(
            last_layer, stateful=stateful)


class AddOutput(merge._Merge):
    """Layer that adds output and removes oldest value.
    It takes as input a list of two tensors,
    all of the same shape except for the concatenation axis,
    and returns a single tensor, the concatenation of all inputs.
    # Arguments
        **kwargs: standard layer keyword arguments.
    """

    def __init__(self, **kwargs):
        super(AddOutput, self).__init__(**kwargs)
        self.supports_masking = True
        self._reshape_required = False

    def build(self, input_shape):
        """Used purely for shape validation."""

    def _merge_function(self, inputs):
        previous_values = inputs[0]
        new_value = inputs[1]
        new_shape = backend.shape(new_value)
        reshaped_new_value = backend.reshape(
            new_value, (new_shape[0], new_shape[1], 1, new_shape[2]))
        return backend.concatenate(
            [previous_values[:, -1:, 1:, :], reshaped_new_value], axis=2)

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('An `AddOutput` layer should be called '
                             'on a list of inputs.')
        shape = list(input_shape[0])
        return tuple(shape[0:1] + [1] + shape[2:])

    def compute_mask(self, inputs, mask=None):
        return None


class AutoInjectLSTM(neural_net.NeuralNet):
    """Neural net model to forecast series with an auto inject LSTM network."""

    def __init__(self, model_options=None):
        """Constructor of AutoInjectLSTM."""
        super(AutoInjectLSTM, self).__init__(model_options=model_options)
        self._distribution_layers = None
        self._output_distribution = (
            distributions.Distribution[
                self._options.nn_output_distribution.upper()]
            if self._options else None)

    def load(self, file_name):
        """Load model from given file name."""
        super(AutoInjectLSTM, self).load(file_name)
        self._output_distribution = distributions.Distribution[
            self._options.nn_output_distribution.upper()]

    def _loss(self):
        """Correct final loss for this model. Default mean squared error."""

        def likelihood_loss(y_true, _):
            """Adding negative log likelihood loss."""
            return -math.reduce_sum([
                self._distribution_layers[pos].log_prob(
                    y_true[:, pos:pos + 1] + distributions.EPSILON)
                for pos in range(len(self._distribution_layers))])

        return likelihood_loss

    def _create_net(self, last_layer, stateful=False):
        """Create neural net model. To be called at the end by subclasses."""
        # X input data shape is (samples, steps, lags, series), so it is
        # transformed here to what LSTM input expects:
        # (samples, steps, lags * series)
        input_size = util.prod(val for val in last_layer.shape[2:])
        input_layer = last_layer
        last_layer = layers.Reshape((-1, input_size))(last_layer)
        # LSTM will learn to add state information to the input features.
        recurrent_dropout = (  # The same for each LSTM layer
            self._options.nn_dropout_recurrence
            if self._options.nn_dropout_recurrence > 0.0 else 0.0)
        lstm_layers = []
        lstm_states = []
        for number in range(self._options.lstm_layers):
            # Each LSTM layer
            # Note we are not adding dropout to the input layer.
            # The training keyword is necessary to use dropout when testing
            lstm_layers.append(layers.LSTM(
                self._options.lstm_nodes, stateful=stateful,
                kernel_regularizer=self._l2_regularization(),
                return_sequences=bool(number != self._options.lstm_layers - 1),
                recurrent_regularizer=self._l2_regularization(),
                recurrent_dropout=recurrent_dropout,
                return_state=True))
            last_layer, l_h, l_c = lstm_layers[-1](
                last_layer, training=self._mc_dropout())
            lstm_states.append([l_h, l_c])
        # LSTM output is already flatten here
        # Outputs dense layers with correct number of outputs.
        previous_input_layer = input_layer
        self._distribution_layers = []
        for number in range(self._number_out_steps()):
            if number != 0:
                last_layer = AddOutput()([previous_input_layer, last_layer])
                previous_input_layer = last_layer
                last_layer = layers.Reshape((-1, input_size))(last_layer)
                for num, the_layer in enumerate(lstm_layers):
                    last_layer, l_h, l_c = the_layer(
                        last_layer, training=self._mc_dropout(),
                        initial_state=lstm_states[num])
                    lstm_states[num] = [l_h, l_c]
            # Add dropout to output layer
            if self._options.nn_dropout_output > 0.0:
                last_layer = layers.Dropout(self._options.nn_dropout_output)(
                    last_layer, training=self._mc_dropout())
            # Using default activation for regression: linear
            last_layer = layers.Dense(
                distributions.number_parameters(self._output_distribution) *
                self._number_out_series(),
                kernel_regularizer=self._l2_regularization())(last_layer)
            # Reshape output to put distribution parameters first
            last_layer = layers.Reshape(
                (distributions.number_parameters(self._output_distribution), 1,
                 self._number_out_series()))(last_layer)
            last_layer = distributions.get_sampler(
                self._output_distribution)(last_layer)
            # We take the output here and build an input for next LSTMs
            self._distribution_layers.append(last_layer)
        if len(self._distribution_layers) > 1:
            # Concatenate all outputs when more than one time step
            last_layer = layers.Concatenate(axis=1)(self._distribution_layers)
        # As we added epsilon to avoid the zero value problem, we subtract it
        last_layer = layers.Lambda(
            lambda param: param - distributions.EPSILON)(last_layer)
        # Base NeuralNet class adds a reshape layer with correct shape
        return super(AutoInjectLSTM, self)._create_net(
            last_layer, stateful=stateful)


class EncoderDecoderLSTM(neural_net.NeuralNet):
    """Neural model to forecast series with two LSTMs: encoder and decoder."""

    def __init__(self, model_options=None):
        """Constructor of PaperLSTMSampled."""
        super(EncoderDecoderLSTM, self).__init__(model_options=model_options)
        self._output_distribution = (
            distributions.Distribution[
                self._options.nn_output_distribution.upper()]
            if self._options else None)

    def _create_net(self, last_layer, stateful=False):
        """Create neural net model. To be called at the end by subclasses."""
        # X input data shape is (samples, steps, lags, series), so it is
        # transformed here to what LSTM input expects:
        # (samples, steps, lags * series)
        input_size = util.prod(val for val in last_layer.shape[2:])
        input_layer = last_layer
        last_layer = layers.Reshape((-1, input_size))(last_layer)
        # LSTM will learn to add state information to the input features.
        recurrent_dropout = (  # The same for each LSTM layer
            self._options.nn_dropout_recurrence
            if self._options.nn_dropout_recurrence > 0.0 else 0.0)
        lstm_layers = []
        lstm_states = []
        for number in range(self._options.lstm_layers):
            # Each LSTM layer
            # Note we are not adding dropout to the input layer.
            # The training keyword is necessary to use dropout when testing
            lstm_layers.append(layers.LSTM(
                self._options.lstm_nodes, stateful=stateful,
                kernel_regularizer=self._l2_regularization(),
                return_sequences=bool(number != self._options.lstm_layers - 1),
                recurrent_regularizer=self._l2_regularization(),
                recurrent_dropout=recurrent_dropout,
                return_state=True))
            last_layer, l_h, l_c = lstm_layers[-1](
                last_layer, training=self._mc_dropout())
            lstm_states.append([l_h, l_c])
        # LSTM output is already flatten here
        # Outputs dense layers with correct number of outputs.
        previous_input_layer = input_layer
        output_lstm_layers = None
        output_layers = []
        for number in range(self._number_out_steps()):
            if number != 0:
                # For other steps than first, add sampling.
                last_layer = distributions.get_sampler(
                    self._output_distribution)(last_layer)
                # We take the output here and build an input for next LSTMs
                last_layer = layers.Reshape(
                    (1, self._number_out_series()))(last_layer)
                last_layer = AddOutput()([previous_input_layer, last_layer])
                previous_input_layer = last_layer
                last_layer = layers.Reshape((-1, input_size))(last_layer)
                if output_lstm_layers is None:
                    output_lstm_layers = []
                    for num in range(self._options.lstm_layers):
                        # Each output LSTM layer
                        # training keyword necessary for dropout when testing
                        output_lstm_layers.append(layers.LSTM(
                            self._options.lstm_nodes, stateful=stateful,
                            kernel_regularizer=self._l2_regularization(),
                            return_sequences=bool(
                                num != self._options.lstm_layers - 1),
                            recurrent_regularizer=self._l2_regularization(),
                            recurrent_dropout=recurrent_dropout,
                            return_state=True))
                for num in range(len(lstm_layers)):
                    last_layer, l_h, l_c = output_lstm_layers[num](
                        last_layer, training=self._mc_dropout(),
                        initial_state=lstm_states[num])
                    lstm_states[num] = [l_h, l_c]
            # Add dropout to output layer
            if self._options.nn_dropout_output > 0.0:
                last_layer = layers.Dropout(self._options.nn_dropout_output)(
                    last_layer, training=self._mc_dropout())
            # Using default activation for regression: linear
            last_layer = layers.Dense(
                distributions.number_parameters(self._output_distribution) *
                self._number_out_series(),
                kernel_regularizer=self._l2_regularization())(last_layer)
            # Reshape output to put mean and variance first
            output_layers.append(layers.Reshape(
                (distributions.number_parameters(self._output_distribution),
                 self._number_out_series()))(last_layer))
            last_layer = output_layers[-1]
        if len(output_layers) > 1:
            # Concatenate all outputs
            last_layer = layers.Concatenate(axis=2)(output_layers)
        # Base NeuralNet class adds a reshape layer with correct shape
        return super(EncoderDecoderLSTM, self)._create_net(
            last_layer, stateful=stateful)
