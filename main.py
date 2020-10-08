# coding=utf-8
"""Script to train and test models to predict time series."""
import argparse
import logging
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers

from modules import data
from modules import distributions
from modules import evaluate_model
from modules import metrics
from modules import preprocessor
from modules import util

PREPARED_DATASET = 'prepared_dataset'
TRAIN_DATASET = 'train_dataset'
TRAIN_METRICS = 'train_metrics'
TRAIN_OUTPUT = 'train_output'
TEST_DATASET = 'test_dataset'
TEST_METRICS = 'test_metrics'
TEST_OUTPUT = 'test_output'
PRE_DATASET = 'pre_dataset'
PRE_METRICS = 'pre_metrics'
PRE_OUTPUT = 'pre_output'
VALIDATION_DATASET = 'validation_dataset'
VALIDATION_METRICS = 'validation_metrics'
VALIDATION_OUTPUT = 'validation_output'
ARGS = 'args'
MODEL = 'model'
PRE_MODEL = 'pre_model'


class Script:
    """Script to train and test models to predict time series."""

    def __init__(self, args):
        """Constructor of Script."""
        self._args = args  # Program arguments

    def _make_dir(self):
        """Creates directory, without failing if it already exists."""
        try:
            os.makedirs(self._args.run_directory)
        except FileExistsError:
            pass  # Do nothing because it already exists

    def _file(self, name):
        """Add directory prefix to file name."""
        return os.path.join(self._args.run_directory, name)

    def _store(self, the_object, file_name):
        """Store object in given file file."""
        return util.store(the_object, self._file(file_name))

    def _get_train_test(self):
        """Build train and test sets from x, y pairs."""
        with util.time_context('prepare_data'):
            # Correct resolution and missing data
            x_data, y_data = data.prepare(
                self._args.file_name, self._args.input_series,
                self._args.input_steps, self._args.input_lags,
                self._args.output_series, self._args.output_steps,
                self._args.resolution, self._args.resolution_method,
                self._args.fillna_method, self._args.use_dates)
        if self._args.store_data:
            self._store([x_data, y_data], PREPARED_DATASET)
        with util.time_context('separate_train_test'):
            x_train, x_test, y_train, y_test = data.separate(
                x_data, y_data, self._args.train_percentage,
                self._args.test_size, self._args.random_split,
                self._args.number_splits, self._args.split_overlap,
                self._args.split_position)
        if self._args.store_data:
            self._store([x_train, y_train], TRAIN_DATASET)
            self._store([x_test, y_test], TEST_DATASET)
        return x_train, x_test, y_train, y_test

    def _get_validation_set(self, x_train, y_train):
        with util.time_context('separate_train_validation'):
            x_pre, x_val, y_pre, y_val = data.separate_validation(
                x_train, y_train, self._args.validation_percentage,
                self._args.random_split)
        if self._args.store_data:
            self._store([x_pre, y_pre], PRE_DATASET)
            self._store([x_val, y_val], VALIDATION_DATASET)
        return x_pre, x_val, y_pre, y_val

    def _train_one_model(self, x_train, y_train, x_val=None, y_val=None,
                         file_model=None):
        """Trains one model with given data."""
        with util.time_context('training_{}'.format(file_model)):
            the_model = evaluate_model.train(
                x_train, y_train, self._args.model, vars(self._args),
                x_val=x_val, y_val=y_val)
        if self._args.store_model:
            the_model.save(self._file(file_model))
        return the_model

    def _train_models(self, x_train, y_train, x_val=None, y_val=None,
                      file_model=None, number_models=1, epochs=None):
        """Trains many model with given data for an ensemble."""
        returned_models = []
        for num in range(number_models):
            if epochs is not None:
                self._args.nn_epochs = epochs[num]
            returned_models.append(self._train_one_model(
                x_train, y_train, x_val=x_val, y_val=y_val,
                file_model='{}_{}'.format(file_model, num)))
        return returned_models

    def _train_model(self, x_train, y_train):
        """Trains model with given data."""
        if self._args.validation_percentage > 0.0:
            x_pre, x_val, y_pre, y_val = self._get_validation_set(x_train,
                                                                  y_train)
            the_models = self._train_models(
                x_pre, y_pre, x_val=x_val, y_val=y_val, file_model=PRE_MODEL,
                number_models=self._args.ensemble_number_models)
            self._evaluate_data(
                the_models, x_pre, y_pre,
                file_output=PRE_OUTPUT, file_metrics=PRE_METRICS)
            del x_pre
            del y_pre
            self._evaluate_data(
                the_models, x_val, y_val,
                file_output=VALIDATION_OUTPUT, file_metrics=VALIDATION_METRICS)
            del x_val
            del y_val
            # To re use same options but with found number of epochs
            epochs = [the_model.stopped_epoch() for the_model in the_models]
        else:
            epochs = None
        the_models = self._train_models(
            x_train, y_train, file_model=MODEL,
            number_models=self._args.ensemble_number_models, epochs=epochs)
        self._evaluate_data(
            the_models, x_train, y_train,
            file_output=TRAIN_OUTPUT, file_metrics=TRAIN_METRICS)
        return the_models

    def _evaluate_data(self, the_models, x_test, y_test, file_output=None,
                       file_metrics=None):
        """Evaluate final model in test data."""
        with util.time_context('sampling_{}'.format(file_output)):
            test_output = [
                evaluate_model.sample(one_model, x_test, int(
                    self._args.evaluation_sampler_number /
                    self._args.ensemble_number_models))
                for one_model in the_models]
        if self._args.store_data:
            self._store(test_output, file_output)
        with util.time_context('metrics_{}'.format(file_output)):
            test_metrics = metrics.get_all(y_test, test_output)
        test_metrics.update({'time_{}'.format(key): value
                             for key, value in util.get_all_times().items()})
        self._store(test_metrics, file_metrics)

    def run(self):
        """Perform script actions."""
        logging.info('Using following args: %s', self._args)
        self._make_dir()
        self._store(vars(self._args), ARGS)
        x_train, x_test, y_train, y_test = self._get_train_test()
        the_models = self._train_model(x_train, y_train)
        del x_train
        del y_train
        self._evaluate_data(the_models, x_test, y_test,
                            file_output=TEST_OUTPUT, file_metrics=TEST_METRICS)


def main():
    """Main execution point."""
    parser = argparse.ArgumentParser(
        description='Script to train models to predict multiple time series.')
    # Base program arguments
    parser.add_argument(
        '--run_directory', default='.',
        help='directory where datasets and models are stored')
    parser.add_argument(
        '--seed', default=-1, type=int,
        help='seed to set for numpy and tensorflow. if -1, do not set any')
    parser.add_argument(
        '--store_data', action='store_true',
        help='when set, store data and outputs, useful for explore results')
    parser.add_argument(
        '--store_model', action='store_true',
        help='when set, store trained model, useful to reuse trained model')
    parser.add_argument(
        '--ensemble_number_models', default=1, type=int,
        help='number of ensemble models to train with same training data')
    # Prepare dataset arguments
    parser.add_argument(
        '--file_name', default='~/datasets/inf475/consumption_lclp.csv',
        help='file name with input data')
    parser.add_argument(
        '--fillna_method', default=data.FillMethod.OLD.name.lower(),
        choices=[one.name.lower() for one in data.FillMethod],
        help='how to fill NA values')
    parser.add_argument(
        '--input_lags', default=24, type=int,
        help='number of time steps to use as input in one time step')
    parser.add_argument(
        '--input_series', nargs='+', default=['D0612', 'N3405'],
        help='column series to use. Special words: "all" and "random[number]"')
    parser.add_argument(
        '--input_steps', default=48, type=int,
        help='number of time steps to run recurrence in models')
    parser.add_argument(
        '--output_steps', default=24, type=int,
        help='number of time steps to perform output prediction from model')
    parser.add_argument(
        '--output_series', nargs='+', default=['all'],
        help='column series to predict. Use word "all" to consider all input')
    parser.add_argument(
        '--resolution', default='H',  # H = hourly
        help=('resolution to consider aggregated data. Format from '
              'http://pandas.pydata.org/pandas-docs/stable/timeseries.html'
              '#offset-aliases'))
    parser.add_argument(
        '--resolution_method', choices=[data.MEAN, data.SUM],
        default=data.MEAN, help='mixing values method to reduce resolution')
    # Separate train and test sets arguments
    parser.add_argument(
        '--train_percentage', default=0.8, type=float,
        help='Train data percentage')
    parser.add_argument(
        '--test_size', default=-1, type=int,
        help=('If > 0, overrides train_percentage and makes number of samples '
              'in test set fixed'))
    parser.add_argument(
        '--validation_percentage', default=0.1, type=float,
        help='Validation data percentage, removed from train set')
    parser.add_argument(
        '--random_split', action='store_true',
        help='when set, random split instead of sequential')
    parser.add_argument(
        '--number_splits', default=1, type=int,
        help='number of splits when non random split is used')
    parser.add_argument(
        '--split_overlap', default=0.0, type=float,
        help='percentage of overlap of non random splits')
    parser.add_argument(
        '--split_position', default=1, type=int,
        help='position of current split in number of non random splits')
    # Evaluation number of samples to consider
    parser.add_argument(
        '--evaluation_sampler_number', default=10, type=int,
        help='number of samples to evaluate methods using sampling')
    # Model arguments
    parser.add_argument(
        '--model', default=evaluate_model.get_available_model_names()[0],
        choices=evaluate_model.get_available_model_names(),
        help='model to use')
    parser.add_argument(
        '--preprocess', default=preprocessor.Preprocess.MINMAX.name.lower(),
        choices=[one.name.lower() for one in preprocessor.Preprocess],
        help='preprocess kind')
    parser.add_argument(
        '--sequential_mini_step', default=0, type=int,
        help='if > 0, multistep is done sequentially with steps of this size')
    parser.add_argument(
        '--use_dates', action='store_true',
        help='when set, models use date info features as input time series')
    # Neural network related parameters
    parser.add_argument(
        '--nn_batch_size', default=32,
        help='neural nets batch size for training', type=int)
    parser.add_argument(
        '--nn_dropout_no_mc', action='store_true',
        help='when set, dropout is done only in training, not in test')
    parser.add_argument(
        '--nn_dropout_output', default=-1.0, type=float,
        help='if > 0.0, dropout rate to use for LSTM outputs')
    parser.add_argument(
        '--nn_dropout_recurrence', default=-1.0, type=float,
        help='if > 0.0, dropout rate to use in LSTM recurrent activations')
    parser.add_argument(
        '--nn_epochs', default=3, help='neural nets epochs for training',
        type=int)
    parser.add_argument(
        '--nn_l2_regularizer', default=-1.0, type=float,
        help='if > 0.0, L2 regularization is added to neural net weights')
    parser.add_argument(
        '--nn_learning_rate', default=0.01, type=float,
        help='learning rate used to train neural nets')
    parser.add_argument(
        '--nn_optimizer', default=optimizers.Adam.__name__,
        help='optimizer used to train neural nets')
    parser.add_argument(
        '--nn_output_distribution',
        default=distributions.Distribution.NORMAL.name.lower(),
        choices=[one.name.lower() for one in distributions.Distribution],
        help='output distribution for sampler models')
    parser.add_argument(
        '--nn_patience', default=-1, type=int,
        help='if >= 0, use early stop with this number of patience steps')
    parser.add_argument(
        '--nn_tensorboard', action='store_true',
        help='when set, store partial metrics for tensorboard debugging')
    parser.add_argument(
        '--nn_use_variable_sigma', action='store_true',
        help='when set, predict sigma of output also, using associated loss')
    # Model specific parameters
    parser.add_argument('--arima_d', default=0, type=int, help='ARIMA order d')
    parser.add_argument('--arima_p', default=1, type=int, help='ARIMA order p')
    parser.add_argument('--arima_q', default=1, type=int, help='ARIMA order q')
    parser.add_argument(
        '--flow_steps', default=1, type=int,
        help='number of normalizing flow steps')
    parser.add_argument(
        '--flow_temperature_max', default=1.0, type=float,
        help='maximum temperature for likelihood in normalizing flows')
    parser.add_argument(
        '--flow_temperature_min', default=1.0, type=float,
        help='minimum temperature for likelihood in normalizing flows')
    parser.add_argument(
        '--flow_temperature_steps', default=10, type=int,
        help='number of epochs to increase from min to max temperature')
    parser.add_argument(
        '--flow_use_temperature', action='store_true',
        help='when set, increase likelihood temperature at first epochs')
    parser.add_argument(
        '--lstm_layers', default=1, type=int, help='number of LSTM layers')
    parser.add_argument(
        '--lstm_nodes', default=5, type=int,
        help='number of LSTM memory units per layer')
    parser.add_argument(
        '--lstm_stateful', action='store_true',
        help=('when set, predict each step with previous state and small '
              'input, instead of complete inputs and resetting state'))
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.DEBUG)
    tf.compat.v1.disable_eager_execution()
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.compat.v1.set_random_seed(args.seed)
    Script(args).run()


if __name__ == '__main__':
    main()
