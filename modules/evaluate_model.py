# coding=utf-8
"""Module to train and evaluate a model in a dataset."""
import logging

from models import model
from models import paper_lstm_sampled
from models import persistence


AVAILABLE_MODELS = [
    paper_lstm_sampled.AutoInjectLSTM,
    paper_lstm_sampled.EncoderDecoderLSTM,
    paper_lstm_sampled.PaperLSTMSampled,
    persistence.Persistence,
    persistence.StochasticPersistence,
]


def get_available_model_names():
    """Returns a list of available model class names."""
    return [one.__name__.lower() for one in AVAILABLE_MODELS]


def _get_model_class(model_name):
    """Creates correct model class from name."""
    model_name = model_name.lower()
    for one in AVAILABLE_MODELS:
        if one.__name__.lower() == model_name:
            return one
    return None


def build_model(model_name, model_options=None):
    """Build correct model instance."""
    model_options = (None if model_options is None else
                     model.build_options(model_options))
    return _get_model_class(model_name)(model_options=model_options)


def sample(the_model, x_data, sampler_repetitions):
    """Perform module actions."""
    logging.info('Will get %s evaluations over set with shape %s',
                 sampler_repetitions, x_data.shape)
    return the_model.predict(x_data, samples=sampler_repetitions)


def train(x_train, y_train, model_name, model_options, x_val=None, y_val=None):
    """Train one model."""
    the_model = build_model(model_name, model_options=model_options)
    the_model.fit(x_train, y_train,
                  validation_x=(None if x_val is None else x_val),
                  validation_y=(None if y_val is None else y_val))
    return the_model
