# coding=utf-8
"""Module to process data and transforms it into regression problem."""
import datetime
import enum
import logging
import random

import numpy as np
import pandas as pd
from sklearn import model_selection

WEEKEND_DAYS = [5, 6]  # Saturday and Sunday
D_TOTAL = 1025  # Number of D series in Low Carbon London Project data
N_TOTAL = 4173  # Number of N series in Low Carbon London Project data
GMT = 'GMT'
# How to reduce resolution
MEAN = 'mean'
SUM = 'sum'
# How to select columns
ALL = 'all'
RANDOM = 'random'


class FillMethod(enum.Enum):
    """How to fill missing values."""
    REPEAT_LAST = 1  # Last valid value observation
    REPEAT_DAILY = 2  # Last valid value of 24 hours before
    OLD = 3


def select_columns(series_list):
    """Return selected series to use."""
    if len(series_list) == 1 and series_list[0].startswith(RANDOM):
        number = int(series_list[0][6:])
        # D goes from 0000 to 1024 (1025 series)
        # N goes from 0000 to 4172 (4173 series)
        selected = set()
        while len(selected) < number:
            # Random number in range 0 <= n <= D_TOTAL + N_TOTAL -1
            new_selected = random.randint(0, D_TOTAL + N_TOTAL - 1)
            if new_selected not in selected:
                selected.add(new_selected)
        columns = []
        for column in sorted(selected):
            if column < D_TOTAL:
                columns.append('D{:04d}'.format(column))
            else:
                columns.append('N{:04d}'.format(column - D_TOTAL))
        logging.info('Using random columns: %s', columns)
        return [GMT] + columns
    if len(series_list) == 1 and series_list[0] == ALL:
        return None
    return [GMT] + series_list


def load(file_name, series_list):
    """Load data from file, considering only given columns."""
    logging.info('Loading data from input file')
    columns = select_columns(series_list)
    the_data = pd.read_csv(
        file_name, parse_dates=True, index_col=GMT, usecols=columns)
    the_data = the_data.set_index(pd.to_datetime(the_data.index, utc=True))
    return the_data


def fill_null_values(data, method):
    """Input missing values, repeating previous value."""
    logging.info('Filling missing values')
    if method == FillMethod.OLD:
        for column in data.columns:
            the_index = data[column].index
            for position in np.where(data[column].isnull())[0]:
                data.loc[the_index[position], column] = (
                    0.0 if position == 0 else
                    data.loc[the_index[position - 1], column])
    elif method == FillMethod.REPEAT_LAST:
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)
    elif method == FillMethod.REPEAT_DAILY:
        daily_step = datetime.timedelta(days=1)
        for column in data.columns:
            the_index = data[column].index
            first_index = the_index[0]
            for position in np.where(data[column].isnull())[0]:
                current_index = the_index[position]
                previous_index = current_index - daily_step
                if previous_index < first_index:
                    continue
                data.loc[current_index, column] = data.loc[previous_index,
                                                           column]
            last_index = the_index[-1]
            for position in np.flip(np.where(data[column].isnull())[0]):
                current_index = the_index[position]
                next_index = current_index + daily_step
                if next_index > last_index:
                    continue
                data.loc[current_index, column] = data.loc[next_index, column]


def reduce_resolution(data, resolution, method=MEAN):
    """Transform time series into sums every given number of minutes."""
    logging.info('Transforming data into resolution %s.', resolution)
    resample = data.resample(resolution)
    if method == SUM:
        return resample.sum()
    return resample.mean()  # Assuming default is MEAN


def date_features(time_series_index):
    """Get additional date features from a pandas time series index."""
    return pd.concat([
        pd.Series(time_series_index.year),
        pd.Series(time_series_index.month),
        pd.Series(time_series_index.day),
        pd.Series(time_series_index.hour),
        pd.Series(time_series_index.dayofweek),
        pd.Series(time_series_index.dayofweek).isin(WEEKEND_DAYS)], axis=1)


def make_instances(data, input_steps, input_lags, output_steps,
                   output_series_list, use_date_features):
    """Make input features and targets for supervised time series task.
    Parameters:
        - I: number of input steps to run recurrences
        - L: number of input lags to consider on each input time step
        - O: number of output steps to predict for each sample
        - output_series: which input columns consider for output
        - date_features: whether to include extra date features or not
    Input is a dataframe with each time series in a column, with T times
    Output is dataset object with x input data, y output data and features.
    X input data is a tensor with shapes:
        - Number of samples (T - O - I - L + 2)
        - Number of input steps (I)
        - Number of input lags (L)
        - Number of input time series (all columns in current dataframe) +
          number of date features (5 if included, 0 otherwise)
    Y output data is a tensor with shapes:
        - Number of samples (T - O - I - L + 2)
        - Number of output steps (O)
        - Number of output time series
    """
    logging.info('Create inputs from time series')
    # Fix output series to consider
    if len(output_series_list) == 1 and output_series_list[0] == ALL:
        output_series_list = data.columns
    output_filter = np.in1d(data.columns, output_series_list)
    if use_date_features:
        # If date features are required, then add them as input time series
        features = date_features(data.index)
        data = np.concatenate([data, features], axis=1).astype(np.float)
        output_filter = np.concatenate([output_filter,
                                        [False] * features.shape[1]])
    else:
        data = np.asarray(data).astype(np.float)
    # Note we forced data type to float, to avoid mixing input types.
    # Build x and y according to parameters
    # Samples go from 0 to T - O - I - L + 2
    last = data.shape[0] - input_steps - output_steps - input_lags + 2
    x_data = np.atleast_3d([
        [data[start + step:start + step + input_lags]
         for step in range(input_steps)] for start in range(last)])
    y_data = np.atleast_3d([
        data[start + input_steps + input_lags - 1:
             (start + input_steps + input_lags +
              output_steps - 1), output_filter]
        for start in range(last)])
    return x_data, y_data


def prepare(input_file_name, input_series_list, input_steps, input_lags,
            output_series_list, output_steps, resolution, resolution_method,
            fillna_method, use_date_features):
    """Reads data and transforms it in desired regression problem."""
    data = load(input_file_name, input_series_list)
    # Adding missing values, just repeating previous ones
    fill_null_values(data, method=FillMethod[fillna_method.upper()])
    data = reduce_resolution(data, resolution, resolution_method)
    # Fill null values again, as reducing resolution may address points
    # with no data, and they are marked as null
    fill_null_values(data, method=FillMethod[fillna_method.upper()])
    x_data, y_data = make_instances(
        data, input_steps, input_lags, output_steps, output_series_list,
        use_date_features)
    return x_data, y_data


def get_test_size(total_len, train_percentage, test_size):
    """Gets correct test size according to options."""
    if test_size > 0:
        return test_size
    return int(total_len * (1 - train_percentage))


def separate_random(x_data, y_data, train_percentage, test_size):
    """Return splitted arrays for random split options."""
    return model_selection.train_test_split(
        x_data, y_data, test_size=get_test_size(len(x_data), train_percentage,
                                                test_size))


def separate_non_random(x_data, y_data, number_splits, split_position,
                        split_overlap, train_percentage, test_size):
    """Return splitted arrays for non random split options."""
    total_len = int(len(x_data) / (number_splits * (1 - split_overlap) +
                                   split_overlap))
    start_pos = int((split_position - 1) * (1 - split_overlap) * total_len)
    end_pos = int(start_pos + total_len)
    mid_pos = end_pos - get_test_size(total_len, train_percentage, test_size)
    return (x_data[start_pos:mid_pos], x_data[mid_pos:end_pos],
            y_data[start_pos:mid_pos], y_data[mid_pos:end_pos])


def separate(x_data, y_data, train_percentage, test_size, random_split,
             number_splits, split_overlap, split_position):
    """Returns separated train and test sets."""
    logging.info('Separating train and test sets')
    if random_split:
        x_train, x_test, y_train, y_test = separate_random(
            x_data, y_data, train_percentage, test_size)
    else:
        x_train, x_test, y_train, y_test = separate_non_random(
            x_data, y_data, number_splits, split_position, split_overlap,
            train_percentage, test_size)
    logging.info('Created sets with shapes:')
    logging.info('Train: %s and %s', x_train.shape, y_train.shape)
    logging.info('Test: %s and %s', x_test.shape, y_test.shape)
    return x_train, x_test, y_train, y_test


def separate_validation(x_data, y_data, validation_percentage, random_split):
    """Create separation of validation set."""
    return separate(x_data, y_data, 1.0 - validation_percentage, -1,
                    random_split, 1, 0.0, 1)
