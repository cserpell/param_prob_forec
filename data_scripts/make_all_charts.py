# coding=utf-8
"""Script to create all meaningfull charts from data."""
import argparse
from os import path

import numpy as np
from numpy import linalg
import pandas as pd
from statsmodels.graphics import tsaplots

import chart_util
from modules import data

EXTENSION = 'png'


def save(file_name):
    """Stores a figure and opens a new one."""
    chart_util.save(path.join('figure', '{}.{}'.format(file_name, EXTENSION)))


def missing_values(the_data):
    """Charts of missing values information."""
    the_data.isna().sum().plot.bar(figsize=chart_util.DEFAULT_FIG_SIZE)
    save('missing_values_per_serie')
    na_data = the_data.isna()
    na_data.sum(axis=1).plot(figsize=chart_util.DEFAULT_FIG_SIZE)
    save('missing_values_per_time')
    for column in na_data:
        na_data[column].round().plot()
        save('missing_values_per_time_serie_{}'.format(column))
    all_na = []
    for column in na_data:
        all_na.append(np.diff(np.where(np.concatenate((
            [na_data[column].values[0]],
            na_data[column].values[:-1] != na_data[column].values[1:],
            [True])))[0])[::2])
    all_na = pd.concat([pd.DataFrame(one) for one in all_na])
    all_na.hist(figsize=chart_util.DEFAULT_FIG_SIZE)
    save('missing_values_length_histogram_10bins')
    all_na.hist(bins=100, figsize=chart_util.DEFAULT_FIG_SIZE)
    save('missing_values_length_histogram_100bins')
    transform = []
    for value in all_na[0]:
        for _ in range(value):
            transform.append(value)
    weighted_na = pd.Series(transform)
    weighted_na.hist(figsize=chart_util.DEFAULT_FIG_SIZE)
    save('missing_values_length_weighted_histogram_10bins')
    weighted_na.hist(bins=100, figsize=chart_util.DEFAULT_FIG_SIZE)
    save('missing_values_length_weighted_histogram_100bins')


def series(the_data, resolution):
    """Charts of each individual series."""
    data_resol = data.reduce_resolution(the_data, resolution)
    for column in data_resol:
        data_resol[column].plot(figsize=chart_util.DEFAULT_FIG_SIZE)
        save('all_series_{}_{}'.format(resolution, column))
        data_resol[column].hist(bins=30, figsize=chart_util.DEFAULT_FIG_SIZE)
        save('histogram_of_values_{}_{}'.format(resolution, column))
        try:
            tsaplots.plot_acf(data_resol[column], ax=chart_util.pyplot.gca(),
                              lags=30)
            save('acf_30_{}_{}'.format(resolution, column))
        except linalg.LinAlgError:
            pass
        try:
            tsaplots.plot_acf(data_resol[column], ax=chart_util.pyplot.gca(),
                              lags=120)
            save('acf_120_{}_{}'.format(resolution, column))
        except linalg.LinAlgError:
            pass
        try:
            tsaplots.plot_pacf(data_resol[column], ax=chart_util.pyplot.gca(),
                               lags=30)
            save('pacf_30_{}_{}'.format(resolution, column))
        except linalg.LinAlgError:
            pass
        try:
            tsaplots.plot_pacf(data_resol[column], ax=chart_util.pyplot.gca(),
                               lags=120)
            save('pacf_120_{}_{}'.format(resolution, column))
        except linalg.LinAlgError:
            pass


def main():
    """Main execution point."""
    parser = argparse.ArgumentParser(
        description=('Create all meaningfull charts. '
                     'Use only in a computer with a lot of RAM.'))
    parser.add_argument(
        '--file_name', help='file name with input data',
        default='~/datasets/inf475/consumption_lclp.csv')
    parser.add_argument(
        '--series', nargs='+', default=['all'],
        help='column series to use. Special words: "all" and "random[number]"')
    parser.add_argument(
        '--fillna_method', default=data.FillMethod.REPEAT_LAST.name.lower(),
        choices=[one.name.lower() for one in data.FillMethod],
        help='how to fill NA values')
    args = parser.parse_args()
    chart_util.start_figure()
    the_data = data.load(args.file_name, args.series)
    missing_values(the_data)
    # Make charts of missing values
    data.fill_null_values(the_data,
                          data.FillMethod[args.fillna_method.upper()])
    series(the_data, 'H')  # Hourly
    # series(the_data, '30T')  # 30 min
    # series(the_data, '15T')  # 15 min
    # series(the_data, 'T')  # 1 min


if __name__ == '__main__':
    main()
