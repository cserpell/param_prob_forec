"""Script to process results manually."""
import os

from matplotlib import pyplot
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics import gofplots
from statsmodels.graphics import tsaplots

import main
from modules import metrics
from modules import util

DATASETS_DICT = {
    'UCI': 'consumption_uci.csv',
    'R69': 'rest_file_69.csv',
    'GEFCom': 'consumption_gefcom2014_load.csv',
    'B08': 'wind_b08.csv',
    'D08': 'wind_d08.csv',
}
DATASETS_INDEX = ['B08', 'D08', 'UCI', 'R69', 'GEFCom']


def load_one(directory, neural_network=True):
    """Load data from one experiment."""
    x_train, y_train = util.load(directory, main.TRAIN_DATASET)
    x_pre, y_pre = util.load(directory, main.PRE_DATASET)
    x_val, y_val = util.load(directory, main.VALIDATION_DATASET)
    x_test, y_test = util.load(directory, main.TEST_DATASET)
    data = {
        'train_output': util.load(directory, main.TRAIN_OUTPUT),
        'x_train': x_train,
        'y_train': y_train,
        'train_metrics': util.load(directory, main.TRAIN_METRICS),
        'pre_output': util.load(directory, main.PRE_OUTPUT),
        'x_pre': x_pre,
        'y_pre': y_pre,
        'pre_metrics': util.load(directory, main.PRE_METRICS),
        'val_output': util.load(directory, main.VALIDATION_OUTPUT),
        'x_val': x_val,
        'y_val': y_val,
        'val_metrics': util.load(directory, main.VALIDATION_METRICS),
        'test_output': util.load(directory, main.TEST_OUTPUT),
        'x_test': x_test,
        'y_test': y_test,
        'test_metrics': util.load(directory, main.TEST_METRICS),
        'args': util.load(directory, main.ARGS),
        'data': util.load(directory, main.PREPARED_DATASET)}
    if neural_network:
        data['model_nn'] = [
            util.load(directory, '{}_{}_nn'.format(main.MODEL, number))
            for number in range(data['args']['ensemble_number_models'])]
    return data


def build_some_charts(test_dataset, test_output):
    """Build some charts from one experiment."""
    print(test_output.shape)
    num_y = test_dataset.y_data.shape[0]
    output_shape = test_output.shape[1]
    pos = 3000
    pd.Series(np.concatenate([test_dataset.x_data[pos].reshape(50,),
                              test_dataset.y_data[pos].reshape(1,)])).plot()
    pd.DataFrame(data={
        'predicted0': test_output[0].reshape((output_shape,)),
        'predicted1': test_output[1].reshape((output_shape,)),
        'predicted2': test_output[2].reshape((output_shape,)),
        'predicted3': test_output[3].reshape((output_shape,)),
        'predicted4': test_output[4].reshape((output_shape,)),
        'real': test_dataset.y_data.reshape((num_y,))})[pos:pos + 50].plot()
    expected = metrics.expected_value_from_sample(test_output)
    quants2 = metrics.quantiles_from_sample(test_output, 20)
    pd.DataFrame(data={
        '5%': quants2[1].reshape((quants2.shape[1],)),
        'median': quants2[10].reshape((quants2.shape[1],)),
        '95%': quants2[19].reshape((quants2.shape[1],)),
        'mean': expected.reshape((quants2.shape[1],)),
        'real': test_dataset.y_data.reshape((num_y,))})[pos:pos + 20].plot()
    pyplot.show()
    quants = metrics.quantiles_from_sample(test_output, 4)
    pd.DataFrame(data={
        '0%': quants[0].reshape((quants.shape[1],)),
        '25%': quants[1].reshape((quants.shape[1],)),
        'median': quants[2].reshape((quants.shape[1],)),
        '75%': quants[3].reshape((quants.shape[1],)),
        '100%': quants[4].reshape((quants.shape[1],)),
        'mean': expected.reshape((quants.shape[1],)),
        'real': test_dataset.y_data.reshape((num_y,))})[pos:pos + 20].plot()
    pyplot.show()
    axes = pd.DataFrame(test_output.reshape(200, output_shape)).iloc[
        :, pos:pos + 20].boxplot(figsize=(9, 3))
    minisample = test_dataset.y_data.reshape((num_y,))[pos:pos + 20]
    pd.Series(np.concatenate([[minisample[0]], minisample])).plot(
        ax=axes, figsize=(9, 3))
    pyplot.show()
    quants_df = pd.DataFrame(data={
        '0%': np.concatenate([test_dataset.x_data[pos].reshape(50,),
                              quants[0][pos].reshape(1,)]),
        '25%': np.concatenate([test_dataset.x_data[pos].reshape(50,),
                               quants[1][pos].reshape(1,)]),
        'median': np.concatenate([test_dataset.x_data[pos].reshape(50,),
                                  quants[2][pos].reshape(1,)]),
        '75%': np.concatenate([test_dataset.x_data[pos].reshape(50,),
                               quants[3][pos].reshape(1,)]),
        '100%': np.concatenate([test_dataset.x_data[pos].reshape(50,),
                                quants[4][pos].reshape(1,)]),
        'mean': np.concatenate([test_dataset.x_data[pos].reshape(50,),
                                expected[pos].reshape(1,)]),
        'real': np.concatenate([test_dataset.x_data[pos].reshape(50,),
                                test_dataset.y_data[pos].reshape(1,)])})
    quants_df.plot(figsize=(9, 3))
    pyplot.legend(loc='upper left')
    pyplot.show()
    quants2_df = pd.DataFrame(data={
        '5%': np.concatenate([test_dataset.x_data[pos].reshape(50,),
                              quants2[1][pos].reshape(1,)]),
        'median': np.concatenate([test_dataset.x_data[pos].reshape(50,),
                                  quants2[10][pos].reshape(1,)]),
        '95%': np.concatenate([test_dataset.x_data[pos].reshape(50,),
                               quants2[19][pos].reshape(1,)]),
        'mean': np.concatenate([test_dataset.x_data[pos].reshape(50,),
                                expected[pos].reshape(1,)]),
        'real': np.concatenate([test_dataset.x_data[pos].reshape(50,),
                                test_dataset.y_data[pos].reshape(1,)])})
    quants2_df.plot(figsize=(9, 3))
    pyplot.legend(loc='upper left')
    pyplot.show()
    minisample = np.concatenate([test_dataset.x_data[pos].reshape((50,)),
                                 test_dataset.y_data[pos].reshape(1,)])
    axes = pd.Series(np.concatenate([[minisample[0]], minisample])).plot(
        figsize=(9, 3))
    pd.DataFrame(test_output.reshape(200, output_shape)).iloc[
        :, pos:(pos + 1)].boxplot(ax=axes, positions=[51], widths=1,
                                  manage_xticks=False)


def residuals_charts(test_dataset, test_output):
    """Build residuals charts for one experiment."""
    prediction_series = test_output.mean(axis=0).reshape(test_output.shape[1])
    residuals = prediction_series - test_dataset.y_data.reshape(
        test_dataset.y_data.shape[0])
    pd.Series(residuals).hist(bins=30)
    print(stats.normaltest(residuals))
    gofplots.qqplot(residuals)
    xxx = np.linspace(-3.5, 3.5, 4)
    pyplot.plot(xxx, xxx)
    tsaplots.plot_acf(residuals, lags=30)
    tsaplots.plot_pacf(residuals, lags=30)


def draw_pinball_loss(quant=0.25):
    """Draw a pinball loss example."""
    xxx = np.linspace(-2, 2, 100)
    pyplot.figure(figsize=(10, 5))
    pyplot.plot(xxx, np.max([-(1.0 - quant) * xxx, quant * xxx], axis=0),
                label=r'$q = {}\%$'.format(int(100.0 * quant)))
    pyplot.legend()


def draw_winkler_loss(alpha=0.05):
    """Draw a winkler loss example."""
    xxx = np.linspace(-2, 2, 100)
    pyplot.figure(figsize=(5, 5))
    pyplot.plot(
        xxx,
        np.max([2.0 * (-1.0 - xxx) / alpha + 2.0, 2.0 + 0.0 * xxx,
                2.0 + 2.0 * (xxx - 1.0) / alpha], axis=0))
    # label=r'$\alpha = {}\%$'.format(int(100.0 * alpha)))
    pyplot.ylim(0.0, 2.0 / alpha + 2.0 + 2.0)
    # pyplot.legend()


def include_metric_key(key, include_quant=True):
    """Check whether the metric has to be included."""
    return include_quant or not key.startswith('quant')


def get_float_metrics_df(the_metrics, include_quant=True):
    """Remove from object all those that are not a single number."""
    return pd.DataFrame(
        data={key: [value] for key, value in the_metrics.items()
              if include_metric_key(key) and
              not isinstance(value, np.ndarray)})


def get_all_metrics_in_dir(base_directory='.', metric_names=None,
                           consider_ensembles=True,
                           consider_non_ensembles=True):
    """Load metrics from all experiments in a directory."""
    return get_all_metrics(
        [os.path.join(base_directory, one_dir.name)
         for one_dir in os.scandir(path=base_directory) if one_dir.is_dir()],
        metric_names=metric_names, consider_ensembles=consider_ensembles,
        consider_non_ensembles=consider_non_ensembles)


def get_all_metrics(directories, metric_names=None, consider_ensembles=True,
                    consider_non_ensembles=True):
    """Load metrics from all directories given."""
    if metric_names is None:
        metric_names = main.VALIDATION_METRICS
    all_losses = []
    for directory in directories:
        try:
            args = util.load(directory, main.ARGS)
        except FileNotFoundError:
            print('Could not load program arguments of', directory)
            continue
        if not consider_non_ensembles and (
                'ensemble_number_models' not in args or
                args['ensemble_number_models'] <= 1):
            continue
        if (not consider_ensembles and 'ensemble_number_models' in args and
                args['ensemble_number_models'] > 1):
            continue
        try:
            test_metrics = util.load(directory, metric_names)
        except FileNotFoundError:
            print('Could not load metrics of', directory)
            continue
        last_epoch = get_last_epoch('{}.out'.format(directory))
        args['last_epoch'] = last_epoch
        args['has_dropout'] = bool(args['nn_dropout_output'] > 0.0)
        args['nn_dropout_no_mc'] = (
            bool(args['nn_dropout_no_mc'])
            if args['has_dropout'] and 'nn_dropout_no_mc' in args else False)
        args['mc_model_dropout'] = bool(
            args['has_dropout'] and
            ('nn_dropout_no_mc' not in args or
             not bool(args['nn_dropout_no_mc'])))
        all_losses.append([args, test_metrics])
    return all_losses


def get_last_epoch(file_name):
    """Get last epoch line, to get correct number of epochs."""
    last_epoch = -1
    try:
        with open(file_name, mode='r') as the_file:
            for line in the_file:
                if line.find('Epoch') != -1:
                    last_epoch = int(line[line.find('/') + 1:])
    except OSError:
        print('Cannot open file', file_name)
    return last_epoch


def os_str(input_str, number):
    """Adds prefix to string."""
    return 'os_{}_{}'.format(number, input_str)


def get_one_step_metrics_df(the_metrics, max_number=24, include_quant=True):
    """Remove from object all those that are not a single number."""
    pd_quants = []
    pd_int90 = []
    pd_others = []
    for number in range(max_number):
        if include_quant:
            pd_quants.append(pd.DataFrame(
                data={os_str(key, number): [value[:, number].mean()
                                            if value is not None else None]
                      for key, value in the_metrics.items()
                      if key.startswith('quant') and
                      not key.endswith('mean')}))
        pd_int90.append(pd.DataFrame(
            data={os_str(key, number): [value[number][0]]
                  for key, value in the_metrics.items()
                  if key.startswith('int90') and not key.endswith('mean')}))
        if ('crps' in the_metrics.keys() and 'mape' in the_metrics.keys() and
                'rmse' in the_metrics.keys() and 'mae' in the_metrics.keys()):
            pd_others.append(pd.DataFrame(
                data={os_str('crps', number): [the_metrics['crps'][number][0]],
                      os_str('rmse', number): [the_metrics['rmse'][0][number]],
                      os_str('mae', number): [the_metrics['mae'][0][number]],
                      os_str('mape', number): [the_metrics['mape'][0][number]]}))
    return_list = pd_quants + pd_int90 + pd_others
    if not return_list:
        return pd.DataFrame()
    return pd.concat(return_list, axis=1)


def build_args_dataframe(args):
    """Return a row dataframe with args."""
    cols = []
    vals = []
    for one, two in args.items():
        cols.append(one)
        vals.append(two)
    return pd.DataFrame(data=[vals], columns=cols)


def build_floats_df(all_losses, include_quant=True, max_number=24):
    """Get float metrics from all losses list."""
    ret = pd.concat([
        pd.concat([build_args_dataframe(args),
                   get_float_metrics_df(the_metrics),
                   get_one_step_metrics_df(the_metrics,
                                           include_quant=include_quant,
                                           max_number=max_number)],
                  axis=1)
        for args, the_metrics in all_losses], axis=0)
    ret['file_name'] = [sss.split('/')[-1] for sss in ret['file_name']]
    return ret


def select_best(the_metrics, sort_by='rmse_mean', filter_zero=False):
    """Extract a filtered table with sorted results."""
    group_columns = [
        'file_name',
        'input_series',
        'split_position',
        'nn_use_variable_sigma',
        'nn_output_distribution',
        'has_dropout',
    ]
    the_metrics = the_metrics.copy()
    if filter_zero:
        the_metrics = the_metrics.loc[the_metrics.loc[:, sort_by] > 0.0, :]
    return the_metrics.sort_values(sort_by).drop_duplicates(group_columns)


def build_tables(the_metrics, sort_by='rmse_mean'):
    """Extract a filtered table with sorted results."""
    group_columns = [
        'input_series',
        'split_position',
        'nn_use_variable_sigma',
        'nn_output_distribution',
        'nn_dropout_output',
        'nn_learning_rate',
        'lstm_layers',
        'lstm_nodes',
        'nn_l2_regularizer',
    ]
    the_metrics = the_metrics.copy()
    return the_metrics.loc[
        :,
        group_columns +
        ['rmse_mean', 'pinballc', 'quant10_pinball_mean',
         'int90_winkler_mean', 'int90_norm_winkler_mean',
         'int90_picp_mean', 'int90_nmpiw_mean', 'int90_cwc_mean', 'mae_mean',
         'mape_mean']].groupby(group_columns).mean().sort_values(sort_by)


def chart_train_test(the_case, steps, prediction_step,
                     train='train', test='test', boxplot=False, y_max=275.0):
    """Draw charts of end of train and start of test."""
    y_train = the_case['y_{}'.format(train)][-steps:, prediction_step, 0]
    y_test = the_case['y_{}'.format(test)][:steps, prediction_step, 0]
    train_output_sample = the_case['{}_output'.format(train)][0][
        :, -steps:, prediction_step, 0]
    test_output_sample = the_case['{}_output'.format(test)][0][
        :, :steps, prediction_step, 0]
    train_output = train_output_sample.mean(axis=0)
    test_output = test_output_sample.mean(axis=0)
    print('SERIES')
    pyplot.figure(figsize=(12, 4))
    pyplot.plot(np.linspace(1, steps, num=steps), y_train,
                label='{} real'.format(train))
    pyplot.plot(np.linspace(1, steps, num=steps), train_output,
                label='{} predicted'.format(train))
    pyplot.plot(np.linspace(steps + 1, 2 * steps, num=steps), y_test,
                label='{} real'.format(test))
    pyplot.plot(np.linspace(steps + 1, 2 * steps, num=steps), test_output,
                label='{} predicted'.format(test))
    pyplot.xlim(0.5, 2 * steps + 0.5)
    pyplot.ylim(0.0, y_max)
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.show()
    print('INTERVAL')
    pyplot.figure(figsize=(12, 4))
    pyplot.plot(np.linspace(1, steps, num=steps), y_train,
                label='{} real'.format(train))
    pyplot.plot(np.linspace(steps + 1, 2 * steps, num=steps), y_test,
                label='{} real'.format(test))
    train_int = np.percentile(train_output_sample, [5, 95], axis=0)
    test_int = np.percentile(test_output_sample, [5, 95], axis=0)
    pyplot.plot(np.linspace(1, steps, num=steps), train_int[0, :])
    pyplot.plot(np.linspace(1, steps, num=steps), train_int[1, :])
    pyplot.plot(np.linspace(steps + 1, 2 * steps, num=steps), test_int[0, :])
    pyplot.plot(np.linspace(steps + 1, 2 * steps, num=steps), test_int[1, :])
    pyplot.ylim(0.0, y_max)
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.show()
    if boxplot:
        print('BOXPLOTS')
        pyplot.figure(figsize=(12, 4))
        pyplot.plot(np.linspace(1, steps, num=steps), y_train)
        pyplot.plot(np.linspace(steps + 1, 2 * steps, num=steps), y_test)
        pyplot.boxplot(np.concatenate(
            [train_output_sample, test_output_sample], axis=1))
        pyplot.ylim(0.0, y_max)
        pyplot.tight_layout()
        pyplot.show()
    pyplot.close()


def chart_forecasts_three(last_pre, start, steps, the_case1, label1,
                          the_case2=None, label2=None,
                          the_case3=None, label3=None):
    """Draw forecasting charts for some cases."""
    total_length = start + steps
    y_inp = the_case1['y_test'][last_pre - start:last_pre, 0, 0]
    y_out = the_case1['y_test'][last_pre, :steps, 0]
    val_output_sample1 = the_case1['test_output'][0][:, last_pre, :steps, 0]
    val_output1 = val_output_sample1.mean(axis=0)
    val_output1 = np.concatenate([y_inp[-1] * np.ones((1,)), val_output1],
                                 axis=0)
    val_output2 = None
    val_output3 = None
    val_output_sample2 = None
    val_output_sample3 = None
    if the_case2:
        val_output_sample2 = the_case2['test_output'][0][:, last_pre, :steps,
                                                         0]
        val_output2 = val_output_sample2.mean(axis=0)
        val_output2 = np.concatenate([y_inp[-1] * np.ones((1,)), val_output2],
                                     axis=0)
    if the_case3:
        val_output_sample3 = the_case3['test_output'][0][:, last_pre, :steps,
                                                         0]
        val_output3 = val_output_sample3.mean(axis=0)
        val_output3 = np.concatenate([y_inp[-1] * np.ones((1,)), val_output3],
                                     axis=0)
    figsize = (10, 5)
    steps += 1

    def draw_final_axes(ylim_max=None):
        """Finish chart."""
        pyplot.xlim(0.5, total_length + 0.5)
        # pyplot.ylim(-0.2, 12.0 if ylim_max is None else ylim_max)
        pyplot.tick_params(labelsize=14)
        pyplot.legend(prop={'size': 14})
        pyplot.tight_layout()
        pyplot.show()

    print('SERIES')
    pyplot.figure(figsize=figsize)
    pyplot.plot(np.linspace(1, total_length, num=total_length),
                np.concatenate([y_inp, y_out], axis=0),
                label='data', color='red')
    pyplot.plot(np.linspace(start, total_length, num=steps),
                val_output1, label=label1, color='orange')
    if the_case2:
        pyplot.plot(np.linspace(start, total_length, num=steps),
                    val_output2, label=label2, color='blue')
    if the_case3:
        pyplot.plot(np.linspace(start, total_length, num=steps),
                    val_output3, label=label3, color='green')
    draw_final_axes()
    # print('BOXPLOTS')
    # pyplot.figure(figsize=figsize)
    # pyplot.plot(np.linspace(1, last_pre, num=last_pre), y_pre1)
    # pyplot.plot(np.linspace(last_pre + 1, total_length, num=last_pre),
    #             y_val1)
    # pyplot.boxplot(np.concatenate([pre_output_sample1, val_output_sample1],
    #                axis=1))
    # pyplot.ylim(100.0, 280.0)
    # pyplot.tight_layout()
    # pyplot.show()
    print('INTERVAL')
    val_int1 = np.percentile(val_output_sample1, [5, 95], axis=0)
    val_int1 = np.concatenate([y_inp[-1] * np.ones((2, 1)), val_int1], axis=1)
    val_int2 = None
    val_int3 = None
    if the_case2:
        val_int2 = np.percentile(val_output_sample2, [5, 95], axis=0)
        val_int2 = np.concatenate([y_inp[-1] * np.ones((2, 1)), val_int2],
                                  axis=1)
    if the_case3:
        val_int3 = np.percentile(val_output_sample3, [5, 95], axis=0)
        val_int3 = np.concatenate([y_inp[-1] * np.ones((2, 1)), val_int3],
                                  axis=1)
    pyplot.figure(figsize=figsize)
    pyplot.plot(np.linspace(1, total_length, num=total_length),
                np.concatenate([y_inp, y_out], axis=0),
                label='data', color='red')
    pyplot.fill_between(np.linspace(start, total_length, num=steps),
                        val_int1[0, :], y2=val_int1[1, :],
                        color='orange', alpha=0.3)
    pyplot.plot(np.linspace(start, total_length, num=steps),
                val_output1, label=label1, color='orange', linewidth=2.0)
    if the_case2:
        pyplot.fill_between(np.linspace(start, total_length, num=steps),
                            val_int2[0, :], y2=val_int2[1, :],
                            color='blue', alpha=0.3)
        pyplot.plot(np.linspace(start, total_length, num=steps),
                    val_output2, label=label2, color='blue', linewidth=2.0)
    if the_case3:
        pyplot.fill_between(np.linspace(start, total_length, num=steps),
                            val_int3[0, :], y2=val_int3[1, :],
                            color='green', alpha=0.3)
        pyplot.plot(np.linspace(start, total_length, num=steps),
                    val_output3, label=label3, color='green', linewidth=2.0)
    draw_final_axes()
    print('EMPIRICAL STANDARD DEVIATION')
    pyplot.figure(figsize=figsize)
    pyplot.plot(np.linspace(start, total_length, num=steps),
                (val_int1[1, :] - val_output1) / 1.645,
                label=label1, color='orange')
    if the_case2:
        pyplot.plot(np.linspace(start, total_length, num=steps),
                    (val_int2[1, :] - val_output2) / 1.645,
                    label=label2, color='blue')
    if the_case3:
        pyplot.plot(np.linspace(start, total_length, num=steps),
                    (val_int3[1, :] - val_output3) / 1.645,
                    label=label3, color='green')
    draw_final_axes(ylim_max=3.0)
    # print('INTERVAL UP VS DOWN')
    # pyplot.figure(figsize=figsize)
    # pyplot.plot(np.linspace(last_pre + 1, 2 * last_pre, num=last_pre),
    #             val_int1[0, :] + val_int1[1, :] - 2 * val_output1,
    #             label='test')
    # draw_final_axes()
    print('RMSE')
    pyplot.figure(figsize=figsize)
    y_out = np.concatenate([[y_inp[-1]], y_out])
    pyplot.plot(np.linspace(start, total_length, num=steps),
                np.sqrt(np.square(val_output1 - y_out)),
                label=label1, color='orange')
    if the_case2:
        pyplot.plot(np.linspace(start, total_length, num=steps),
                    np.sqrt(np.square(val_output2 - y_out)),
                    label=label2, color='blue')
    if the_case3:
        pyplot.plot(np.linspace(start, total_length, num=steps),
                    np.sqrt(np.square(val_output3 - y_out)),
                    label=label3, color='green')
    draw_final_axes(ylim_max=3.0)
    pyplot.close()


def chart_one_step_three(last_pre, step, the_case1, label1, the_case2=None,
                         label2=None, the_case3=None, label3=None):
    """Draw charts with train and test data for some cases."""
    total_length = 2 * last_pre
    y_pre = the_case1['y_train'][-last_pre:, step, 0]
    y_val = the_case1['y_test'][:last_pre, step, 0]
    pre_output_sample1 = the_case1['train_output'][0][:, -last_pre:, step, 0]
    val_output_sample1 = the_case1['test_output'][0][:, :last_pre, step, 0]
    pre_output_sample2 = None
    val_output_sample2 = None
    pre_output2 = None
    val_output2 = None
    pre_output_sample3 = None
    val_output_sample3 = None
    pre_output3 = None
    val_output3 = None
    pre_output1 = pre_output_sample1.mean(axis=0)
    val_output1 = val_output_sample1.mean(axis=0)
    if the_case2:
        pre_output_sample2 = the_case2['train_output'][0][:, -last_pre:, step,
                                                          0]
        val_output_sample2 = the_case2['test_output'][0][:, :last_pre, step, 0]
        pre_output2 = pre_output_sample2.mean(axis=0)
        val_output2 = val_output_sample2.mean(axis=0)
    if the_case3:
        pre_output_sample3 = the_case3['train_output'][0][:, -last_pre:, step,
                                                          0]
        val_output_sample3 = the_case3['test_output'][0][:, :last_pre, step, 0]
        pre_output3 = pre_output_sample3.mean(axis=0)
        val_output3 = val_output_sample3.mean(axis=0)
    figsize = (10, 5)

    def draw_final_axes(ylim_max=None):
        """Finish chart."""
        pyplot.text(14 * last_pre / 16, 0, 'train', fontsize='xx-large')
        pyplot.text(17.1 * last_pre / 16, 0, 'test', fontsize='xx-large')
        pyplot.axvline(x=(last_pre + 0.5), dashes=[1, 1])
        pyplot.xlim(0.5, total_length + 0.5)
        # pyplot.ylim(-0.2, 12.0 if ylim_max is None else ylim_max)
        pyplot.legend()
        pyplot.tight_layout()
        pyplot.show()

    print('SERIES')
    pyplot.figure(figsize=figsize)
    pyplot.plot(np.linspace(1, total_length, num=total_length),
                np.concatenate([y_pre, y_val], axis=0),
                label='data', color='red')
    pyplot.plot(np.linspace(1, total_length, num=total_length),
                np.concatenate([pre_output1, val_output1], axis=0),
                label=label1, color='orange')
    if the_case2:
        pyplot.plot(np.linspace(1, total_length, num=total_length),
                    np.concatenate([pre_output2, val_output2], axis=0),
                    label=label2, color='blue')
    if the_case3:
        pyplot.plot(np.linspace(1, total_length, num=total_length),
                    np.concatenate([pre_output3, val_output3], axis=0),
                    label=label3, color='green')
    draw_final_axes()
    # print('BOXPLOTS')
    # pyplot.figure(figsize=figsize)
    # pyplot.plot(np.linspace(1, last_pre, num=last_pre), y_pre1)
    # pyplot.plot(np.linspace(last_pre + 1, total_length, num=last_pre),
    #             y_val1)
    # pyplot.boxplot(np.concatenate([pre_output_sample1, val_output_sample1],
    #                               axis=1))
    # pyplot.ylim(100.0, 280.0)
    # pyplot.show()
    print('INTERVAL')
    pre_int1 = np.percentile(pre_output_sample1, [5, 95], axis=0)
    val_int1 = np.percentile(val_output_sample1, [5, 95], axis=0)
    pre_int2 = None
    val_int2 = None
    pre_int3 = None
    val_int3 = None
    if the_case2:
        pre_int2 = np.percentile(pre_output_sample2, [5, 95], axis=0)
        val_int2 = np.percentile(val_output_sample2, [5, 95], axis=0)
    if the_case3:
        pre_int3 = np.percentile(pre_output_sample3, [5, 95], axis=0)
        val_int3 = np.percentile(val_output_sample3, [5, 95], axis=0)
    pyplot.figure(figsize=figsize)
    pyplot.plot(np.linspace(1, total_length, num=total_length),
                np.concatenate([y_pre, y_val], axis=0),
                label='data', color='red')
    pyplot.plot(np.linspace(1, total_length, num=total_length),
                np.concatenate([pre_int1[0, :], val_int1[0, :]], axis=0),
                label=('5% {}'.format(label1)), color='orange')
    if the_case2:
        pyplot.plot(np.linspace(1, total_length, num=total_length),
                    np.concatenate([pre_int2[0, :], val_int2[0, :]], axis=0),
                    label=('5% {}'.format(label2)), color='blue')
    if the_case3:
        pyplot.plot(np.linspace(1, total_length, num=total_length),
                    np.concatenate([pre_int3[0, :], val_int3[0, :]], axis=0),
                    label=('5% {}'.format(label3)), color='green')
    pyplot.plot(np.linspace(1, total_length, num=total_length),
                np.concatenate([pre_int1[1, :], val_int1[1, :]], axis=0),
                label=('95% {}'.format(label1)), color='orange')
    if the_case2:
        pyplot.plot(np.linspace(1, total_length, num=total_length),
                    np.concatenate([pre_int2[1, :], val_int2[1, :]], axis=0),
                    label=('95% {}'.format(label2)), color='blue')
    if the_case3:
        pyplot.plot(np.linspace(1, total_length, num=total_length),
                    np.concatenate([pre_int3[1, :], val_int3[1, :]], axis=0),
                    label=('95% {}'.format(label3)), color='green')
    # pyplot.plot(np.linspace(1, total_length, num=total_length),
    #             np.concatenate([pre_output1, val_output1], axis=0),
    #             label=('mean {}'.format(label3))
    draw_final_axes()
    print('EMPIRICAL STANDARD DEVIATION')
    pyplot.figure(figsize=figsize)
    # pyplot.plot(np.linspace(1, total_length, num=total_length),
    #             np.concatenate([pre_int1[0, :] - pre_output1,
    #                             val_int1[0, :] - val_output1], axis=0),
    #             label=('5% {}'.format(label1)))
    # pyplot.plot(np.linspace(1, total_length, num=total_length),
    #             np.concatenate([pre_int2[0, :] - pre_output2,
    #                             val_int2[0, :] - val_output2], axis=0),
    #             label=('5% {}'.format(label2)))
    # pyplot.plot(np.linspace(1, total_length, num=total_length),
    #             np.concatenate([pre_int3[0, :] - pre_output3,
    #                             val_int3[0, :] - val_output3], axis=0),
    #             label=('5% {}'.format(label3)))
    pyplot.plot(np.linspace(1, total_length, num=total_length),
                np.concatenate([(pre_int1[1, :] - pre_output1) / 1.645,
                                (val_int1[1, :] - val_output1) / 1.645],
                               axis=0),
                label=label1, color='orange')
    if the_case2:
        pyplot.plot(np.linspace(1, total_length, num=total_length),
                    np.concatenate([(pre_int2[1, :] - pre_output2) / 1.645,
                                    (val_int2[1, :] - val_output2) / 1.645],
                                   axis=0),
                    label=label2, color='blue')
    if the_case3:
        pyplot.plot(np.linspace(1, total_length, num=total_length),
                    np.concatenate([(pre_int3[1, :] - pre_output3) / 1.645,
                                    (val_int3[1, :] - val_output3) / 1.645],
                                   axis=0),
                    label=label3, color='green')
    draw_final_axes(ylim_max=5.0)
    # print('INTERVAL UP VS DOWN')
    # pyplot.figure(figsize=figsize)
    # pyplot.plot(np.linspace(1, last_pre, num=last_pre),
    #             pre_int1[0, :] + pre_int1[1, :] - 2 * pre_output1,
    #             label='train')
    # pyplot.plot(np.linspace(last_pre + 1, 2 * last_pre, num=last_pre),
    #             val_int1[0, :] + val_int1[1, :] - 2 * val_output1,
    #             label='test')
    # draw_final_axes()
    print('RMSE')
    pyplot.figure(figsize=figsize)
    pyplot.plot(np.linspace(1, total_length, num=total_length),
                np.concatenate([np.sqrt(np.square(pre_output1 - y_pre)),
                                np.sqrt(np.square(val_output1 - y_val))],
                               axis=0),
                label=label1, color='orange')
    if the_case2:
        pyplot.plot(np.linspace(1, total_length, num=total_length),
                    np.concatenate([np.sqrt(np.square(pre_output2 - y_pre)),
                                    np.sqrt(np.square(val_output2 - y_val))],
                                   axis=0),
                    label=label2, color='blue')
    if the_case3:
        pyplot.plot(np.linspace(1, total_length, num=total_length),
                    np.concatenate([np.sqrt(np.square(pre_output3 - y_pre)),
                                    np.sqrt(np.square(val_output3 - y_val))],
                                   axis=0),
                    label=label3, color='green')
    draw_final_axes(ylim_max=5.0)
    pyplot.close()


def one_data(file_data, dataset_name, features, column_name,
             block=None, consider_preprocess=False, max_number=4):
    """Get one piece of data to draw charts comparing metrics."""
    the_data = file_data[file_data['file_name'] == DATASETS_DICT[dataset_name]]
    the_data = the_data[the_data['ensemble_number_models'] == 1]
    the_data = the_data[the_data['mc_model_dropout'] == features['drop_mc']]
    the_data = the_data[the_data['has_dropout'] == features['dropout']]
    if consider_preprocess:
        the_data = the_data[the_data['preprocess'] == features['preprocess']]
    the_data = the_data[the_data['nn_output_distribution'] ==
                        features['output_distribution']]
    the_data = the_data[the_data['model'] == features['model']]
    the_data = the_data[the_data['sequential_mini_step'] ==
                        features['sequential_mini_step']]
    if block is not None:
        the_data = the_data[the_data['split_position'] == block]
    return_value = the_data[column_name]
    if return_value.shape[0] == 0:
        return np.zeros((max_number, 1))
    return_value = return_value[:max_number].values.reshape((max_number, 1))
    if column_name in ['mean_int90_cwc_mean',
                       'mean_int90_cwc_mean_test',
                       'std_int90_cwc_mean',
                       'std_int90_cwc_mean_test']:
        return_value = np.log(return_value + 1.0)
    return return_value


def get_file_name(block):
    """Get correct file name to read metrics from."""
    if block is None:
        return '{}_means_d.csv'
    return '{}_blocks_d.csv'


def draw_one(metric_name, draw_metric, dataset_name, series_features,
             is_test=True, block=None, consider_preprocess=False):
    """Draw one line of metrics according to given parameters."""
    metrics_table = pd.read_csv(get_file_name(block).format(metric_name))
    mean_metric_name = 'mean_{}_mean{}'.format(draw_metric,
                                               '_test' if is_test else '')
    std_metric_name = 'std_{}_mean{}'.format(draw_metric,
                                             '_test' if is_test else '')
    vals = pd.DataFrame(np.concatenate([
        one_data(metrics_table, dataset_name, features, mean_metric_name,
                 block=block, consider_preprocess=consider_preprocess)
        for features in series_features], axis=1))
    errs = pd.DataFrame(np.concatenate([
        one_data(metrics_table, dataset_name, features, std_metric_name,
                 block=block, consider_preprocess=consider_preprocess)
        for features in series_features], axis=1))
    vals.columns = [features['label'] for features in series_features]
    errs.columns = [features['label'] for features in series_features]
    the_index = list(range(1, 5))
    vals.index = the_index
    errs.index = the_index
    pyplot.figure(figsize=(8, 5))
    for features in series_features:
        pyplot.fill_between(
            the_index, vals[features['label']] - errs[features['label']],
            y2=(vals[features['label']] + errs[features['label']]), alpha=0.3,
            hatch=features['hatch'])
    for features in series_features:
        pyplot.plot(the_index, vals[features['label']],
                    label=features['label'], linewidth=3.0)
    pyplot.ylabel(draw_metric)
    pyplot.xlabel('Time steps ahead')
    pyplot.legend()
    title = '{} {} metric {} selected by {} {}'.format(
        dataset_name, block, draw_metric, metric_name,
        'TRAIN' if not is_test else '')
    print(title)
    pyplot.title(title)
    pyplot.tight_layout()
    # pyplot.savefig('winkres_{} metric {} selected by {}.eps'.format(
    #     dataset_name, draw_metric, metric_name))
    pyplot.show()
    pyplot.close()
