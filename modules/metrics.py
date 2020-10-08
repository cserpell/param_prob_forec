# coding=utf-8
"""Module to compute metrics over evaluated model."""
import logging

from tensorflow.python.keras import backend
from tensorflow.keras import losses
import numpy as np


METRICS_NAMES = [
    'quant5_pinball_mean',
    'quant5_winkler_mean',
    'quant5_norm_winkler_mean',
    'quant5_picp_mean',
    'quant5_nmpiw_mean',
    'quant5_cwc_mean',
    'quant5_apd_mean',
    'quant5_pinball',
    'quant5_winkler',
    'quant5_norm_winkler',
    'quant5_picp',
    'quant5_nmpiw',
    'quant5_cwc',
    'quant5_apd',
    'quant5_mae_mean',
    'quant5_rmse_mean',
    'quant5_mape_mean',
    'quant5_mae',
    'quant5_rmse',
    'quant5_mape',
    'quant10_pinball_mean',
    'quant10_winkler_mean',
    'quant10_norm_winkler_mean',
    'quant10_picp_mean',
    'quant10_nmpiw_mean',
    'quant10_cwc_mean',
    'quant10_apd_mean',
    'quant10_pinball',
    'quant10_winkler',
    'quant10_norm_winkler',
    'quant10_picp',
    'quant10_nmpiw',
    'quant10_cwc',
    'quant10_apd',
    'quant10_mae_mean',
    'quant10_rmse_mean',
    'quant10_mape_mean',
    'quant10_mae',
    'quant10_rmse',
    'quant10_mape',
    'quant20_pinball_mean',
    'quant20_winkler_mean',
    'quant20_norm_winkler_mean',
    'quant20_picp_mean',
    'quant20_nmpiw_mean',
    'quant20_cwc_mean',
    'quant20_apd_mean',
    'quant20_pinball',
    'quant20_winkler',
    'quant20_norm_winkler',
    'quant20_picp',
    'quant20_nmpiw',
    'quant20_cwc',
    'quant20_apd',
    'quant20_mae_mean',
    'quant20_rmse_mean',
    'quant20_mape_mean',
    'quant20_mae',
    'quant20_rmse',
    'quant20_mape',
    'int90_winkler_mean',
    'int90_norm_winkler_mean',
    'int90_picp_mean',
    'int90_nmpiw_mean',
    'int90_cwc_mean',
    'int90_pimse_mean',
    'int90_ncwc_mean',
    'int90_pird_mean',
    'int90_pinrw_mean',
    'int90_awd_mean',
    'int90_ace_mean',
    'int90_winkler',
    'int90_norm_winkler',
    'int90_picp',
    'int90_nmpiw',
    'int90_cwc',
    'int90_pimse',
    'int90_ncwc',
    'int90_pird',
    'int90_pinrw',
    'int90_awd',
    'int90_ace',
    'mae_mean',
    'rmse_mean',
    'mape_mean',
    'mae',
    'rmse',
    'mape',
    'crps_mean',
    'crps',
]


def quantiles_cuts(number):
    """Get quantiles percentage cuts."""
    return np.linspace(0.0, 100.0, num=(number + 1))


def quantiles_from_sample(data, number):
    """Produce given number of quantiles from sample data.
    Expected input shape is:
    (number samples, number times, output lags, number series).
    Output includes minimum and maximum values (0.0 and 1.0), and its shape is:
    (number quantiles, number time steps, output lags, number series).
    """
    return np.percentile(data, quantiles_cuts(number), axis=0)


def expected_value_from_sample(data):
    """Get expected value and median from sample data."""
    return np.mean(data, axis=0)


def expected_value_from_quantiles(quantiles):
    """Get estimate of expected value from quantiles data."""
    number_quantiles = quantiles.shape[0] - 1
    return (np.sum(quantiles, axis=0) -
            (quantiles[0] + quantiles[-1]) / 2.0) / number_quantiles


def pinball_all(quantiles, y_data):
    """Compute Pinball Loss - Quantile Score (Rho) for all quantile cuts."""
    each_quantile_shape = [1] + list(y_data.shape[1:])
    pinball_array = []
    for number, quantile in enumerate(quantiles_cuts(quantiles.shape[0] - 1)):
        pinball_val = pinball(quantiles[number], y_data, quantile / 100.0)
        pinball_val.shape = each_quantile_shape
        pinball_array.append(pinball_val)
    return np.concatenate(pinball_array)


def pinball(quantile_cut, y_data, quantile_level):
    """Compute Pinball Loss - Quantile Score (Rho) for quantile cut.
    Input shapes: (test samples, output lags, time series).
    Output shape: (output lags, time series).
    """
    diff_q = y_data - quantile_cut
    return np.multiply(
        diff_q, quantile_level - np.multiply(1.0, diff_q < 0)).mean(axis=0)


def apd_all(quantiles, y_data):
    """Compute Average Proportion Deviation (APD) for all quantile cuts."""
    each_quantile_shape = [1] + list(y_data.shape[1:])
    apd_array = []
    for number, quantile in enumerate(quantiles_cuts(quantiles.shape[0] - 1)):
        apd_val = apd(quantiles[number], y_data, quantile / 100.0)
        apd_val.shape = each_quantile_shape
        apd_array.append(apd_val)
    return np.concatenate(apd_array)


def apd(quantile_cut, y_data, quantile_level):
    """Compute Average Proportion Deviation (APD) for quantile cut.
    Input shapes: (test samples, output lags, time series).
    Output shape: (output lags, time series).
    """
    return (y_data <= quantile_cut).mean(axis=0) - quantile_level


def crps(outputs, y_data):
    """Compute Continuous Ranked Probability Score (CRPS) for all samples."""
    num_samples = outputs.shape[0]
    repeated_y = np.tile(y_data, (num_samples, 1, 1, 1))
    part_1 = np.abs(outputs - repeated_y).mean(axis=0)
    tot_sum = 0.0
    for elem in range(num_samples):
        repeated_y = np.tile(outputs[elem], (num_samples, 1, 1, 1))
        tot_sum += np.abs(outputs - repeated_y).mean(axis=0)
    return (part_1 - 0.5 * tot_sum.mean(axis=0) / num_samples).mean(axis=0)


def alt_winkler(lower, upper, y_data, pinc):
    """Compute alternative Winkler Score (IS), dividing by alpha.
    Input shapes: (test samples, output lags, time series).
    Output shape: (output lags, time series).
    """
    diff_l = lower - y_data
    diff_u = y_data - upper
    diff_ul = upper - lower
    alpha = 1.0 - pinc
    return (diff_ul + 2 * (
        np.multiply(diff_l, diff_l > 0.0) +
        np.multiply(diff_u, diff_u > 0.0)) / alpha).mean(axis=0)


def winkler(lower, upper, y_data, pinc):
    """Compute Interval Sharpness - Interval Score - Winkler Score (IS).
    Input shapes: (test samples, output lags, time series).
    Output shape: (output lags, time series).
    """
    diff_l = lower - y_data
    diff_u = y_data - upper
    diff_ul = upper - lower
    alpha = 1.0 - pinc
    return 2 * (alpha * diff_ul + 2 * (
        np.multiply(diff_l, diff_l > 0.0) +
        np.multiply(diff_u, diff_u > 0.0))).mean(axis=0)


def quantiles_winkler(quantiles, y_data):
    """Compute winkler loss for quantiles against y_data."""
    each_quantile_shape = [1] + list(y_data.shape[1:])
    last_quantile = np.zeros(each_quantile_shape)
    winkler_array = []
    alt_winkler_array = []
    pinc = 1.0 / (quantiles.shape[0] - 1)
    for number in range(quantiles.shape[0]):
        if number < quantiles.shape[0] - 1:
            # Note differences are computed twice (upper[n - 1] = lower[n])
            lower = quantiles[number]
            upper = quantiles[number + 1]
            winkler_val = winkler(lower, upper, y_data, pinc)
            alt_winkler_val = alt_winkler(lower, upper, y_data, pinc)
            winkler_val.shape = each_quantile_shape
            alt_winkler_val.shape = each_quantile_shape
        else:
            winkler_val = last_quantile
            alt_winkler_val = last_quantile
        winkler_array.append(winkler_val)
        alt_winkler_array.append(alt_winkler_val)
    return np.concatenate(winkler_array), np.concatenate(alt_winkler_array)


def max_range(y_data):
    """Compute maximum range per time series.
    Input shape: (test samples, output lags, time series).
    Return shape: (time series).
    """
    return (np.max(np.max(y_data, axis=0), axis=0) -
            np.min(np.min(y_data, axis=0), axis=0))


def picp(lower, upper, y_data):
    """Compute Prediction Interval Coverage Probability (PICP) for interval.
    Input shapes: (test samples, output lags, time series).
    Output shape: (output lags, time series).
    """
    return np.logical_and(upper > y_data, y_data > lower).mean(axis=0)


def ace(lower, upper, y_data, pinc, picp_val=None):
    """Compute Average Coverage Error (ACE) for interval.
    Input shapes: (test samples, output lags, time series).
    Output shape: (output lags, time series).
    """
    if picp_val is None:
        picp_val = picp(lower, upper, y_data)
    return picp_val - pinc


def pinaw(lower, upper, y_data, max_range_val=None):
    """Compute Prediction Interval Normalized Average Width (PINAW).
    Input shapes: (test samples, output lags, time series).
    Output shape: (output lags, time series).
    """
    if max_range_val is None:
        max_range_val = max_range(y_data)
    return (upper - lower).mean(axis=0) / max_range_val


def pinrw(lower, upper, y_data, max_range_val=None):
    """Compute Prediction Interval Normalized Root Mean Square Width (PINRW).
    Input shapes: (test samples, output lags, time series).
    Output shape: (output lags, time series).
    """
    if max_range_val is None:
        max_range_val = max_range(y_data)
    return np.sqrt(np.square(upper - lower).mean(axis=0)) / max_range_val


def pimse(lower, upper, y_data):
    """Compute Prediction Interval Mean Squared Error (PIMSE) for interval.
    Input shapes: (test samples, output lags, time series).
    Output shape: (output lags, time series).
    """
    return (np.square(upper - y_data) + np.square(lower - y_data)).mean(axis=0)


def pird(lower, upper, y_data):
    """Compute Prediction Interval Relative Deviation (PIRD) for interval.
    Input shapes: (test samples, output lags, time series).
    Output shape: (output lags, time series).
    """
    mid_point = (upper + lower) / 2.0
    return np.abs(np.divide(mid_point - y_data, mid_point)).mean(axis=0)


def cwc(lower, upper, y_data, pinc, eta=50.0, picp_val=None, pinaw_val=None):
    """Compute Coverage Width based Criterion (CWC) for interval.
    Input shapes: (test samples, output lags, time series).
    Output shape: (output lags, time series).
    """
    if picp_val is None:
        picp_val = picp(lower, upper, y_data)
    if pinaw_val is None:
        pinaw_val = pinaw(lower, upper, y_data)
    ace_val = ace(lower, upper, y_data, pinc, picp_val=picp_val)
    return np.multiply(pinaw_val, 1.0 + np.multiply(ace_val < 0,
                                                    np.exp(-eta * ace_val)))


def ncwc(lower, upper, y_data, pinc, eta=50.0, cwc_val=None, pimse_val=None):
    """Compute New Coverage Width based Criterion (NCWC) for interval.
    Input shapes: (test samples, output lags, time series).
    Output shape: (output lags, time series).
    """
    if cwc_val is None:
        cwc_val = cwc(lower, upper, y_data, pinc, eta=eta)
    if pimse_val is None:
        pimse_val = pimse(lower, upper, y_data)
    return cwc_val + pimse_val


def awd(lower, upper, y_data):
    """Compute Accumulated Width Deviation (AWD) for interval.
    Input shapes: (test samples, output lags, time series).
    Output shape: (output lags, time series).
    """
    diff_l = lower - y_data
    diff_u = y_data - upper
    diff_ul = upper - lower
    return np.divide(
        np.multiply(diff_l, diff_l > 0.0) + np.multiply(diff_u, diff_u > 0.0),
        diff_ul).mean(axis=0)


def quantiles_cwc(quantiles, y_data, eta=50.0):
    """Compute CWC loss for quantiles against y_data."""
    each_quantile_shape = [1] + list(y_data.shape[1:])
    last_quantile = np.zeros(each_quantile_shape)
    picp_losses_array = []
    pinaw_losses_array = []
    cwc_array = []
    pinc = 1.0 / (quantiles.shape[0] - 1)
    overall_max_range = np.max(y_data) - np.min(y_data)
    if overall_max_range == 0:
        logging.error('CWC loss was not calculated due to constant range')
        return 0.0, 0.0, 0.0
    for number in range(quantiles.shape[0]):
        if number < quantiles.shape[0] - 1:
            # Note differences are computed twice (upper[n - 1] = lower[n])
            lower = quantiles[number]
            upper = quantiles[number + 1]
            picp_val = picp(lower, upper, y_data)
            pinaw_val = pinaw(lower, upper, y_data)
            cwc_val = cwc(lower, upper, y_data, pinc, eta=eta,
                          picp_val=picp_val, pinaw_val=pinaw_val)
            picp_val.shape = each_quantile_shape
            pinaw_val.shape = each_quantile_shape
            cwc_val.shape = each_quantile_shape
        else:
            picp_val = last_quantile
            pinaw_val = last_quantile
            cwc_val = last_quantile
        picp_losses_array.append(picp_val)
        pinaw_losses_array.append(pinaw_val)
        cwc_array.append(cwc_val)
    picp_losses = np.concatenate(picp_losses_array)
    pinaw_losses = np.concatenate(pinaw_losses_array)
    cwc_array = np.concatenate(cwc_array)
    return picp_losses, pinaw_losses, cwc_array


def quantile_metrics(y_data, quantiles):
    """Evaluate quantiles generated by model comparing real values.
    Expected input shape is
    (number quantiles, number times, output lags, number series).
    """
    # point_estimate = expected_value_from_quantiles(quantiles)
    # local_point_metrics = point_metrics(y_data, point_estimate)
    pinball_val = pinball_all(quantiles, y_data)
    pinball_mean = pinball_val.mean()
    # winkler_val, alt_winkler_val = quantiles_winkler(quantiles, y_data)
    # winkler_mean = winkler_val.mean()
    # alt_winkler_mean = alt_winkler_val.mean()
    # picp_val, pinaw_val, cwc_val = quantiles_cwc(quantiles, y_data)
    # picp_mean = picp_val.mean()
    # pinaw_mean = pinaw_val.mean()
    # cwc_mean = cwc_val.mean()
    apd_val = apd_all(quantiles, y_data)
    apd_mean = apd_val.mean()
    return [pinball_mean, None, None, None,
            None, None, apd_mean, pinball_val, None, None,
            None, None, None, apd_val] + [None, None, None, None, None, None]


def interval_metrics(y_data, lower, upper, pinc):
    """Evaluate interval metrics given by lower and upper bounds."""
    winkler_val = winkler(lower, upper, y_data, pinc)
    winkler_mean = winkler_val.mean()
    alt_winkler_val = alt_winkler(lower, upper, y_data, pinc)
    alt_winkler_mean = alt_winkler_val.mean()
    picp_val = picp(lower, upper, y_data)
    pinaw_val = pinaw(lower, upper, y_data)
    cwc_val = cwc(lower, upper, y_data, pinc,
                  picp_val=picp_val, pinaw_val=pinaw_val)
    picp_mean = picp_val.mean()
    pinaw_mean = pinaw_val.mean()
    cwc_mean = cwc_val.mean()
    pimse_val = pimse(lower, upper, y_data)
    pimse_mean = pimse_val.mean()
    ncwc_val = ncwc(lower, upper, y_data, pinc,
                    cwc_val=cwc_val, pimse_val=pimse_val)
    ncwc_mean = ncwc_val.mean()
    pird_val = pird(lower, upper, y_data)
    pird_mean = pird_val.mean()
    pinrw_val = pinrw(lower, upper, y_data)
    pinrw_mean = pinrw_val.mean()
    awd_val = awd(lower, upper, y_data)
    awd_mean = awd_val.mean()
    ace_val = ace(lower, upper, y_data, pinc, picp_val=picp_val)
    ace_mean = ace_val.mean()
    return [winkler_mean, alt_winkler_mean, picp_mean, pinaw_mean, cwc_mean,
            pimse_mean, ncwc_mean, pird_mean, pinrw_mean, awd_mean, ace_mean,
            winkler_val, alt_winkler_val, picp_val, pinaw_val, cwc_val,
            pimse_val, ncwc_val, pird_val, pinrw_val, awd_val, ace_val]


def quantile_metrics_size(y_data, output, size):
    """Get quantile metrics from given number of quantiles."""
    quantiles = quantiles_from_sample(output, size)
    return quantiles, quantile_metrics(y_data, quantiles)


def sampler_metrics(y_data, output):
    """Evaluate sampling generated by model comparing real values."""
    # output has shape (algorithm samples, data samples, out steps, series)
    point_estimate = expected_value_from_sample(output)
    # point_estimate has the average of algorithm samples, with shape
    # (data samples, out steps, series)
    local_point_metrics = point_metrics(y_data, point_estimate)
    _, quantile_metrics_5 = quantile_metrics_size(y_data, output, 5)
    _, quantile_metrics_10 = quantile_metrics_size(y_data, output, 10)
    quantiles, quantile_metrics_20 = quantile_metrics_size(y_data, output, 20)
    local_interval_metrics = interval_metrics(y_data, quantiles[1],
                                              quantiles[19], 0.9)
    crps_val = crps(output, y_data)
    crps_mean = crps_val.mean()
    return (quantile_metrics_5 + quantile_metrics_10 + quantile_metrics_20 +
            local_interval_metrics + local_point_metrics +
            [crps_mean, crps_val])


def point_metrics(y_data, output):
    """Evaluate metrics comparing real and predicted output values."""
    # Reshaping in memory to avoid duplicate use of RAM
    original_real_shape = y_data.shape
    y_data.shape = (y_data.shape[2], y_data.shape[1], y_data.shape[0])
    original_output_shape = output.shape
    output.shape = (output.shape[2], output.shape[1], output.shape[0])
    mae = losses.mean_absolute_error(y_data, output)
    mse = losses.mean_squared_error(y_data, output)
    mape = losses.mean_absolute_percentage_error(y_data, output)
    try:
        keras_session = backend.get_session()
        mae = mae.eval(session=keras_session)
        mse = mse.eval(session=keras_session)
        mape = mape.eval(session=keras_session)
    except NotImplementedError:
        mae = mae.numpy()
        mse = mse.numpy()
        mape = mape.numpy()
    y_data.shape = original_real_shape
    output.shape = original_output_shape
    return [np.mean(mae), np.sqrt(np.mean(mse)), np.mean(mape),
            mae, np.sqrt(mse), mape]


def get_all(y_data, output):
    """Compute metrics over evaluated dataset.
    Expected shapes:
    y_data: (test samples, output lags, time series),
    output: 1+ x [(evaluation samples, test samples, output lags, time series)]
    """
    the_metrics = sampler_metrics(y_data, np.vstack(output))
    return {name: metric for name, metric in zip(METRICS_NAMES, the_metrics)}
