# coding=utf-8
"""Preprocessors to apply before training and testing models."""
import abc
import enum

import numpy as np
from sklearn import preprocessing


class Preprocess(enum.Enum):
    """Available preprocessing steps."""
    NONE = 1
    MINMAX = 2  # Same as normalization
    STANDARD = 3  # (value - mean) / std
    # ANOMALIZATION = 4  # (value - mean) / std every N steps (usually 24)
    QUANTILE = 5  # Apply quantile transformation to normal
    LOG = 6  # Apply log to series
    LOG_MINMAX = 7  # Apply log before normalization
    LOG_STANDARD = 8  # Apply log before standardization
    # LOG_ANOMALIZATION = 9  # Apply log before anomalization
    LOG_QUANTILE = 10  # Apply log and then quantile transformation to normal


def build_preprocess(preprocess_kind):
    """Build correct preprocess instance."""
    if preprocess_kind == Preprocess.MINMAX:
        return_instance = MinMax()
    elif preprocess_kind == Preprocess.STANDARD:
        return_instance = Standard()
    elif preprocess_kind == Preprocess.QUANTILE:
        return_instance = NormalQuantile()
    elif preprocess_kind == Preprocess.LOG:
        return_instance = Log()
    elif preprocess_kind == Preprocess.LOG_MINMAX:
        return_instance = LogMinMax()
    elif preprocess_kind == Preprocess.LOG_STANDARD:
        return_instance = LogStandard()
    elif preprocess_kind == Preprocess.LOG_QUANTILE:
        return_instance = LogNormalQuantile()
    else:
        return None
    return return_instance


class Preprocessor(abc.ABC):
    """Perform preprocessing according to defined rule."""

    def __init__(self):
        """Constructor of Preprocessor."""
        super(Preprocessor, self).__init__()
        self._base_sklearn = None
        self._apply_log = False

    def transform(self, input_x):
        """Apply transformation according to current preprocessor."""
        if self._apply_log:
            input_x = np.log(input_x + 1.0)
        if self._base_sklearn is not None:
            input_x = self._base_sklearn.transform(input_x)
        return input_x

    def inverse_transform(self, output_x):
        """Apply reverse transformation."""
        if self._base_sklearn is not None:
            output_x = self._base_sklearn.inverse_transform(output_x)
        if self._apply_log:
            output_x = np.exp(output_x) - 1.0
        return output_x

    def fit_transform(self, input_x):
        """Fit preprocessing transformation."""
        if self._apply_log:
            input_x = np.log(input_x + 1.0)
        if self._base_sklearn is not None:
            input_x = self._base_sklearn.fit_transform(input_x)
        return input_x


class MinMax(Preprocessor):
    """Perform normalization preprocessing."""

    def __init__(self):
        """Constructor of MinMax."""
        super(MinMax, self).__init__()
        self._base_sklearn = preprocessing.MinMaxScaler(feature_range=(0, 1))


class Standard(Preprocessor):
    """Perform standardization preprocessing."""

    def __init__(self):
        """Constructor of Standard."""
        super(Standard, self).__init__()
        self._base_sklearn = preprocessing.StandardScaler()


class NormalQuantile(Preprocessor):
    """Perform preprocessing transforming to normal distribution."""

    def __init__(self):
        """Constructor of Standard."""
        super(NormalQuantile, self).__init__()
        self._base_sklearn = preprocessing.QuantileTransformer(
            output_distribution='normal')


class Log(Preprocessor):
    """Perform preprocessing applying log."""

    def __init__(self):
        """Constructor of LogMinMax."""
        super(Log, self).__init__()
        self._apply_log = True


class LogMinMax(MinMax):
    """Perform normalization preprocessing after log."""

    def __init__(self):
        """Constructor of LogMinMax."""
        super(LogMinMax, self).__init__()
        self._apply_log = True


class LogStandard(Standard):
    """Perform standardization preprocessing after log."""

    def __init__(self):
        """Constructor of LogStandard."""
        super(LogStandard, self).__init__()
        self._apply_log = True


class LogNormalQuantile(Preprocessor):
    """Perform preprocessing transforming to normal distribution after log."""

    def __init__(self):
        """Constructor of Standard."""
        super(LogNormalQuantile, self).__init__()
        self._apply_log = True
