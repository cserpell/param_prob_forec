# coding=utf-8
"""Tests for prepare_data module."""
import unittest

import numpy as np
from numpy import testing
import pandas as pd

from modules import data


class DataTest(unittest.TestCase):
    """Unit test for prepare_data module."""

    def setUp(self):
        """Build common data for this test."""
        self._data = pd.DataFrame(data={
            's1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            's2': [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            's3': [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0]})
        time_index = pd.date_range(start='2018-01-01T00:00',
                                   end='2018-01-02T20:00', periods=12)
        self._fillna_df = pd.DataFrame(
            data={'c1': [1, 2, np.NaN, 4, 5, 6, 7, 8, 9, np.NaN, 11, 12],
                  'c2': [np.NaN, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, np.NaN]},
            index=time_index)
        self._x_data = np.array(list(float(num) for num in range(12)))
        self._y_data = np.array(list(float(10 + num) for num in range(12)))

    def test_make_instances_one(self):
        """Test inputs for three inputs, one step, one lag and one output."""
        x_data, y_data = data.make_instances(self._data, 1, 1, 1, ['s1'],
                                             False)
        testing.assert_allclose([[[[1.0, 8.0, 15.0]]], [[[2.0, 9.0, 16.0]]],
                                 [[[3.0, 10.0, 17.0]]], [[[4.0, 11.0, 18.0]]],
                                 [[[5.0, 12.0, 19.0]]], [[[6.0, 13.0, 20.0]]]],
                                x_data)
        testing.assert_allclose([[[2.0]], [[3.0]], [[4.0]],
                                 [[5.0]], [[6.0]], [[7.0]]], y_data)

    def test_make_instances_many(self):
        """Test inputs for 3 inputs, 3 steps, 2 lags and 3 output."""
        x_data, y_data = data.make_instances(self._data, 3, 2, 2,
                                             ['s1', 's2', 's3'], False)
        testing.assert_allclose([2, 3, 2, 3], x_data.shape)
        testing.assert_allclose([[[[1.0, 8.0, 15.0], [2.0, 9.0, 16.0]],
                                  [[2.0, 9.0, 16.0], [3.0, 10.0, 17.0]],
                                  [[3.0, 10.0, 17.0], [4.0, 11.0, 18.0]]],
                                 [[[2.0, 9.0, 16.0], [3.0, 10.0, 17.0]],
                                  [[3.0, 10.0, 17.0], [4.0, 11.0, 18.0]],
                                  [[4.0, 11.0, 18.0], [5.0, 12.0, 19.0]]]],
                                x_data)
        testing.assert_allclose([2, 2, 3], y_data.shape)
        testing.assert_allclose([[[5.0, 12.0, 19.0], [6.0, 13.0, 20.0]],
                                 [[6.0, 13.0, 20.0], [7.0, 14.0, 21.0]]],
                                y_data)

    def test_fill_null_values_last(self):
        """Test fill_null_values method repeating last."""
        data.fill_null_values(self._fillna_df, data.FillMethod.REPEAT_LAST)
        testing.assert_allclose(
            np.transpose(
                [[1, 2, 2, 4, 5, 6, 7, 8, 9, 9, 11, 12],
                 [3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12]]), self._fillna_df)

    def test_fill_null_values_old(self):
        """Test fill_null_values method using old method."""
        data.fill_null_values(self._fillna_df, data.FillMethod.OLD)
        testing.assert_allclose(
            np.transpose(
                [[1, 2, 2, 4, 5, 6, 7, 8, 9, 9, 11, 12],
                 [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12]]), self._fillna_df)

    def test_fill_null_values_daily(self):
        """Test fill_null_values method repeating previous day."""
        data.fill_null_values(self._fillna_df, data.FillMethod.REPEAT_DAILY)
        testing.assert_allclose(
            np.transpose(
                [[1, 2, 9, 4, 5, 6, 7, 8, 9, 4, 11, 12],
                 [8, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 7]]), self._fillna_df)

    def test_non_random(self):
        """Test non random split method without overlap."""
        x_train, x_test, y_train, y_test = data.separate_non_random(
            self._x_data, self._y_data, 3, 1, 0.0, 0.75, -1)
        testing.assert_allclose([0.0, 1.0, 2.0], x_train)
        testing.assert_allclose([10.0, 11.0, 12.0], y_train)
        testing.assert_allclose([3.0], x_test)
        testing.assert_allclose([13.0], y_test)

    def test_fized_test_size(self):
        """Test non random split method without overlap and fixed test size."""
        x_train, x_test, y_train, y_test = data.separate_non_random(
            self._x_data, self._y_data, 3, 1, 0.0, 0.75, 2)
        testing.assert_allclose([0.0, 1.0], x_train)
        testing.assert_allclose([10.0, 11.0], y_train)
        testing.assert_allclose([2.0, 3.0], x_test)
        testing.assert_allclose([12.0, 13.0], y_test)

    def test_non_random_nooverlap(self):
        """Test non random split method without overlap."""
        x_train, x_test, y_train, y_test = data.separate_non_random(
            self._x_data, self._y_data, 3, 1, 0.0, 0.75, -1)
        testing.assert_allclose([0.0, 1.0, 2.0], x_train)
        testing.assert_allclose([10.0, 11.0, 12.0], y_train)
        testing.assert_allclose([3.0], x_test)
        testing.assert_allclose([13.0], y_test)
        x_train, x_test, y_train, y_test = data.separate_non_random(
            self._x_data, self._y_data, 3, 2, 0.0, 0.75, -1)
        testing.assert_allclose([4.0, 5.0, 6.0], x_train)
        testing.assert_allclose([14.0, 15.0, 16.0], y_train)
        testing.assert_allclose([7.0], x_test)
        testing.assert_allclose([17.0], y_test)
        x_train, x_test, y_train, y_test = data.separate_non_random(
            self._x_data, self._y_data, 3, 2, 0.0, 0.5, -1)
        testing.assert_allclose([4.0, 5.0], x_train)
        testing.assert_allclose([14.0, 15.0], y_train)
        testing.assert_allclose([6.0, 7.0], x_test)
        testing.assert_allclose([16.0, 17.0], y_test)

    def test_non_random_overlap(self):
        """Test non random split method with overlap."""
        x_train, x_test, y_train, y_test = data.separate_non_random(
            self._x_data, self._y_data, 3, 1, 0.5, 0.666666666, -1)
        testing.assert_allclose([0.0, 1.0, 2.0, 3.0], x_train)
        testing.assert_allclose([10.0, 11.0, 12.0, 13.0], y_train)
        testing.assert_allclose([4.0, 5.0], x_test)
        testing.assert_allclose([14.0, 15.0], y_test)
        x_train, x_test, y_train, y_test = data.separate_non_random(
            self._x_data, self._y_data, 3, 2, 0.5, 0.666666666, -1)
        testing.assert_allclose([3.0, 4.0, 5.0, 6.0], x_train)
        testing.assert_allclose([13.0, 14.0, 15.0, 16.0], y_train)
        testing.assert_allclose([7.0, 8.0], x_test)
        testing.assert_allclose([17.0, 18.0], y_test)
        x_train, x_test, y_train, y_test = data.separate_non_random(
            self._x_data, self._y_data, 3, 3, 0.5, 0.666666666, -1)
        testing.assert_allclose([6.0, 7.0, 8.0, 9.0], x_train)
        testing.assert_allclose([16.0, 17.0, 18.0, 19.0], y_train)
        testing.assert_allclose([10.0, 11.0], x_test)
        testing.assert_allclose([20.0, 21.0], y_test)


if __name__ == '__main__':
    unittest.main()
