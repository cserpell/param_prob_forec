# coding=utf-8
"""Tests for metrics module."""
import math
import unittest

import numpy as np
from numpy import testing

from modules import metrics


class MetricsTest(unittest.TestCase):
    """Unit test for metrics module."""

    @staticmethod
    def test_expected_value_from_sample():
        """Test expected_value_from_sample method."""
        testing.assert_allclose(
            metrics.expected_value_from_sample([[1.0, 2.0, 3.0],
                                                [1.5, 2.6, 4.0]]),
            [1.25, 2.3, 3.5])
        testing.assert_allclose(
            metrics.expected_value_from_sample([[1.0, 2.0, 3.0]]),
            [1.0, 2.0, 3.0])
        testing.assert_allclose(
            metrics.expected_value_from_sample([[[1.0, 2.0], [3.4, 3.0]],
                                                [[1.5, 2.6], [3.2, 4.0]]]),
            [[1.25, 2.3], [3.3, 3.5]])

    @staticmethod
    def test_quantiles_cuts():
        """Test quantiles_cuts method."""
        testing.assert_allclose(metrics.quantiles_cuts(1), [0.0, 100.0])
        testing.assert_allclose(metrics.quantiles_cuts(2), [0.0, 50.0, 100.0])
        testing.assert_allclose(metrics.quantiles_cuts(4),
                                [0.0, 25.0, 50.0, 75.0, 100.0])

    @staticmethod
    def test_quantiles_from_sample():
        """Test quantiles_from_sample method."""
        data = [[[1.0, 10.0, -1.0]],
                [[2.0, 9.0, -0.5]],
                [[3.0, 8.0, 0.0]],
                [[4.0, 7.0, 0.5]],
                [[5.0, 6.0, 0.7]],
                [[6.0, 5.0, 0.8]],
                [[7.0, 4.0, 1.0]],
                [[8.0, 3.0, 1.7]],
                [[9.0, 2.0, 2.0]],
                [[10.0, 1.0, 1.1]],
                [[11.0, 0.0, 1.15]],
                [[12.0, -1.0, 1.2]]]
        testing.assert_allclose(
            metrics.quantiles_from_sample(data, 4),
            [[[1.0, -1.0, -1.0]],
             [[3.75, 1.75, 0.375]],
             [[6.5, 4.5, 0.9]],
             [[9.25, 7.25, 1.1625]],
             [[12.0, 10.0, 2.0]]])

    @staticmethod
    def test_crps():
        """Test crps method."""
        y_data = np.array([[[1.0, 10.0, -1.0]],
                           [[10.0, 0.1, 1.2]],
                           [[11.0, 0.5, 1.15]],
                           [[12.0, -1.0, 1.2]]])
        samples = np.array([[[[2.5, 0.0, 0.0]],
                             [[2.5, 0.0, 0.0]],
                             [[2.5, 0.0, 0.0]],
                             [[2.5, 0.0, 0.0]]],
                            [[[11.5, 9.0, 1.18]],
                             [[11.5, 9.0, 1.18]],
                             [[11.5, 9.0, 1.18]],
                             [[11.5, 9.0, 1.18]]]])
        testing.assert_allclose(
            metrics.crps(samples, y_data), [[
                ((1.5 + 10.5) / 2.0 - 0.5 * (9.0 + 9.0) / 4.0) / 4.0 +
                ((7.5 + 1.5) / 2.0 - 0.5 * (9.0 + 9.0) / 4.0) / 4.0 +
                ((8.5 + 0.5) / 2.0 - 0.5 * (9.0 + 9.0) / 4.0) / 4.0 +
                ((9.5 + 0.5) / 2.0 - 0.5 * (9.0 + 9.0) / 4.0) / 4.0,
                ((10.0 + 1.0) / 2.0 - 0.5 * (9.0 + 9.0) / 4.0) / 4.0 +
                ((0.1 + 8.9) / 2.0 - 0.5 * (9.0 + 9.0) / 4.0) / 4.0 +
                ((0.5 + 8.5) / 2.0 - 0.5 * (9.0 + 9.0) / 4.0) / 4.0 +
                ((1.0 + 10.0) / 2.0 - 0.5 * (9.0 + 9.0) / 4.0) / 4.0,
                ((1.0 + 2.18) / 2.0 - 0.5 * (1.18 + 1.18) / 4.0) / 4.0 +
                ((1.2 + 0.02) / 2.0 - 0.5 * (1.18 + 1.18) / 4.0) / 4.0 +
                ((1.15 + 0.03) / 2.0 - 0.5 * (1.18 + 1.18) / 4.0) / 4.0 +
                ((1.2 + 0.02) / 2.0 - 0.5 * (1.18 + 1.18) / 4.0) / 4.0]])

    @staticmethod
    def test_picp():
        """Test picp method."""
        y_data = np.array([[[1.0, 10.0, -1.0]],
                           [[10.0, 0.1, 1.2]],
                           [[11.0, 0.5, 1.15]],
                           [[12.0, -1.0, 1.2]]])
        lower = np.array([[[2.5, 0.0, 0.0]],
                          [[2.5, 0.0, 0.0]],
                          [[2.5, 0.0, 0.0]],
                          [[2.5, 0.0, 0.0]]])
        upper = np.array([[[11.5, 9.0, 1.18]],
                          [[11.5, 9.0, 1.18]],
                          [[11.5, 9.0, 1.18]],
                          [[11.5, 9.0, 1.18]]])
        testing.assert_allclose(
            metrics.picp(lower, upper, y_data), [[0.5, 0.5, 0.25]])

    @staticmethod
    def test_pinaw():
        """Test pinaw method."""
        y_data = np.array([[[1.0, 10.0, -1.0]],
                           [[10.0, 0.1, 1.2]],
                           [[11.0, 0.5, 1.15]],
                           [[12.0, -1.0, 1.2]]])
        lower = np.array([[[2.5, 0.0, 0.0]],
                          [[2.5, 0.0, 0.0]],
                          [[2.5, 0.0, 0.0]],
                          [[2.5, 0.0, 0.0]]])
        upper = np.array([[[11.5, 9.0, 1.18]],
                          [[11.5, 9.0, 1.18]],
                          [[11.5, 9.0, 1.18]],
                          [[11.5, 9.0, 1.18]]])
        testing.assert_allclose(
            metrics.pinaw(lower, upper, y_data),
            [[9.0 / 11.0, 9.0 / 11.0, 1.18 / 2.2]])

    @staticmethod
    def test_pinrw():
        """Test pinrw method."""
        y_data = np.array([[[1.0, 10.0, -1.0]],
                           [[10.0, 0.1, 1.2]],
                           [[11.0, 0.5, 1.15]],
                           [[12.0, -1.0, 1.2]]])
        lower = np.array([[[2.5, 0.0, 0.0]],
                          [[2.5, 0.0, 0.0]],
                          [[2.5, 0.0, 0.0]],
                          [[2.5, 0.0, 0.0]]])
        upper = np.array([[[11.5, 9.0, 1.18]],
                          [[11.5, 9.0, 1.18]],
                          [[10.5, 10.0, 1.0]],
                          [[11.5, 9.0, 1.18]]])
        testing.assert_allclose(
            metrics.pinrw(lower, upper, y_data),
            [[8.76071 / 11.0, 9.26013 / 11.0, 1.13767 / 2.2]], rtol=1e-4)

    @staticmethod
    def test_pimse():
        """Test pimse method."""
        y_data = np.array([[[1.0, 10.0, -1.0]],
                           [[10.0, 0.1, 1.2]],
                           [[11.0, 0.5, 1.15]],
                           [[12.0, -1.0, 1.2]]])
        lower = np.array([[[2.5, 0.0, 0.0]],
                          [[2.5, 0.0, 0.0]],
                          [[2.5, 0.0, 0.0]],
                          [[2.5, 0.0, 0.0]]])
        upper = np.array([[[11.5, 9.0, 1.18]],
                          [[11.5, 9.0, 1.18]],
                          [[10.5, 10.0, 1.0]],
                          [[11.5, 9.0, 1.18]]])
        testing.assert_allclose(
            metrics.pimse(lower, upper, y_data), [[83.5, 92.93, 2.49455]])

    @staticmethod
    def test_pird():
        """Test pird method."""
        y_data = np.array([[[1.0, 10.0, -1.0]],
                           [[10.0, 0.1, 1.2]],
                           [[11.0, 0.5, 1.15]],
                           [[12.0, -1.0, 1.2]]])
        lower = np.array([[[2.5, 0.0, 0.0]],
                          [[2.5, 0.0, 0.0]],
                          [[2.5, 0.0, 0.0]],
                          [[2.5, 0.0, 0.0]]])
        upper = np.array([[[11.5, 9.0, 1.18]],
                          [[11.5, 9.0, 1.18]],
                          [[10.5, 10.0, 1.0]],
                          [[11.5, 9.0, 1.18]]])
        testing.assert_allclose(metrics.pird(lower, upper, y_data),
                                [[0.67308, 1.08056, 1.51568]], rtol=1e-4)

    @staticmethod
    def test_awd():
        """Test awd method."""
        y_data = np.array([[[1.0, 10.0, -1.0]],
                           [[10.0, 0.1, 1.2]],
                           [[11.0, 0.5, 1.15]],
                           [[12.0, -1.0, 1.2]]])
        lower = np.array([[[2.5, 0.0, 0.0]],
                          [[2.5, 0.0, 0.0]],
                          [[2.5, 0.0, 0.0]],
                          [[2.5, 0.0, 0.0]]])
        upper = np.array([[[11.5, 9.0, 1.18]],
                          [[11.5, 9.0, 1.18]],
                          [[10.5, 10.0, 1.0]],
                          [[11.5, 9.0, 1.18]]])
        testing.assert_allclose(metrics.awd(lower, upper, y_data),
                                [[0.071181, 0.055556, 0.257839]], rtol=1e-4)

    @staticmethod
    def test_ace():
        """Test ace method."""
        y_data = np.array([[[1.0, 10.0, -1.0]],
                           [[10.0, 0.1, 1.2]],
                           [[11.0, 0.5, 1.15]],
                           [[12.0, -1.0, 1.2]]])
        lower = np.array([[[2.5, 0.0, 0.0]],
                          [[2.5, 0.0, 0.0]],
                          [[2.5, 0.0, 0.0]],
                          [[2.5, 0.0, 0.0]]])
        upper = np.array([[[11.5, 9.0, 1.18]],
                          [[11.5, 9.0, 1.18]],
                          [[10.5, 10.0, 1.0]],
                          [[11.5, 9.0, 1.18]]])
        testing.assert_allclose(metrics.ace(lower, upper, y_data, 0.5),
                                [[-0.25, 0.0, -0.5]])

    @staticmethod
    def test_alt_winkler():
        """Test alt_winkler method."""
        y_data = np.array([[[1.0, 10.0, -1.0]],
                           [[11.0, 0.0, 1.15]],
                           [[12.0, -1.0, 1.2]]])
        lower = np.array([[[0.5, 9.0, 0.0]],
                          [[0.6, 9.1, 0.1]],
                          [[0.7, 11.0, 1.3]]])
        upper = np.array([[[2.5, 11.0, 1.0]],
                          [[2.6, 11.1, 1.1]],
                          [[2.7, 12.0, 1.3]]])
        expected = [[(6.0 + 2.0 * (8.4 + 9.3) / 0.1) / 3.0,
                     (5.0 + 2.0 * (9.1 + 12.0) / 0.1) / 3.0,
                     (2.0 + 2.0 * (1.0 + 0.05 + 0.1) / 0.1) / 3.0]]
        alt_winkler = metrics.alt_winkler(lower, upper, y_data, 0.9)
        testing.assert_allclose(alt_winkler, expected)

    @staticmethod
    def test_winkler():
        """Test winkler method."""
        y_data = np.array([[[1.0, 10.0, -1.0]],
                           [[11.0, 0.0, 1.15]],
                           [[12.0, -1.0, 1.2]]])
        lower = np.array([[[0.5, 9.0, 0.0]],
                          [[0.6, 9.1, 0.1]],
                          [[0.7, 11.0, 1.3]]])
        upper = np.array([[[2.5, 11.0, 1.0]],
                          [[2.6, 11.1, 1.1]],
                          [[2.7, 12.0, 1.3]]])
        expected = [[(2.0 * 6.0 * 0.1 + 4.0 * (8.4 + 9.3)) / 3.0,
                     (2.0 * 5.0 * 0.1 + 4.0 * (9.1 + 12.0)) / 3.0,
                     (2.0 * 2.0 * 0.1 + 4.0 * (1.0 + 0.05 + 0.1)) / 3.0]]
        winkler = metrics.winkler(lower, upper, y_data, 0.9)
        testing.assert_allclose(winkler, expected)

    @staticmethod
    def test_cwc():
        """Test cwc method."""
        y_data = np.array([[[1.0, 10.0, -1.0]],
                           [[11.0, 0.0, 1.15]],
                           [[12.0, -1.0, 1.2]]])
        lower = np.array([[[0.5, 9.0, 0.0]],
                          [[0.6, 9.1, 0.1]],
                          [[0.7, 11.0, 1.3]]])
        upper = np.array([[[2.5, 11.0, 1.0]],
                          [[2.6, 11.1, 1.1]],
                          [[2.7, 12.0, 1.3]]])
        cwc = metrics.cwc(lower, upper, y_data, 0.9)
        testing.assert_allclose(
            cwc, [[2.0 / 11.0 * (1.0 + math.exp(-50 * (1 / 3.0 - 0.90))),
                   5.0 / 11.0 / 3.0 * (1.0 + math.exp(-50 * (1 / 3.0 - 0.90))),
                   2.0 / 2.2 / 3.0 * (1.0 + math.exp(-50 * -0.90))]])


if __name__ == '__main__':
    unittest.main()
