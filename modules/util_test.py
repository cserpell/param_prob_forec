# coding=utf-8
"""Tests for util module."""
import unittest

import numpy as np
from numpy import testing

from modules import util


class UtilTest(unittest.TestCase):
    """Unit test for util module."""

    @staticmethod
    def test_mean_squared_loss_sigma():
        """Test mean_squared_loss_with_sigma method."""
        y_data = np.array([[[1.0, 2.0, -1.0],
                            [-1.386, 0.0, 0.811]]])  # sigma = 0.5, 1.0, 1.5
        y_true = np.array([[[1.2, 2.5, -1.2], [0.0, 0.0, 0.0]]])
        out = util.mean_squared_loss_with_sigma(y_true, y_data)
        testing.assert_allclose(
            out.numpy(), (-1.226047 + 0.25 + 0.828777) / 3.0, rtol=0.00001)


if __name__ == '__main__':
    unittest.main()
