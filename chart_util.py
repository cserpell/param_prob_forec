# coding=utf-8
"""Methods to handle chart creation and storage."""
import matplotlib
matplotlib.use('Agg')  # Required when using via SSH with no display
from matplotlib import pyplot

DEFAULT_FIG_SIZE = (8, 8)


def start_figure():
    """Starts a new figure."""
    figure = pyplot.figure(figsize=DEFAULT_FIG_SIZE)
    figure.tight_layout()


def save(file_name, create_new=True):
    """Stores a figure and opens a new one."""
    pyplot.savefig(file_name)
    pyplot.close()
    if create_new:
        start_figure()
