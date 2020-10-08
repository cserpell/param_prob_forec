# coding=utf-8
"""Module with distributions sampling layers and helper methods."""
import enum

from tensorflow import math
from tensorflow_probability import bijectors
from tensorflow_probability import distributions
from tensorflow_probability import layers

EPSILON = 0.5


class Error(Exception):
    """Error raised for not implemented distribution"""


class Distribution(enum.Enum):
    """Available distributions."""
    NORMAL = 1
    WEIBULL = 2
    LOG_NORMAL = 3
    GAMMA = 4
    F = 5
    CHI_SQUARED = 6
    MAF = 7
    BETA = 8


def number_parameters(distribution):
    """Gets number of parameters required for distribution."""
    if distribution == Distribution.CHI_SQUARED:
        return 1
    return 2


def normal_distribution(param):
    """Normal distribution parameterization from input."""
    return distributions.Normal(param[:, 0], math.sqrt(math.exp(param[:, 1])))


def log_normal_distribution(param):
    """Log Normal distribution parameterization from input."""
    return distributions.LogNormal(
        param[:, 0], math.sqrt(math.exp(param[:, 1])))


def gamma_distribution(param):
    """Gamma distribution parameterization from input."""
    return distributions.Gamma(math.exp(param[:, 0]), math.exp(param[:, 1]))


def beta_distribution(param):
    """Beta distribution parameterization from input."""
    return distributions.Beta(math.exp(param[:, 0]), math.exp(param[:, 1]))


def weibull_distribution(param):
    """Weibull distribution parameterization from input."""
    return distributions.TransformedDistribution(
        distribution=distributions.Uniform(low=(0.0 * param[:, 0]), high=1.0),
        bijector=bijectors.Invert(bijectors.WeibullCDF(
            scale=math.exp(param[:, 1]), concentration=math.exp(param[:, 0]))))


def maf_distribution(param):
    """Masked Autoregressive Flow from input."""
    # The flow starts from a known distribution
    base_distribution = distributions.MultivariateNormalDiag(
        loc=param[:, 1], scale_diag=math.sqrt(math.exp(param[:, 1])))
    # Iterate over number of flow transformations
    local_bijectors = []
    # num_bijectors = (param.shape[2] * param.shape[3]).value
    # A permutation is performed after each flow step, in order to learn
    # different conditional distributions
    permutation_order = list(range(param.shape[-1]))[1:] + [0]
    for _ in range(3):
        # MaskedAutoregressiveFlow makes one flow transformation, applying
        # given autoregressive transformation below (MADE in this case), which
        # uses hidden layers of given sizes.
        made = bijectors.masked_autoregressive_default_template([20, 20])
        local_bijectors.append(bijectors.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn=made))
        # Permutation to be able to learn all conditional distributions
        local_bijectors.append(bijectors.Permute(permutation_order))
    # Discard the last Permute layer
    flow_bijector = bijectors.Chain(list(reversed(local_bijectors[:-1])))
    return distributions.TransformedDistribution(
        distribution=base_distribution, bijector=flow_bijector)


def get_sampler(distribution):
    """Gets correct sampler layer."""
    if distribution == Distribution.NORMAL:
        distribution_fn = normal_distribution
    elif distribution == Distribution.WEIBULL:
        distribution_fn = weibull_distribution
    elif distribution == Distribution.LOG_NORMAL:
        distribution_fn = log_normal_distribution
    elif distribution == Distribution.GAMMA:
        distribution_fn = gamma_distribution
    elif distribution == Distribution.BETA:
        distribution_fn = beta_distribution
    elif distribution == Distribution.MAF:
        distribution_fn = maf_distribution
    else:
        raise Error('Not implemented {}'.format(distribution))
    return layers.DistributionLambda(make_distribution_fn=distribution_fn)
