import numpy as np
from scipy.linalg import toeplitz
from lifelines.statistics import logrank_test
from warnings import warn

def features_normal_cov_toeplitz(n_samples, n_features: int = 30,
                                 cov_corr: float = 0.5, dtype="float64"):
    """Normal features generator with toeplitz covariance

    An example of features obtained as samples of a centered Gaussian
    vector with a toeplitz covariance matrix

    Parameters
    ----------
    n_samples : `int`, default=200
        Number of samples

    n_features : `int`, default=30
        Number of features

    cov_corr : `float`, default=0.5
        correlation coefficient of the Toeplitz correlation matrix

    dtype : `{'float64', 'float32'}`, default='float64'
        Type of the arrays used.

    Returns
    -------
    output : `numpy.ndarray`, shape=(n_samples, n_features)
        n_samples realization of a Gaussian vector with the described
        covariance

    """
    cov = toeplitz(cov_corr ** np.arange(0, n_features))
    features = np.random.multivariate_normal(
        np.zeros(n_features), cov, size=n_samples)
    if dtype != "float64":
        return features.astype(dtype)
    return features

def weights_sparse_exp(n_weigths: int = 100, nnz: int = 10, scale: float = 10.,
                       dtype="float64") -> np.ndarray:
    """Sparse and exponential model weights generator

    Instance of weights for a model, given by a vector with
    exponentially decaying components: the j-th entry is given by

    .. math: (-1)^j \exp(-j / scale)

    for 0 <= j <= nnz - 1. For j >= nnz, the entry is zero.

    Parameters
    ----------
    n_weigths : `int`, default=100
        Number of weights

    nnz : `int`, default=10
        Number of non-zero weights

    scale : `float`, default=10.
        The scaling of the exponential decay

    dtype : `{'float64', 'float32'}`, default='float64'
        Type of the arrays used.

    Returns
    -------
    output : np.ndarray, shape=(n_weigths,)
        The weights vector
    """
    if nnz >= n_weigths:
        warn(("nnz must be smaller than n_weights "
              "using nnz=n_weigths instead"))
        nnz = n_weigths
    idx = np.arange(nnz)
    out = np.zeros(n_weigths, dtype=dtype)
    out[:nnz] = np.exp(-idx / scale)
    out[:nnz:2] *= -1
    return out


def compute_logrank_test(control, treat):
    """
    Perform a two-sample log-rank test comparing the survival distributions
    of control and treatment groups.

    Args:
        control (DataFrame): Subset of the dataset where treatment == 0.
        treat (DataFrame): Subset of the dataset where treatment == 1.

    Returns:
        float: Negative logarithm of the p-value from the log-rank test.
    """
    surv_time_control = control['time'].values.astype(bool)
    surv_event_control = control['censor'].values
    surv_time_treat = treat['time'].values.astype(bool)
    surv_event_treat = treat['censor'].values

    result = logrank_test(
        surv_time_control, surv_time_treat,
        event_observed_A=surv_event_control,
        event_observed_B=surv_event_treat
    )
    return -np.log(result.p_value)