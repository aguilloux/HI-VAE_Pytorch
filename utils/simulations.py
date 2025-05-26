import numpy as np
import pandas as pd
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

    .. math: (-1)^j exp(-j / scale)

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
    surv_time_control = control['time'].values
    surv_event_control = control['censor'].values.astype(bool)
    surv_time_treat = treat['time'].values
    surv_event_treat = treat['censor'].values.astype(bool)

    result = logrank_test(
        surv_time_control, surv_time_treat,
        event_observed_A=surv_event_control,
        event_observed_B=surv_event_treat
    )
    return -np.log(result.p_value)




def simulation(beta_features, treatment_effect , n_samples , independent = True, surv_type = 'surv_piecewise', n_features_bytype = 4, 
                n_features_multiplier = 3, nnz = 3 , p_treated = 0.5,a_T=2,
                a_C = 2., lamb_C = 6., lamb_C_indpt = 2.5, data_types_create = True):
    n_features = n_features_multiplier * n_features_bytype
    beta = np.insert(beta_features, 0, treatment_effect)
    X = features_normal_cov_toeplitz(n_samples,n_features)
    X[:,(n_features_bytype ) : (2*n_features_bytype )] = np.abs(X[:,(n_features_bytype ) : (2*n_features_bytype )])
    X[:,(2*n_features_bytype ) : (3*n_features_bytype )] = 1 * (X[:,(2*n_features_bytype ) : (3*n_features_bytype )]>= 0)
    treatment = np.random.binomial(1, p_treated, size=(n_samples,1))
    design = np.hstack((treatment,X))
    marker = np.dot(design,beta)
    U = np.random.uniform(size = n_samples)
    V = np.random.uniform(size = n_samples)
    T = (- np.log(1-U) / np.exp(marker))**(1/a_T)
    if independent:
        C = lamb_C * (- np.log(1-V))**(1/a_C)
    else:
        C = lamb_C_indpt * (- np.log(1-V) / np.exp(marker))**(1/a_C)
    data = pd.DataFrame(X)
    data['treatment'] = treatment
    data['time'] = np.min([T,C],axis=0)
    data['censor'] = np.argmin([C,T],axis=0)
    control = data[data['treatment'] == 0]
    treated = data[data['treatment'] == 1]
    if data_types_create == True:
        names = []
        for x in range(1, n_features_bytype  * n_features_multiplier + 1):
            names.append("feat{0}".format(x))
        names.append("survcens")
        types = np.concatenate([np.repeat("real",n_features_bytype),np.repeat("pos",n_features_bytype),np.repeat("cat",n_features_bytype)]).tolist()
        types.append(surv_type)
        dims = np.repeat(1,n_features_bytype * n_features_multiplier).tolist()
        dims.append(2)
        nclasses = np.concatenate([np.repeat("",n_features_bytype),np.repeat("",n_features_bytype),np.repeat("2",n_features_bytype)]).tolist()
        nclasses.append("")
        data_types = pd.DataFrame({'name' : names , 'type' : types , 'dim' : dims, 'nclass' : nclasses})
        return(control,treated,data_types)
    else :
        return(control,treated)


def cpower(mc , mi , loghaz,alpha):
    """
    mc : number of survivors in control arm
    mi : number of survivors in treated arm
    loghaz : log of hazard ratios / treatment coefficient
    alpha : level of test
    """
    ## Find its variance
    v = 1/mc + 1/mi

    ## Get same as /sasmacro/samsizc.sas if use 4/(mc+mi)

    sd = np.sqrt(v)

    z =  -norm.ppf(alpha/2)

    Power = 1 - (norm.cdf(z - np.abs(loghaz)/sd) - norm.cdf(-z - np.abs(loghaz)/sd))
    return(Power)


    
    