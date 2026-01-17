import numpy as np
import scipy.stats as stats
from sklearn.metrics.pairwise import rbf_kernel

def ks_test(reference, current):
    # reference/current are 1D arrays (numeric)
    stat, p = stats.ks_2samp(reference, current)
    return stat, p

def psi(expected, actual, buckets=10):
    # Population Stability Index (PSI)
    def _get_bins(arr, buckets):
        percentiles = np.linspace(0,100,buckets+1)
        return np.percentile(arr, percentiles)
    bins = _get_bins(expected, buckets)
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)
    expected_pct = expected_counts / np.maximum(1, expected_counts.sum())
    actual_pct = actual_counts / np.maximum(1, actual_counts.sum())
    # avoid zeros
    actual_pct = np.where(actual_pct==0, 0.0001, actual_pct)
    expected_pct = np.where(expected_pct==0, 0.0001, expected_pct)
    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_val)

def mmd_rbf(X, Y, gamma=None):
    # Maximum Mean Discrepancy with RBF kernel
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)
    m = X.shape[0]
    n = Y.shape[0]
    mmd = Kxx.sum()/(m*m) + Kyy.sum()/(n*n) - 2*Kxy.sum()/(m*n)
    return float(mmd)