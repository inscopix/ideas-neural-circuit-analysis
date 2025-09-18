"""useful mathematical functions and wrappers"""

import numpy as np
import scipy
import scipy.cluster.hierarchy as sch
from beartype import beartype
from beartype.typing import Optional
from ideas.types import (
    NumpyBooleanVector,
    NumpyFloat2DArray,
    NumpyFloatArray,
    NumpyFloatVector,
)
from scipy import stats


@beartype
def matrix_spearman(
    traces: NumpyFloatArray,
    *,
    only_when: Optional[NumpyBooleanVector] = None,
) -> tuple:
    """
    computes spearman rho and p for raw traces

    Returns:
        rho: vector of spearman rho
        p: vector of p-value from spearman test
    """

    if only_when is None:
        only_when = np.full_like(traces[:, 0], True, dtype=bool)

    num_neurons = traces.shape[1]
    time = np.arange(traces.shape[0])

    rho = np.full(num_neurons, np.nan)
    p = np.full(num_neurons, np.nan)
    for i in np.arange(num_neurons):
        temp = stats.spearmanr(
            time[only_when],
            traces[only_when, i],
            nan_policy="omit",
        )

        p[i] = temp.pvalue
        rho[i] = temp.correlation

    return rho, p


def matrix_ratio_max_min_dev(
    traces: NumpyFloat2DArray,
    *,
    only_when: Optional[np.array] = None,
) -> NumpyFloatVector:
    """Chris Donahue's  measure of a feature to help judge quality


    ### Returns:
        np.ndarray: ratio
    """

    if only_when is None:
        only_when = np.full_like(traces[:, 0], True, dtype=bool)

    # make a copy so we don't mutate
    data = np.copy(traces[only_when, :])

    data -= np.nanmedian(data, axis=0)
    data /= np.nanstd(data, axis=0)
    return np.abs(np.nanmax(data, axis=0) / np.nanmin(data, axis=0))


@beartype
def matrix_exponential_gof(
    traces: NumpyFloatArray,
    time: NumpyFloatVector,
    *,
    subsample: int = 100,
    nan_policy: Optional[str] = None,
) -> (NumpyFloatVector, float):
    """
    fits decaying exponentials to row of a matrix, returning
    goodness of fit metric

    ### Arguments

    - traces: matrix of cell traces
    - time: vector of time, as long as traces
    - subsample: integer to subsample data to speed up fits
    - nan_policy: passed to underlying scipy.optimize.curve_fit function:
        Defines how to handle when input contains nan. The following options are available (default is None):
            - raise: throws an error
            - omit: performs the calculations ignoring nan values
            - None: no special handling of NaNs is performed

    ### Returns

    - gof, a numpy array of goodness of fit values

    """
    if len(traces.shape) == 1:
        traces = traces[:, np.newaxis]

    num_neurons = traces.shape[1]

    def exponential(t, a, tau, offset):
        """simple decaying exponential"""
        return a * np.exp(tau * t) + offset

    bounds = ([0.0, -0.1, -np.inf], [np.inf, 0, np.inf])

    gof = np.zeros(num_neurons)

    for i in range(num_neurons):
        y = traces[:, i]

        try:
            # may get an error if scipy can't fit it
            params, _ = scipy.optimize.curve_fit(
                exponential,
                time[::subsample],
                y[::subsample],
                bounds=bounds,
                nan_policy=nan_policy,
            )
        except RuntimeError:
            continue

        fit_y = exponential(time, *params)

        if len(np.unique(y)) > 1:
            try:
                gof[i] = scipy.stats.linregress(y, fit_y).rvalue
            except Exception:
                # don't know what's happening here
                # but it's safe to set gof to 0
                pass

    gof = gof**2

    if num_neurons == 1:
        return gof[0]
    else:
        return gof


@beartype
def matrix_skew(
    traces: NumpyFloat2DArray,
    *,
    only_when: Optional[NumpyBooleanVector] = None,
) -> NumpyFloatVector:
    """
    helper function that computes the skew of a matrix.
    Useful to compute the skew of a traces matrix, especially
    only at certain times.

    """

    if only_when is None:
        only_when = np.full_like(traces[:, 0], True, dtype=bool)

    return stats.skew(traces[only_when, :], axis=0, nan_policy="omit")


@beartype
def correlation_matrix(
    x: NumpyFloatArray,
    *,
    rearrange_rows: bool = False,
    fill_diagonal: float = np.nan,
) -> (NumpyFloat2DArray, float):
    """
    computes a correlation matrix of some input matrix,
    operating along the first dimension.

    ### Arguments:

    - x: some matrix, each row is a variable
    - rearrange_rows: reorder output cluster similar rows?
    - fill_diagonal : fill diagonal of correlation matrix with this

    ### Returns

    rearranged correlation matrix
    """

    if len(x.shape) == 1:
        # only one thing, so there is nothing to compute
        return fill_diagonal

    # some timepoints of x maybe nan, let's remove those
    x = x[~np.isnan(x.sum(axis=1)), :]

    corr = np.corrcoef(x, rowvar=False)

    if rearrange_rows:
        corr = rearrange_correlation_matrix(corr)
    np.fill_diagonal(corr, fill_diagonal)

    return corr


@beartype
def rearrange_correlation_matrix(
    corr_matrix: NumpyFloat2DArray,
    threshold: float = 0.5,
    inplace: bool = False,
) -> NumpyFloat2DArray:
    """
    Rearranges a correlation matrix, so that groups of highly
    correlated variables are next to each other

    ### Arguments

    - corr_matrix : a NxN correlation matrix
    - threshold (float): Threshold value between 0 and 1 (inclusive)
      representing a percentage for assigning clusters.

    ### Returns

    numpy.ndarray

    a NxN correlation matrix with the columns and rows rearranged
    """

    idx, _ = _get_new_matrix_index(corr_matrix, threshold=threshold)

    if not inplace:
        corr_matrix = corr_matrix.copy()

    return corr_matrix[idx, :][:, idx]


def _get_new_matrix_index(
    corr_matrix: np.array, threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get new row/column index for correlation matrix, so that
    groups of highly correlated variables are next to each other

    ### Arguments

    - corr_matrix :  a NxN correlation matrix
    - threshold (float): Threshold value between 0 and 1
    (inclusive) representing a fraction for
    assigning clusters.

    ### Returns

    idx -  Nx1 vector with new index order
    clusters - Nx1 vector with new cluster assignment
    """

    pairwise_distances = sch.distance.pdist(corr_matrix)
    linkage = sch.linkage(pairwise_distances, method="complete")
    cluster_distance_threshold = pairwise_distances.max() * threshold
    clusters = sch.fcluster(
        linkage,
        cluster_distance_threshold,
        criterion="distance",
    )
    idx = np.argsort(clusters)

    return idx, clusters
