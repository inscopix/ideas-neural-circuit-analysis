import logging
import warnings
from typing import List

import numpy as np
import pingouin as pg
from scipy import stats

from utils.statistical_validation import _suppress_pingouin_warnings

logger = logging.getLogger()


def safe_nanmean(data, axis=0):
    """Compute nanmean without warnings for empty/NaN slices.

    This function safely computes the mean of an array while ignoring NaN values,
    handling edge cases like empty arrays or all-NaN slices without raising warnings.

    :param data: Input array-like data
    :param axis: Axis along which to compute the mean (default: 0)
    :return: Array of means with float32 dtype, NaN where no valid data exists
    """
    data = np.asarray(data)
    if data.size == 0:
        out_shape = tuple(dim for i, dim in enumerate(data.shape) if i != axis)
        return np.full(out_shape, np.nan, dtype="float32")

    finite_count = np.sum(np.isfinite(data), axis=axis)
    out_shape = tuple(dim for i, dim in enumerate(data.shape) if i != axis)
    total = np.nansum(data, axis=axis, dtype="float32")
    mean = np.divide(
        total,
        finite_count,
        out=np.full(out_shape, np.nan, dtype="float32"),
        where=finite_count > 0,
    )
    return mean.astype("float32")


def safe_sem(data, axis=0):
    """Compute standard error of the mean without small-sample warnings.

    This function safely computes SEM while handling edge cases like empty arrays,
    all-NaN slices, or samples with fewer than 2 finite values without raising warnings.

    :param data: Input array-like data
    :param axis: Axis along which to compute SEM (default: 0)
    :return: Array of SEM values with float32 dtype, NaN where insufficient data exists
    """
    data = np.asarray(data)
    if data.size == 0:
        out_shape = tuple(dim for i, dim in enumerate(data.shape) if i != axis)
        return np.full(out_shape, np.nan, dtype="float32")

    finite_count = np.sum(np.isfinite(data), axis=axis)
    out_shape = tuple(dim for i, dim in enumerate(data.shape) if i != axis)
    mean = safe_nanmean(data, axis=axis)
    diff = data - np.expand_dims(mean, axis=axis)
    diff = np.where(np.isfinite(diff), diff, 0.0)
    sum_sq = np.sum(diff**2, axis=axis, dtype="float32")
    var = np.divide(
        sum_sq,
        finite_count - 1,
        out=np.full(out_shape, np.nan, dtype="float32"),
        where=finite_count > 1,
    )
    sem = np.sqrt(
        np.divide(
            var,
            finite_count,
            out=np.full(out_shape, np.nan, dtype="float32"),
            where=finite_count > 0,
        )
    )
    return sem.astype("float32")


def ttest(
    group1_data: List[float],
    group2_data: List[float],
    comparison_type: str,
    data_pairing: str,
):
    """Perform a statistical comparison using a standard t-test compare data from groups 1 and 2.

    :param group1_data: list of data points for the first group
    :param group2_data: list of data points for the second group
    :param comparison_type: type of statistical test to perform
                            possible values: "two_tailed", "one_tailed_less", "one_tailed_greater"
    :param data_pairing: indicates whether observations should be paired for statistical comparison
                         possible values: "paired", "unpaired"
    """
    # define alternative hypothesis
    if comparison_type == "two_tailed":
        alternative = "two-sided"
    elif comparison_type == "one_tailed_less":
        alternative = "less"
    else:
        # comparison_type == "one_tailed_greater"
        alternative = "greater"

    # define population variances assumption
    if data_pairing == "paired":
        equal_population_var = False
    else:
        # data_pairing == "unpaired"
        equal_population_var = True

    # compare the means of the two groups using a t-test
    t_statistic, p_value = stats.ttest_ind(
        group1_data,
        group2_data,
        equal_var=equal_population_var,
        alternative=alternative,
    )

    return t_statistic, p_value


def is_normal(
    data,
    metric,
    ind_variable_name,
    significance_threshold=0.05,
    min_sample_size_for_normality=8,
):
    """Determine whether the data follows a normal distribution."""
    try:
        if (
            len(data) >= min_sample_size_for_normality
        ):  # if there are less than 'min_sample_size_for_normality' samples, use parametric tests
            # Suppress warnings during normality test
            # This includes warnings about:
            # - Small sample sizes (already handled by MIN_SAMPLE_SIZE_FOR_NORMALITY check)
            # - Non-normal distributions (expected and handled by parametric flag)
            # - Numerical precision issues
            # - Statistical assumption violations
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                normal = pg.normality(
                    data=data,
                    dv=metric,
                    group=ind_variable_name,
                    alpha=significance_threshold,
                )
                parametric = normal["normal"].all()
        else:
            logger.info(
                f"Fewer than {min_sample_size_for_normality} observations found. "
                f"Non-parametric tests will be used."
            )
            parametric = False
    except Exception as e:
        logger.warning(
            f"An error occurred during the normality test: {str(e)}. Defaulting to non-parametric tests."
        )
        parametric = False
    return parametric


@_suppress_pingouin_warnings
def statistically_compare_two_groups(
    x, y, comparison_type, data_pairing, parametric, epoch_name
):
    """Run statistical comparison between two groups of data.

    :param x: data from group 1
    :param y: data from group 2
    :param alternative: alternative hypothesis
    :param data_pairing: indicates whether observations should be paired for statistical comparison
    :param parametric: indicates whether to perform a parametric or non-parametric test
    """
    # convert comparison type to format expected by the pingouin library
    if comparison_type == "two_tailed":
        alternative = "two-sided"
    elif comparison_type == "one_tailed_less":
        alternative = "less"
    else:
        # comparison_type == "one_tailed_greater"
        alternative = "greater"

    # convert data pairing string to a boolean
    # representing whether observations are paired
    paired = True if data_pairing == "paired" else False

    if parametric:
        # parametric test for paired/unpaired data
        logger.info(f"Performing t-test (parametric, {data_pairing})")
        df = pg.ttest(
            x=x,
            y=y,
            paired=paired,
            alternative=alternative,
        )
        test = "t-test"
    else:
        if paired:
            # non-parametric test for paired data
            logger.info("Performing Wilcoxon signed-rank test (non-parametric, paired)")
            df = pg.wilcoxon(x=x, y=y, alternative=alternative, nan_policy="omit")
            test = "Wilcoxon signed-rank test"
        else:
            # non-parametric test for unpaired data
            logger.info("Performing Mann–Whitney U test (non-parametric, unpaired)")
            df = pg.mwu(x=x, y=y, alternative=alternative, nan_policy="omit")
            test = "Mann–Whitney U test"

    cols = list(df.columns)

    # add data being compared and test perform to output data
    df["Comparison"] = "true_mean_post-pre"
    df["Epoch"] = epoch_name
    df["Test"] = test
    df["Paired"] = paired
    df["Parametric"] = parametric

    # reorder output dataframe columns
    cols = ["Comparison", "Epoch", "Test", "Paired", "Parametric"] + cols
    df = df[cols]

    return df


@_suppress_pingouin_warnings
def perform_paired_pairwise_comparisons(
    df,
    comparison_type,
    dependent_var_col,
    comparison_name,
    within_col,
    subject_col,
):
    """Perform paired pairwise comparisons to compare epochs."""
    # define alternative hypothesis
    if comparison_type == "two_tailed":
        alternative = "two-sided"
    elif comparison_type == "one_tailed_less":
        alternative = "less"
    else:
        # comparison_type == "one_tailed_greater"
        alternative = "greater"

    # perform pairwise tests between the epochs
    pairwise = pg.pairwise_tests(
        data=df,
        dv=dependent_var_col,
        within=within_col,
        subject=subject_col,
        alternative=alternative,
    )

    # add comparison column to specify what is being compared across epochs
    pairwise["Comparison"] = comparison_name

    # reorder columns
    columns = pairwise.columns
    columns = [columns[-1]] + list(columns[:-1])
    pairwise = pairwise[columns]

    return pairwise
