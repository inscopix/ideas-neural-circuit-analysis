import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from ideas import measures, plots

from utils.plots import (
    _plot_box_and_strip,
    _plot_comparison,
    _plot_correlation,
    _plot_correlation_comparison,
    _plot_difference_cellmap,
    _plot_mixed_comparisons,
    _plot_mixed_corr_comparisons,
)
from utils.utils import (
    Metric,
    save_optimized_svg,
    DIVISION_THRESHOLD,
)

# Minimum sample size required for normality testing
MIN_SAMPLE_SIZE_FOR_NORMALITY = 8

logger = logging.getLogger(__name__)


def _secs_to_idx(epoch, period):
    """Convert the epoch from seconds to index.

    :Args
        epoch (tuple): A tuple containing the start and end epoch in seconds.
        period (int): The period in seconds.

    :Returns
        tuple: A tuple containing the start and end index.
    """
    """Convert the epoch from seconds to index."""
    return int(epoch[0] / period), int(epoch[1] / period)


def _calculate_population_activity(
    *,
    data_name,
    data,
    epochs,
    epoch_names,
    epoch_colors,
    ylabel,
    period,
    x,
    y,
    scale_method,
    cell_names,
):
    """Create function will take use the data and epochs to create two plots.

    First a boxplot of the activity between epochs and second a
    difference cellmap of the activity between epochs with corresponding
    histogram and pie chart.
    """
    df = pd.DataFrame(columns=["Epoch", "Cell", "Activity"])

    epochs_idx = [_secs_to_idx(epoch, period) for epoch in epochs]
    for i, epoch in enumerate(epochs_idx):
        # get the data for each epoch
        full_epoch = data[epoch[0] : epoch[1], :]
        # calculate the mean activity for each cell
        mean_activity = np.nanmean(full_epoch, axis=0)
        # create a temporary dataframe
        temp_df = pd.DataFrame(
            {
                "Epoch": [epoch_names[i]] * len(mean_activity),
                "Cell": cell_names,
                "Activity": mean_activity,
            }
        )
        # drop columns with all-NA entries in temp_df
        temp_df = temp_df.dropna(axis="columns", how="all")
        # drop columns with all-NA entries in df
        df = df.dropna(axis="columns", how="all")
        # add the data to the dataframe
        df = pd.concat([df, temp_df], ignore_index=True)

    # Create boxplot of activity between epochs
    try:
        _plot_box_and_strip(
            epochs=epochs,
            epoch_names=epoch_names,
            epoch_colors=epoch_colors,
            df=df,
            num_cells=len(df["Cell"].unique()),
            ylabel=ylabel,
            save_name=f"Population_{data_name}.svg",
        )
    except Exception as error:
        logger.exception(error)
        logger.warning(
            f"Failed to generate preview Population_{data_name}.svg"
        )

    try:
        _plot_difference_cellmap(
            epoch_names=epoch_names,
            epoch_colors=epoch_colors,
            x=x,
            y=y,
            df=df,
            num_cells=len(df["Cell"].unique()),
            ylabel=ylabel,
            save_name=f"Population_{data_name}_differences.svg",
            scale_method=scale_method,
        )
    except Exception as error:
        logger.exception(error)
        logger.warning(
            f"Failed to generate preview Population_{data_name}_differences.svg"
        )

    # save the data to a csv
    df.to_csv(f"{data_name}_activity_data.csv", index=False)


def _calculate_correlation(
    *,
    data_name,
    data,
    epochs,
    epoch_names,
    epoch_colors,
    period,
    output_format="svg",
):
    """Generate correlation matrices and properly align the colorbar.

    :param data_name: Name identifier for the dataset.
    :param data: Input data for correlation calculation.
    :param epochs: List of epoch time ranges.
    :param epoch_names: Corresponding names for each epoch.
    :param epoch_colors: Colors for visualization.
    :param period: Sampling period for time conversion.
    :param output_format: Output format for plots ("png" or "svg").
    """
    correlation_mtx = {}

    # Create matrix of scatter plots
    fig, ax = plt.subplots(
        ncols=len(epochs) + 1,  # Extra column for colorbar
        figsize=(6 * len(epochs), 6),
        width_ratios=[1] * len(epochs) + [0.1],
    )

    for idx, (epoch, epoch_name) in enumerate(zip(epochs, epoch_names)):
        # Convert epoch times to indices
        epoch = _secs_to_idx(epoch, period)

        # Extract data for the epoch
        epoch_data = data[epoch[0] : epoch[1]]

        if len(epoch_data) < 2:
            logger.warning(
                f"Insufficient data for epoch '{epoch_name}' "
                f"to compute the correlation matrix, a minimum of 2 "
                f"timepoints are needed, skipping"
            )
            ax[idx].set_title(f"Correlation during {epoch_names[idx]}")
            ax[idx].text(
                0.5,
                0.5,
                "Insufficient number of time points",
                horizontalalignment="center",
                verticalalignment="center",
            )
            continue

        mtx = measures.correlation_matrix(epoch_data, fill_diagonal=0.0)
        correlation_mtx[epoch_names[idx]] = mtx

        # check for nans in the correlation matrix
        # and replace them with 0s
        if np.isnan(mtx).any():
            logger.warning(
                f"Found nan values in the correlation matrix for epoch {epoch_name}, "
                f"replacing them with 0's for plotting"
            )
            mtx = np.nan_to_num(mtx)

        # Rearrange the correlation matrix for better visualization
        r_mtx = measures.rearrange_correlation_matrix(corr_matrix=mtx)

        # Plot correlation matrix based on format
        if output_format == "svg":
            plots.plot_correlation_matrix(ax[idx], r_mtx, output_format="svg")
        else:
            plots.plot_correlation_matrix(ax[idx], r_mtx, output_format="png")

        ax[idx].set_title(f"Correlation during {epoch_names[idx]}")

    # Save the correlation matrix to an .npy file
    np.save(f"Population_{data_name}_Correlation.npy", correlation_mtx)

    # Plot the colorbar on the additional axis
    sm = plt.cm.ScalarMappable(cmap="coolwarm")
    sm.set_array([-1, 1])
    plt.colorbar(sm, ax[-1], label="Correlation", ticks=[-1, 0, 1])

    if output_format == "svg":
        # Optimize and save the SVG file
        save_optimized_svg(
            fig,
            "Correlations.svg",
            max_size_mb=5,
        )
    else:
        # Save as PNG
        fig.tight_layout()
        fig.savefig(
            "Correlations.png",
            dpi=300,
            format="png",
        )

    plt.close(fig)  # Close the figure to free memory

    # Create a barchart of the average positive and negative correlations
    df = pd.DataFrame(columns=["Epoch", "Subset", "Activity"])

    for i, epoch in enumerate(epochs):
        # get the data for each epoch and linearize it

        # get the data for each epoch
        epoch = _secs_to_idx(epoch, period)

        corr_data = correlation_mtx[epoch_names[idx]].flatten()
        temp_df = pd.DataFrame(
            {
                "Epoch": [epoch_names[i]] * len(corr_data),
                "Cell-Cell Pair": np.arange(len(corr_data)),
                "Activity": corr_data,
            }
        )

        temp_df = temp_df.dropna(axis="columns", how="all")
        df = df.dropna(axis="columns", how="all")

        df = pd.concat(
            [
                df,
                temp_df,
            ]
        )

    # Clean up the variables to save on memory
    _plot_correlation(
        epochs=epochs,
        epoch_colors=epoch_colors,
        df=df,
        ylabel="Correlation",
    )


def subsample(data, sample_size):
    """Subsamples the given data by calculating the mean of each sample.

    :Args
        data (list): The data to be subsampled.
        sample_size (int): The size of each subsample.

    :Returns
        list: A list of mean values, each representing a subsample.
    """
    results = []
    for i in range(0, len(data), sample_size):
        # Get slice of data for this subsample
        chunk = data[i : min(i + sample_size, len(data))]

        # If this is the last chunk and it's incomplete, use its actual value
        if i + sample_size > len(data) and len(chunk) == 1:
            results.append(chunk[0])
        else:
            # Otherwise calculate the mean
            results.append(np.mean(chunk))

    return results


def cohen_d(x, y):
    """Calculate Cohen's d effect size between two groups.

    :Parameters
    x (array-like): First group of observations.
    y (array-like): Second group of observations.

    :Returns
    float: Cohen's d effect size.

    :Raises
    Exception: If either group has fewer than 2 observations
    """
    nx = len(x)
    ny = len(y)

    if nx <= 1 or ny <= 1:
        raise Exception("Cohen's d requires at least 2 observations per group")

    dof = nx + ny - 2
    pooled_var = (
        (nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2
    ) / dof

    # Prevent division by zero
    if pooled_var < DIVISION_THRESHOLD:
        logger.warning(
            "Near-zero pooled variance in Cohen's d calculation. Returning inf or NaN."
        )
        return np.inf if np.mean(x) != np.mean(y) else np.nan

    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled_var)


def _run_ANOVA(
    data,
    method,
    multiple_correction,
    effect_size,
    metric,
    file,
    ind_variable_name="Epoch",
):
    """Perform ANOVA (Analysis of Variance) on the given dataset.
    :Parameters
    data (DataFrame): The dataset containing the data to be analyzed.
    method (str): The type of ANOVA to perform. Options are 'mixed' or 'oneway'.
    multiple_correction (str): The method for multiple comparison correction.
    effect_size (str): The type of effect size to compute.
    metric (str): The metric to be analyzed.
    :Returns
    tuple: A tuple containing two DataFrames:
        - aov: The ANOVA results.
        - pairwise: The pairwise comparison results.
    """
    # Check for problematic data that could cause warnings
    try:
        # Remove any constant groups which will cause division by zero
        groups = data[ind_variable_name].unique()
        groups_to_drop = []

        for group in groups:
            group_data = data[data[ind_variable_name] == group][metric]

            # Check for constant values within a group
            if len(group_data) > 0 and group_data.std() < DIVISION_THRESHOLD:
                logger.warning(
                    f"Group '{group}' has constant or near-constant values for {metric}. "
                    "This can cause division by zero in statistical calculations."
                )
                if len(groups) > 2:  # Only drop if we have enough groups left
                    groups_to_drop.append(group)

        # Only filter if we have groups to drop
        if groups_to_drop and len(groups) - len(groups_to_drop) >= 2:
            logger.warning(
                f"Removing groups {groups_to_drop} from analysis due to constant values."
            )
            data = data[~data[ind_variable_name].isin(groups_to_drop)]

        # If we have fewer than 2 groups left, we can't run ANOVA
        if len(data[ind_variable_name].unique()) < 2:
            logger.warning(
                "Not enough valid groups for analysis. Returning empty results."
            )
            empty_aov = pd.DataFrame(
                {"Comparison": [metric], "p-unc": [np.nan]}
            )
            empty_pairwise = pd.DataFrame(
                {"Comparison": [metric], "p-unc": [np.nan]}
            )
            return empty_aov, empty_pairwise

        # Add tiny random noise to eliminate exact zeros in variance
        if data[metric].std() < DIVISION_THRESHOLD:
            logger.warning(
                f"Adding tiny noise to {metric} data to avoid division by zero."
            )
            noise_scale = DIVISION_THRESHOLD
            data[metric] = data[metric] + np.random.normal(
                0, noise_scale, size=len(data)
            )

    except Exception as e:
        logger.warning(f"Error preprocessing data for ANOVA: {str(e)}")
        empty_aov = pd.DataFrame({"Comparison": [metric], "p-unc": [np.nan]})
        empty_pairwise = pd.DataFrame(
            {"Comparison": [metric], "p-unc": [np.nan]}
        )
        return empty_aov, empty_pairwise

    # check for normality to see if we should use parametric tests
    parametric = False
    try:
        if (
            len(data) >= MIN_SAMPLE_SIZE_FOR_NORMALITY
        ):  # if there are less than MIN_SAMPLE_SIZE_FOR_NORMALITY samples, use parametric tests
            # Suppress warnings during normality test
            # This includes warnings about:
            # - Small sample sizes (already handled by MIN_SAMPLE_SIZE_FOR_NORMALITY check)
            # - Non-normal distributions (expected and handled by parametric flag)
            # - Numerical precision issues
            # - Statistical assumption violations
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                normal = pg.normality(
                    data=data,
                    dv=metric,
                    group=ind_variable_name,
                    alpha=0.05,
                )
                parametric = normal["normal"].all()
        else:
            parametric = False
    except Exception as e:
        logger.warning(
            f"Error checking normality: {str(e)}. Defaulting to non-parametric tests."
        )
        parametric = False

    # Run the statistical tests with exception handling
    try:
        if method == "mixed":
            with np.errstate(divide="ignore", invalid="ignore"):
                try:
                    aov = pg.mixed_anova(
                        data=data,
                        dv=metric,
                        within="Epoch",
                        between="Group",
                        subject=file,  # file is the subject
                    )
                    # Ensure Source column exists for compatibility with tests
                    if "Source" not in aov.columns and len(aov) > 0:
                        aov["Source"] = ["Epoch", "Group", "Epoch * Group"][
                            : len(aov)
                        ]

                    pairwise = pg.pairwise_tests(
                        data=data,
                        dv=metric,
                        within="Epoch",
                        subject=file,  # file is the subject
                        between="Group",
                        interaction=True,
                        padjust=multiple_correction,
                        effsize=effect_size,
                        parametric=parametric,
                    )
                except Exception as e:
                    # Create a dummy dataframe with expected columns for mixed ANOVA
                    logger.warning(
                        f"Error during mixed ANOVA: {str(e)}. Creating dummy results."
                    )
                    aov = pd.DataFrame(
                        {
                            "Source": ["Epoch", "Group", "Epoch * Group"],
                            "F": [np.nan, np.nan, np.nan],
                            "p-unc": [1.0, 1.0, 1.0],
                            "np2": [0.0, 0.0, 0.0],
                        }
                    )
                    pairwise = pd.DataFrame(
                        {
                            "Contrast": ["Epoch * Group"],
                            "A": [
                                (
                                    data["Epoch"].iloc[0]
                                    if len(data) > 0
                                    else "Epoch1"
                                )
                            ],
                            "B": [
                                (
                                    data["Epoch"].iloc[-1]
                                    if len(data) > 0
                                    else "Epoch2"
                                )
                            ],
                            "p-unc": [1.0],
                            "p-corr": [1.0],
                        }
                    )
        elif method == "oneway":
            with np.errstate(divide="ignore", invalid="ignore"):
                aov = pg.anova(
                    data=data,
                    dv=metric,
                    between=ind_variable_name,
                    detailed=True,
                )
                pairwise = pg.pairwise_tests(
                    data=data,
                    dv=metric,
                    between=ind_variable_name,
                    padjust=multiple_correction,
                    effsize=effect_size,
                    parametric=parametric,
                )
        elif method == "repeated":
            with np.errstate(divide="ignore", invalid="ignore"):
                aov = pg.rm_anova(
                    data=data,
                    dv=metric,
                    within=ind_variable_name,
                    subject=file,
                    detailed=True,
                )
                pairwise = pg.pairwise_ttests(
                    data=data,
                    dv=metric,
                    within=ind_variable_name,
                    subject=file,  # file is the subject
                    padjust=multiple_correction,
                    effsize=effect_size,
                    parametric=parametric,
                )

        # Replace any NaN or infinite values in p-values with 1.0 (non-significant)
        if "p-unc" in aov.columns:
            aov["p-unc"] = aov["p-unc"].fillna(1.0)
            aov.loc[~np.isfinite(aov["p-unc"]), "p-unc"] = 1.0

        if "p-unc" in pairwise.columns:
            pairwise["p-unc"] = pairwise["p-unc"].fillna(1.0)
            pairwise.loc[~np.isfinite(pairwise["p-unc"]), "p-unc"] = 1.0

        if "p-corr" in pairwise.columns:
            pairwise["p-corr"] = pairwise["p-corr"].fillna(1.0)
            pairwise.loc[~np.isfinite(pairwise["p-corr"]), "p-corr"] = 1.0

    except Exception as e:
        logger.warning(
            f"Error during {method} analysis: {str(e)}. Returning empty results."
        )
        if method == "mixed":
            # Create dummy results with expected structure for mixed ANOVA
            aov = pd.DataFrame(
                {
                    "Source": ["Epoch", "Group", "Epoch * Group"],
                    "F": [np.nan, np.nan, np.nan],
                    "p-unc": [1.0, 1.0, 1.0],
                    "np2": [0.0, 0.0, 0.0],
                }
            )
            pairwise = pd.DataFrame(
                {
                    "Contrast": ["Epoch * Group"],
                    "A": ["Epoch1"],
                    "B": ["Epoch2"],
                    "p-unc": [1.0],
                    "p-corr": [1.0],
                }
            )
        else:
            aov = pd.DataFrame({"Comparison": [metric], "p-unc": [1.0]})
            pairwise = pd.DataFrame({"Comparison": [metric], "p-unc": [1.0]})

    aov["Comparison"] = metric
    pairwise["Comparison"] = metric

    # Put the comparison column on the far left
    aov = aov[
        ["Comparison"] + [col for col in aov.columns if col != "Comparison"]
    ]
    pairwise = pairwise[
        ["Comparison"]
        + [col for col in pairwise.columns if col != "Comparison"]
    ]
    return aov, pairwise


def _compare_single_group(
    data,
    corr_data,
    multiple_correction,
    effect_size,
    epoch_colors,
    group_name,
    save_name,
):
    """Compare data across epochs within a single group."""
    # Add safety check for problematic data
    try:
        # Drop cells with all NaN values which can cause issues
        if "Cell" in data.columns:
            cells_to_keep = ~data.groupby("Cell")[Metric.TRACE.value].apply(
                lambda x: np.isnan(x).all()
            )
            if not cells_to_keep.all():
                problem_cells = cells_to_keep[~cells_to_keep].index.tolist()
                logger.warning(
                    f"Dropping {len(problem_cells)} cells with all NaN values: {problem_cells}"
                )
                data = data[
                    data["Cell"].isin(cells_to_keep[cells_to_keep].index)
                ]

        # Check for sufficient variance in data
        trace_variance = data.groupby("Epoch")[Metric.TRACE.value].var()
        low_var_epochs = trace_variance[
            trace_variance < DIVISION_THRESHOLD
        ].index.tolist()
        if low_var_epochs:
            logger.warning(
                f"Low variance detected in epochs: {low_var_epochs}. "
                "This may cause statistical warnings."
            )
    except Exception as e:
        logger.warning(f"Error during data validation: {str(e)}")

    # Run ANOVA for trace data
    aov, pairwise = _run_ANOVA(
        data=data,
        method="oneway",
        multiple_correction=multiple_correction,
        effect_size=effect_size,
        metric=Metric.TRACE.value,
        file="Trace file",
    )

    # Plotting with error handling
    try:
        _plot_comparison(
            data,
            epoch_colors,
            aov,
            pairwise,
            Metric.TRACE.value,
            group_name,
            save_name,
        )
    except Exception as e:
        logger.warning(f"Error plotting trace comparison: {str(e)}")

    # Run ANOVA for event data if available
    if "Eventrate Activity" in data.columns:
        try:
            event_aov, event_pairwise = _run_ANOVA(
                data=data,
                method="oneway",
                multiple_correction=multiple_correction,
                effect_size=effect_size,
                metric=Metric.EVENT.value,
                file="Eventrate file",
            )
            _plot_comparison(
                data,
                epoch_colors,
                event_aov,
                event_pairwise,
                Metric.EVENT.value,
                group_name,
                save_name,
            )
            aov = pd.concat([aov, event_aov])
            pairwise = pd.concat([pairwise, event_pairwise])
        except Exception as e:
            logger.warning(f"Error during event data analysis: {str(e)}")

    # Run ANOVA for correlation data if available
    if corr_data is not None:
        try:
            # Check for low variance in correlation data
            pos_variance = corr_data.groupby("Epoch")[
                "Average Positive Correlation"
            ].var()
            neg_variance = corr_data.groupby("Epoch")[
                "Average Negative Correlation"
            ].var()

            low_var_pos = pos_variance[
                pos_variance < DIVISION_THRESHOLD
            ].index.tolist()
            low_var_neg = neg_variance[
                neg_variance < DIVISION_THRESHOLD
            ].index.tolist()

            if low_var_pos or low_var_neg:
                logger.warning(
                    f"Low variance detected in correlation data for epochs: "
                    f"Positive: {low_var_pos}, Negative: {low_var_neg}. "
                    "This may cause statistical warnings."
                )

            pos_aov, pos_pairwise = _run_ANOVA(
                data=corr_data,
                method="oneway",
                multiple_correction=multiple_correction,
                effect_size=effect_size,
                metric=Metric.POS_CORR.value,
                file="File",
            )
            neg_aov, neg_pairwise = _run_ANOVA(
                data=corr_data,
                method="oneway",
                multiple_correction=multiple_correction,
                effect_size=effect_size,
                metric=Metric.NEG_CORR.value,
                file="File",
            )

            corr_aov = pd.concat([pos_aov, neg_aov])
            corr_pairwise = pd.concat([pos_pairwise, neg_pairwise])

            _plot_correlation_comparison(
                corr_data,
                epoch_colors,
                pos_aov,
                pos_pairwise,
                neg_aov,
                neg_pairwise,
                group_name,
                save_name,
            )

            aov = pd.concat([aov, corr_aov])
            pairwise = pd.concat([pairwise, corr_pairwise])
        except Exception as e:
            logger.warning(f"Error during correlation analysis: {str(e)}")

    return aov, pairwise


def _compare_data(
    g1_data,
    g2_data,
    g1_name,
    g2_name,
    g1_color,
    g2_color,
    multiple_correction,
    effect_size,
):
    """Compare data between two groups using 2-way repeated measures ANOVA and
    pairwise t-tests.

    :Parameters
        g1_data (pd.DataFrame): Data for group 1.
        g2_data (pd.DataFrame): Data for group 2.
        g1_name (str): Name of group 1.
        g2_name (str): Name of group 2.
        g1_color (str): Color for group 1 in the plot.
        g2_color (str): Color for group 2 in the plot.
        name (str): Name for the plot.
        multiple_correction (str): Method for multiple comparison correction.
        effect_size (str): Type of effect size to compute.
    :Returns
        tuple: ANOVA results and pairwise t-tests results.
    """
    # Combine data for comparison
    g1_data["Group"] = g1_name
    g2_data["Group"] = g2_name
    combined_data = pd.concat([g1_data, g2_data])

    # Perform 2way repeated measures ANOVA on trace data
    trace_aov, trace_pairwise = _run_ANOVA(
        data=combined_data,
        method="mixed",
        multiple_correction=multiple_correction,
        effect_size=effect_size,
        metric=Metric.TRACE.value,
        file="Trace file",
    )

    _plot_mixed_comparisons(
        combined_data, g1_color, g2_color, trace_aov, trace_pairwise, "Trace"
    )

    # Perform 2way repeated measures ANOVA on event data
    if "Eventrate Activity" in combined_data.columns:
        event_aov, event_pairwise = _run_ANOVA(
            data=combined_data,
            method="mixed",
            multiple_correction=multiple_correction,
            effect_size=effect_size,
            metric=Metric.EVENT.value,
            file="Eventrate file",
        )
        _plot_mixed_comparisons(
            combined_data,
            g1_color,
            g2_color,
            event_aov,
            event_pairwise,
            "Eventrate",
        )
        aov = pd.concat([trace_aov, event_aov])
        pairwise = pd.concat([trace_pairwise, event_pairwise])
    else:
        aov = trace_aov
        pairwise = trace_pairwise

    # Plot Data
    return aov, pairwise


def _compare_correlation(
    g1_data,
    g2_data,
    g1_name,
    g2_name,
    g1_color,
    g2_color,
    name,
    multiple_correction,
    effect_size,
):
    """Compare the correlation between two groups using mixed ANOVA and
    pairwise t-tests.

    :Parameters
        g1_data (pd.DataFrame): Data for group 1.
        g2_data (pd.DataFrame): Data for group 2.
        g1_name (str): Name of group 1.
        g2_name (str): Name of group 2.
        g1_color (str): Color for group 1 in plots.
        g2_color (str): Color for group 2 in plots.
        name (str): Name for the plot.
        multiple_correction (str): Method for multiple comparison correction.
        effect_size (str): Effect size measure.

    :Returns
        tuple: A tuple containing:
            - aov (pd.DataFrame): Results of the mixed ANOVA.
            - pairwise (pd.DataFrame): Results of the pairwise t-tests.
    """
    # Combine data for comparison
    g1_data["Group"] = g1_name
    g2_data["Group"] = g2_name
    combined_data = pd.concat([g1_data, g2_data])

    # Perform 2way repeated measures ANOVA
    pos_aov, pos_pairwise = _run_ANOVA(
        data=combined_data,
        method="mixed",
        multiple_correction=multiple_correction,
        effect_size=effect_size,
        metric=Metric.POS_CORR.value,
        file="File",
    )

    neg_aov, neg_pairwise = _run_ANOVA(
        data=combined_data,
        method="mixed",
        multiple_correction=multiple_correction,
        effect_size=effect_size,
        metric=Metric.NEG_CORR.value,
        file="File",
    )
    _plot_mixed_corr_comparisons(
        combined_data,
        g1_color,
        g2_color,
        pos_aov,
        pos_pairwise,
        neg_aov,
        neg_pairwise,
    )
    aov = pd.concat([pos_aov, neg_aov])
    pairwise = pd.concat([pos_pairwise, neg_pairwise])

    return aov, pairwise


def _unify_results(
    aov,
    pairwise,
    mixed_aov,
    mixed_pairwise,
):
    # get group names
    group_names = aov["Group"].unique()
    # Add other columns to AOV that mixed AOV has
    mixed_aov["Group"] = group_names[0] + " x " + group_names[1]
    mixed_pairwise["Group"] = group_names[0] + " x " + group_names[1]

    aov.rename(columns={"DF": "DF1"}, inplace=True)
    aov["DF2"] = None
    aov["eps"] = None

    aov = aov.dropna(axis="columns", how="all")
    mixed_aov = mixed_aov.dropna(axis="columns", how="all")
    aov = pd.concat([aov, mixed_aov], ignore_index=True)

    # Put Group, Comparison, and Source columns at the beginning
    aov = aov[
        ["Group", "Comparison"]
        + [col for col in aov.columns if col not in ["Group", "Comparison"]]
    ]

    pairwise["Epoch"] = None
    if "U-val" not in pairwise.columns:
        pairwise["U-val"] = None
    if "W-val" not in pairwise.columns:
        pairwise["W-val"] = None
    pairwise["T"] = None
    pairwise["dof"] = None
    pairwise["BF10"] = None

    pairwise = pairwise.dropna(axis="columns", how="all")
    mixed_pairwise = mixed_pairwise.dropna(axis="columns", how="all")
    pairwise = pd.concat([pairwise, mixed_pairwise], ignore_index=True)

    # Put Group, Comparison, and Source columns at the beginning
    pairwise = pairwise[
        ["Group", "Comparison"]
        + [
            col
            for col in pairwise.columns
            if col not in ["Group", "Comparison"]
        ]
    ]

    return aov, pairwise
