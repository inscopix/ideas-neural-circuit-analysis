"""Plotting utilities for state-epoch analysis.

This module provides reusable plotting utilities and helpers
for creating consistent visualizations across different tools.
"""

import logging
import matplotlib.pyplot as plt
from beartype.typing import List, Optional, Tuple, Callable, Any

logger = logging.getLogger(__name__)


def add_epoch_overlays_to_axes(
    ax: "plt.Axes",
    epochs: List[Tuple[float, float]],
    epoch_colors: List[str],
    epoch_names: List[str],
    alpha: float = 0.3,
) -> None:
    """Add epoch overlays to a matplotlib axes.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to add overlays to
    epochs : List[Tuple[float, float]]
        List of epoch time ranges (start, end)
    epoch_colors : List[str]
        List of colors for each epoch
    epoch_names : List[str]
        List of names for each epoch
    alpha : float, default=0.3
        Transparency level for overlays

    """
    for i, (start, end) in enumerate(epochs):
        color = epoch_colors[i] if i < len(epoch_colors) else f"C{i % 10}"
        name = epoch_names[i] if i < len(epoch_names) else f"epoch_{i+1}"

        ax.axvspan(
            start, end, alpha=alpha, color=color, label=f"Epoch: {name}"
        )


def save_figure_with_cleanup(
    fig: "plt.Figure",
    output_path: str,
    title: str = "preview",
    file_format: str = "svg",
    dpi: int = 300,
) -> None:
    """Save figure with consistent formatting and cleanup.

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure to save
    output_path : str
        Full path for output file
    title : str, default="preview"
        Description for logging
    file_format : str, default="svg"
        File format (svg, png, pdf, etc.)
    dpi : int, default=300
        Resolution for raster formats

    """
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=dpi,
        format=file_format,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)
    logger.info(f"Created {title}: {output_path}")


def validate_data_availability(
    data: Any, data_name: str, warning_msg: Optional[str] = None
) -> bool:
    """Validate that data is available for plotting.

    Args
    ----
        data: Data to validate
        data_name: Name of data for logging
        warning_msg: Custom warning message

    Returns
    -------
        True if data is valid, False otherwise

    """
    if data is None:
        msg = warning_msg or f"No {data_name} data available"
        logger.warning(msg)
        return False
    return True


def create_dual_panel_plot_with_epoch_overlays(
    data,
    epochs: List[Tuple[float, float]],
    boundaries: List[float],
    period: float,
    epoch_colors: List[str],
    epoch_names: List[str],
    output_path: str,
    bottom_panel_callback: Callable,
    top_panel_ylabel: str = "Mean Activity",
    **callback_kwargs,
) -> None:
    """Create unified dual-panel plot with epoch overlays.

    This consolidates common plotting logic for trace and event epoch overlays.

    :Args
    ----
        data: Input data array for population average calculation
        epochs: List of epoch time periods as tuples
        boundaries: Cellset boundaries for vertical lines
        period: Sampling period for time axis
        epoch_colors: Colors for epoch overlays
        epoch_names: Names for epoch legend
        output_path: Full path for output file
        bottom_panel_callback: Function to plot bottom panel content
        top_panel_ylabel: Y-axis label for top panel
        **callback_kwargs: Additional arguments for bottom panel callback

    """
    import numpy as np

    # Extend epoch parameters to match epoch count for plotting
    while len(epoch_colors) < len(epochs):
        epoch_colors.append("lightblue")
    while len(epoch_names) < len(epochs):
        epoch_names.append(f"Epoch_{len(epoch_names)}")

    # Set consistent font size
    plt.rcParams.update({"font.size": 12})

    # Create figure with 2 panels
    fig, ax = plt.subplots(
        nrows=2, ncols=1, figsize=(15, 10), height_ratios=[1, 8]
    )

    # Top panel: Population average activity
    if "smoothed_activity" in callback_kwargs:
        mean_activity = callback_kwargs["smoothed_activity"]
    else:
        mean_activity = np.nanmean(data, axis=1)

    # Create time axis for population average
    if data.ndim > 1:
        time_axis = np.arange(data.shape[0]) * period
    else:
        time_axis = np.arange(len(data)) * period

    ax[0].plot(time_axis, mean_activity, color="black")
    last_time = time_axis[-1] if len(time_axis) > 0 else 0
    ax[0].set_xlim([0, last_time])
    ax[0].set_title("Population Average Activity", fontdict={"fontsize": 13})
    ax[0].set_ylabel(top_panel_ylabel, fontdict={"fontsize": 12})
    ax[0].spines["right"].set_visible(False)
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["bottom"].set_visible(False)
    ax[0].set_xticks([])

    # Bottom panel: Use callback to plot specific content
    bottom_panel_callback(ax[1], data, period, **callback_kwargs)

    # Add epoch overlays to both panels
    add_epoch_overlays_to_axes(
        ax[0], epochs, epoch_colors, epoch_names, alpha=0.4
    )
    add_epoch_overlays_to_axes(
        ax[1], epochs, epoch_colors, epoch_names, alpha=0.4
    )

    # Add cellset boundaries to both panels
    if boundaries is not None and len(boundaries) > 2:
        for idx, boundary in enumerate(boundaries[1:-1]):
            label = "cellset boundary" if idx == 0 else None
            ax[0].axvline(boundary, linestyle="--", color="k", label=label)
            ax[1].axvline(boundary, linestyle="--", color="k", label=label)

    # Add consistent legend
    handles = []
    for epoch_name, epoch_color in zip(epoch_names, epoch_colors):
        handles.append(
            plt.Line2D([0], [0], color=epoch_color, lw=4, label=epoch_name)
        )

    # Add legend to the top panel
    if len(epoch_names) > 0:
        ax[0].legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
        )

    # Save figure using optimized SVG
    from utils.utils import save_optimized_svg

    save_optimized_svg(
        fig,
        output_path,
        max_size_mb=10,
        pad_inches=0.3,
    )


def plot_traces_bottom_panel(ax, traces, period: float, **kwargs) -> None:
    """Bottom panel callback for trace plotting.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes for plotting
    traces : ndarray
        Trace data array
    period : float
        Sampling period
    **kwargs : dict
        Additional keyword arguments

    """
    import numpy as np
    import scipy.stats

    # Create time axis
    time = np.arange(0, traces.shape[0] * period, period)
    num_cells = min(traces.shape[1], 50)

    # Account for off-by-one error
    if len(time) != traces.shape[0]:
        time = time[:-1]

    # Plot traces with proper spacing
    spacing = 5.01
    for y in range(num_cells):
        x = y * spacing + scipy.stats.zscore(traces[:, y], nan_policy="omit")
        ax.plot(time, x, linewidth=1, color="black")

    # Style the bottom panel
    ax.set_ylabel("Cell index", fontdict={"fontsize": 12})
    ax.set_xlabel("Time (s)", fontdict={"fontsize": 12})
    ax.set_title(
        f"Traces of the first {num_cells} cells", fontdict={"fontsize": 13}
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    last_time = time[-1] if len(time) > 0 else 0
    ax.set_xlim([0, last_time])
    ax.set_ylim([-5, 15 + num_cells * spacing])
    ax.set_yticks([])


def plot_events_bottom_panel(
    ax,
    event_timeseries,
    period: float,
    events: List,
    **kwargs,
) -> None:
    """Bottom panel callback for event raster plotting.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes for plotting
    event_timeseries : ndarray
        Event timeseries data
    period : float
        Sampling period
    events : List
        List of event times
    **kwargs : dict
        Additional keyword arguments

    """
    # Bottom panel: Event raster plot
    ax.eventplot(
        events, color="black", linelengths=0.5, linewidths=0.8, alpha=0.5
    )
    ax.set_ylabel("Cell index", fontdict={"fontsize": 12})
    ax.set_xlabel("Time (s)", fontdict={"fontsize": 12})
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Raster plot of events", fontdict={"fontsize": 13})
