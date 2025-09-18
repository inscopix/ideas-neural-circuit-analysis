import warnings

import numpy as np
import scipy
from beartype import beartype
from beartype.typing import List, Optional, Tuple, Union
from scipy.ndimage import convolve
from skimage.measure import find_contours, perimeter

import ideas.measures as measures
from ideas.types import NumpyFloatVector, NumpyVector
from ideas.validation import (
    cell_set_series,
    event_set_series,
    vessel_set_series,
)

try:
    import isx
except ImportError as err:
    print(
        """
+-----------------------------------------+
| Could not import isx. You need to have  |
| the IDPS API installed or the python-   |
| based experimental API                  |
+-----------------------------------------+
"""
    )
    raise err


def movie_to_correlation_image(
    movie_array: np.ndarray,
    *,
    eight_neighbours: bool = True,
    swap_dim: bool = True,
    rolling_window: int = None,
) -> np.ndarray:
    """Compute the correlation image for the input dataset
    Y using a faster FFT based method.
    Taken from caiman implementation:
    https://github.com/flatironinstitute/CaImAn/blob/
    b5932abe85c478fae4004e8df14535f3f5f7274a/caiman/summary_images.py#L73
    Parameters:
        movie_array <np.array>: Input movie data in 3D or 4D format
        eight_neighbours <Bool>: Use 8 neighbors if true,
        and 4 if false for 3D data (default =
        True) Use 6 neighbors for 4D data, irrespectively.
        swap_dim <Bool>: True indicates that time is listed in the last axis of Y (matlab format)
            and moves it in the front


        rolling_window <int>: Rolling window
    Returns:
        corr_img <np.array>: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels


    this was originally in the IDPS toolbox, moved here and
    cleaned up a bit.
    """

    if swap_dim:
        movie_array = np.transpose(
            movie_array,
            tuple(
                np.hstack(
                    (movie_array.ndim - 1, list(range(movie_array.ndim))[:-1])
                )
            ),
        )

    movie_array = movie_array.astype("float32")
    if rolling_window is None:
        movie_array -= np.mean(movie_array, axis=0)
        movie_array_std = np.std(movie_array, axis=0)
        movie_array_std[movie_array_std == 0] = np.inf
        movie_array /= movie_array_std
    else:
        movie_array_sum = np.cumsum(movie_array, axis=0)
        movie_array_rm = (
            movie_array_sum[rolling_window:]
            - movie_array_sum[:-rolling_window]
        ) / rolling_window
        movie_array[:rolling_window] -= movie_array_rm[0]
        movie_array[rolling_window:] -= movie_array_rm
        del movie_array_rm, movie_array_sum
        movie_array_std = np.cumsum(movie_array**2, axis=0)
        movie_array_rst = np.sqrt(
            (
                movie_array_std[rolling_window:]
                - movie_array_std[:-rolling_window]
            )
            / rolling_window
        )
        movie_array_rst[movie_array_rst == 0] = np.inf
        movie_array[:rolling_window] /= movie_array_rst[0]
        movie_array[rolling_window:] /= movie_array_rst
        del movie_array_std, movie_array_rst

    if movie_array.ndim == 4:
        if eight_neighbours:
            sz = np.ones((3, 3, 3), dtype="float32")
            sz[1, 1, 1] = 0
        else:
            # yapf: disable
            sz = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                           [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                           [[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
                          dtype='float32')
            # yapf: enable
    else:
        if eight_neighbours:
            sz = np.ones((3, 3), dtype="float32")
            sz[1, 1] = 0
        else:
            sz = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype="float32")

    movie_array_conv = convolve(
        movie_array, sz[np.newaxis, :], mode="constant"
    )
    MASK = convolve(
        np.ones(movie_array.shape[1:], dtype="float32"),
        sz,
        mode="constant",
    )

    # YYconv is the product of the actual movie array
    # with its convolved version
    YYconv = movie_array_conv * movie_array
    del movie_array, movie_array_conv
    if rolling_window is None:
        corr_img = np.mean(YYconv, axis=0) / MASK
    else:
        YYconv_cs = np.cumsum(YYconv, axis=0)
        del YYconv
        YYconv_rm = (
            YYconv_cs[rolling_window:] - YYconv_cs[:-rolling_window]
        ) / rolling_window
        del YYconv_cs
        corr_img = YYconv_rm / MASK

    return corr_img


@beartype
def cell_set_to_traces(
    files: Union[str, List[str]],
    sort_by_time: bool = True,
    tolerance: float = 1e-6,
    flush_concatenate: bool = True,
):
    # -> (NumpyFloat2DArray, NumpyFloatVector):
    """Reads cell set file(s) and retrieves all traces.

    Parameters:
    files (Union[str, List[str]]): A single file path or a list of file paths to read.
    sort_by_time (bool): If True (default), sorts cell set series by start time
    tolerance (float): tolerance for potential discrepancies in the sampling
    rate of cell set series
    flush_concatenate (bool): If True, concatenates traces directly without considering time gaps.
                              If False, fills gaps between traces with NaNs.

    Returns:
    numpy.ndarray: A 2D numpy array (time x num_cells) containing the traces.
    """

    if isinstance(files, str):
        files = [files]

    files = cell_set_series(
        files=files,
        sort_by_time=sort_by_time,
        tolerance=tolerance,
    )

    all_traces = []

    for file in files:
        cellset = isx.CellSet.read(file)
        num_cells = cellset.num_cells
        num_samples = cellset.timing.num_samples

        if not flush_concatenate:
            # If its the first file, set the initial timing information
            if file == files[0]:
                time_origin = cellset.timing.start.to_datetime()
                period = cellset.timing.period.secs_float
                origin_rec_length = num_samples
            else:
                # if it is not the first file, calculate the time offset
                cur_start_time = cellset.timing.start.to_datetime()
                # convert the seconds to approximate number of samples
                time_offset = int(
                    (cur_start_time - time_origin).total_seconds() / period
                )
                # remove the number of samples from the previous file
                time_offset -= origin_rec_length
                if time_offset < 0:
                    time_offset = (
                        0  # Make sure we don't have negative time offset
                    )
                # create a matrix of NaNs to fill the gap
                time_indices = np.full(
                    (time_offset, cellset.num_cells), np.nan
                )
                # Update the origin for the next file
                time_origin = cur_start_time
                origin_rec_length = num_samples

        traces = np.full((num_samples, num_cells), np.nan)

        for i in range(num_cells):
            traces[:, i] = cellset.get_cell_trace_data(i)
        # If there is a gap, concatenate the gap before the traces
        if not flush_concatenate and file != files[0]:
            all_traces.append(time_indices)

        # Add the real data
        all_traces.append(traces)

    return np.concatenate(all_traces)


@beartype
def event_set_to_events(
    files: Union[str, List[str]], flush_concatenate: bool = True
):
    """Converts a series of event set files into lists of offsets and amplitudes for each cell.
    Parameters:
    files (Union[str, List[str]]): A single file path or a list of file paths to event set files.
    flush_concatenate (bool): If True, concatenates events by adding the number of samples to the time offset.
                              If False, concatenates events based on the recording start times. Default is True.
    Returns:
    Tuple[List[List[float]], List[List[float]]]: Two lists of lists, where each inner list contains the offsets and amplitudes
                                                 for each cell across all event set files.
    """
    # Ensure that files is a list
    if isinstance(files, str):
        files = [files]

    # Ensure that files are in a series, and in order
    files = event_set_series(files)
    # initialize time offset
    time_offset = 0

    # Initalize storage for all offsets and amplitudes
    # Each cell gets its own list of offsets and amplitudes
    all_offsets = []
    all_amplitudes = []
    # Go through each file in the series
    for file in files:
        # load the event set
        eventset = isx.EventSet.read(file)
        if file == files[0]:
            # get initial time value
            time_origin = eventset.timing.start.to_datetime()
        else:
            if flush_concatenate:
                # the time_offset is exactly the previous time offset plus the number of samples
                time_offset += (
                    eventset.timing.num_samples
                    * eventset.timing.period.secs_float
                )
            else:
                # time_offset is based on the recording start times
                cur_start_time = eventset.timing.start.to_datetime()
                # Calculate the time offset in seconds
                time_offset = (cur_start_time - time_origin).total_seconds()

        # get the number of cells This is assumed to be the same accross all files
        num_cells = eventset.num_cells
        for i in range(num_cells):
            # get the offsets and amplitudes for each cell
            # offsets is a uint in microseconds since start of recording
            offsets, amplitudes = eventset.get_cell_data(i)
            # convert offsets to seconds
            offsets = offsets / 1e6
            # add the time offset to the offsets
            offsets += time_offset
            # offsets and amplitudes are stored in an array, convert to list and add values to the correct cell in the storage list
            if len(all_offsets) < i + 1:
                # if the cell does not exist in the storage list, add it
                all_offsets.append(offsets.tolist())
                all_amplitudes.append(amplitudes.tolist())
            else:
                # the cell already exists in the storage list, extend the list
                all_offsets[i].extend(offsets.tolist())
                all_amplitudes[i].extend(amplitudes.tolist())

    return all_offsets, all_amplitudes


def offsets_to_timeseries(
    offsets, status, timeseries_shape, period, only_accepted=False
):
    """
    Converts event offsets to a timeseries representation.
    Parameters:
    offsets (list of lists): A list where each element is a list of event offsets for a cell.
    status (list): A list of status strings corresponding to each cell's events, e.g., "accepted" or "rejected".
    timeseries_shape (tuple): The shape of the output timeseries array (num_timepoints, num_cells).
    period (float): The period used to convert offsets to indices.
    only_accepted (bool, optional): If True, only include events with status "accepted". If False, include all events except those with status "rejected". Default is False.
    Returns:
    np.ndarray: A 2D numpy array of shape `trace_shape` where each event is represented in Hz at the corresponding timepoint and cell index.
    """
    # filter out rejected cells if only_accepted is True
    if not only_accepted:
        offsets = [
            cell for cell, stat in zip(offsets, status) if stat != "rejected"
        ]
    else:
        offsets = [
            cell for cell, stat in zip(offsets, status) if stat == "accepted"
        ]

    # convert the event times to indices
    offset_indices = [
        np.array([int(offset / period) for offset in cell]) for cell in offsets
    ]
    assert timeseries_shape[1] == len(
        offsets
    ), f"Number of cells in cell set ({timeseries_shape[1]}) does not match number of cells in event set ({len(offsets)})"

    assert (
        len(offset_indices) == timeseries_shape[1]
    ), f"Number of cells in cell set ({timeseries_shape[1]}) does not match number of cells in event set ({len(offset_indices)})"

    # convert the indices to timeseries
    event_timeseries = np.zeros(timeseries_shape)
    for idx, cell in enumerate(offset_indices):
        for event in cell:
            event_timeseries[event, idx] = (
                1 / period
            )  # this encodes the event in Hz, which allows for easy smoothing, averaging, etc.

    return event_timeseries


@beartype
def vessel_set_to_diameters(
    files: Union[str, List[str]],
    sort_by_time: bool = True,
    tolerance: float = 1e-6,
    flush_concatenate: bool = True,
):
    """Reads vessel set file(s) and retrieves all diameters.

    Parameters:
    files (Union[str, List[str]]): A single file path or a list of file paths to read.
    sort_by_time (bool): If True (default), sorts vessel set series by start time
    tolerance (float): tolerance for potential discrepancies in the sampling
    rate of vessel set series
    flush_concatenate (bool): If True, concatenates traces directly without considering time gaps.
                              If False, fills gaps between traces with NaNs.

    Returns:
    numpy.ndarray: A 2D numpy array (time x num_cells) containing the traces.
    """

    if isinstance(files, str):
        files = [files]

    files = vessel_set_series(
        files=files,
        sort_by_time=sort_by_time,
        tolerance=tolerance,
    )

    all_diams = []

    for file in files:
        vesselset = isx.VesselSet.read(file)
        num_vessels = vesselset.num_vessels
        num_samples = vesselset.timing.num_samples

        if not flush_concatenate:
            # If its the first file, set the initial timing information
            if file == files[0]:
                time_origin = vesselset.timing.start.to_datetime()
                period = vesselset.timing.period.secs_float
                origin_rec_length = num_samples
            else:
                # if it is not the first file, calculate the time offset
                cur_start_time = vesselset.timing.start.to_datetime()
                # convert the seconds to approximate number of samples
                time_offset = int(
                    (cur_start_time - time_origin).total_seconds() / period
                )
                # remove the number of samples from the previous file
                time_offset -= origin_rec_length
                if time_offset < 0:
                    time_offset = (
                        0  # Make sure we don't have negative time offset
                    )
                # create a matrix of NaNs to fill the gap
                time_indices = np.full(
                    (time_offset, vesselset.num_vessels), np.nan
                )
                # Update the origin for the next file
                time_origin = cur_start_time
                origin_rec_length = num_samples

        diams = np.full((num_samples, num_vessels), np.nan)

        for i in range(num_vessels):
            diams[:, i] = vesselset.get_vessel_trace_data(i)
        # If there is a gap, concatenate the gap before the traces
        if not flush_concatenate and file != files[0]:
            all_diams.append(time_indices)

        # Add the real data
        all_diams.append(diams)

    return np.concatenate(all_diams)


def vessel_set_plot_elements(
    files: Union[str, List[str]],
    sort_by_time: bool = True,
    tolerance: float = 1e-6,
):
    """Reads vessel set file(s) and image data and line information

    Parameters:
    files (Union[str, List[str]]): A single file path or a list of file paths to read.
    sort_by_time (bool): If True (default), sorts cell set series by start time
    tolerance (float): tolerance for potential discrepancies in the sampling
    rate of cell set series
    Returns:
    list: A list of numpy array images
    list: A list of lists containing vessel line information to plot on each image
    """

    if isinstance(files, str):
        files = [files]

    files = vessel_set_series(
        files=files,
        sort_by_time=sort_by_time,
        tolerance=tolerance,
    )

    # initialize outputs
    vesselset_images = []
    vesselset_lines = []

    # run through each file
    for file in files:
        vesselset = isx.VesselSet.read(file)
        num_vessels = vesselset.num_vessels
        cur_file_lines = []

        # get each line coordinate for each file
        for i in range(num_vessels):
            cur_file_lines.append(vesselset.get_vessel_line_data(i))

        vesselset_images.append(vesselset.get_vessel_image_data(0))
        vesselset_lines.append(cur_file_lines)

    return vesselset_images, vesselset_lines


@beartype
def cell_set_to_time(
    files: Union[str, List[str]],
    sort_by_time: bool = True,
    tolerance: float = 1e-6,
) -> NumpyFloatVector:
    """returns time vector for a cellset.isxd file

    time is a vector in seconds from the start of the recording

    """

    if isinstance(files, str):
        files = [files]

    files = cell_set_series(
        files=files,
        sort_by_time=sort_by_time,
        tolerance=tolerance,
    )

    all_time = []
    last_time = 0
    for file in files:
        cellset = isx.CellSet.read(file)

        dt = cellset.timing.period.secs_float
        time = np.linspace(
            0,
            (cellset.timing.num_samples - 1) * dt,
            cellset.timing.num_samples,
        )

        all_time.append(time + last_time)
        last_time += time[-1]

    return np.concatenate(all_time)


@beartype
def cell_set_to_positions(
    files: Union[str, List[str]],
    sort_by_time: bool = True,
    tolerance: float = 1e-6,
    *,
    threshold: float = 4.0,
):
    """returns cell positions given a cell set file path"""

    if isinstance(files, str):
        files = [files]

    files = cell_set_series(
        files=files,
        sort_by_time=sort_by_time,
        tolerance=tolerance,
    )

    # just read out from first file
    # assume that this is a valid series

    cellset = isx.CellSet.read(files[0])

    num_cells = cellset.num_cells

    positions_x = np.full((num_cells), np.nan)
    positions_y = np.full((num_cells), np.nan)

    for i in np.arange(num_cells).tolist():
        footprint = cellset.get_cell_image_data(i)
        footprint -= footprint.min()
        footprint /= footprint.max()
        footprint[footprint < 1 / threshold] = 0
        pos = scipy.ndimage.center_of_mass(footprint)
        positions_y[i] = pos[0]
        positions_x[i] = pos[1]

    return positions_x, positions_y


@beartype
def cell_set_to_status(
    files: Union[str, List[str]],
    sort_by_time: bool = True,
    tolerance: float = 1e-6,
) -> NumpyVector:
    """cellset file to array of status for each neuron

    Parameters:
    files (Union[str, List[str]]): A single file path or a list of file paths to read.
    sort_by_time (bool): If True (default), sorts cell set series by start time
    tolerance (float): tolerance for potential discrepancies in the sampling
    rate of cell set series

    Returns:
    np.array: status array (Nx1)
    """

    if isinstance(files, str):
        files = [files]

    files = cell_set_series(
        files=files,
        sort_by_time=sort_by_time,
        tolerance=tolerance,
    )

    # simply return the status of the first file
    # cell set series validation guarantees that
    # if status are different, then it will not be a
    # valid series

    cellset = isx.CellSet.read(files[0])
    num_cells = cellset.num_cells
    status = [cellset.get_cell_status(i) for i in range(num_cells)]

    return np.array(status)


@beartype
def cell_set_to_footprint_metrics(
    files: Union[str, List[str]],
    sort_by_time: bool = True,
    tolerance: float = 1e-6,
    *,
    threshold: float = 4.0,
) -> dict:
    """cellset file to array of areas of footprints

    counts the number of pixels above some threshold, defined
    by the max of that footprint.

    """

    if isinstance(files, str):
        files = [files]

    files = cell_set_series(
        files=files,
        sort_by_time=sort_by_time,
        tolerance=tolerance,
    )

    # just operate on the first one, assume that
    # this is a valid series (validation done
    # in cell_set_series)

    cellset = isx.CellSet.read(files[0])

    areas = np.zeros(cellset.num_cells)
    perimeters = np.zeros(cellset.num_cells)

    for i in range(cellset.num_cells):
        footprint = cellset.get_cell_image_data(i)

        # discretize
        footprint = footprint > footprint.max() / threshold

        areas[i] = np.sum(footprint)
        perimeters[i] = perimeter(footprint, neighborhood=8)

        if perimeters[i] < 16:
            # we can't estimate this well, so we will default
            # to a value so that the circularity is
            # constrained to 1
            perimeters[i] = np.sqrt(areas[i] * np.pi * 4)

    areas = areas.astype(int)

    # TODO replace this with a nicer roundness measure
    # that depends on the ratios of inscribed to
    # superscribed circles, see:
    # https://en.wikipedia.org/wiki/Roundness
    circularity = (4 * np.pi * areas) / (perimeters**2)

    return dict(
        areas=areas,
        perimeters=perimeters,
        circularity=circularity,
    )


def cell_set_to_contours(
    files: Union[str, List[str]],
    sort_by_time: bool = True,
    tolerance: float = 1e-6,
    *,
    threshold: float = 4.0,
) -> Tuple[List, List]:
    """generate list of contours from a cell set file

    Parameters:
    files (Union[str, List[str]]): A single file path or a list of file paths to read.
    sort_by_time (bool): If True (default), sorts cell set series by start time
    tolerance (float): tolerance for potential discrepancies in the sampling
    rate of cell set series
    threshold (float): to consider footprints within cell
    threshold is a parameter that divides the maximum of each cell's
    footprint, and defines the threshold to draw a contour around


    Returns:
    contours_x: list of list of arrays of contours
    contours_y: list of list of arrays of contours

    This convoluted format is so that this can be
    fed directly into bokeh.multi_polygons
    """

    if isinstance(files, str):
        files = [files]

    files = cell_set_series(
        files=files,
        sort_by_time=sort_by_time,
        tolerance=tolerance,
    )

    # just operate on the first one, assume that
    # this is a valid series (validation done
    # in cell_set_series)

    # generate contours of all cells
    contours_x = []
    contours_y = []

    cellset = isx.CellSet.read(files[0])

    for i in range(cellset.num_cells):
        footprint = cellset.get_cell_image_data(i)

        if footprint.max() == 0:
            warnings.warn(
                f"""cell {i} in {files[0]} has all-zero 
    footprints, which is a bug in the cell extraction. 
    We cannot draw contours for this cell.""",
                stacklevel=2,
            )

            contours_x.append([[0]])
            contours_y.append([[0]])
            continue

        xy = find_contours(footprint.transpose(), footprint.max() / threshold)

        contours_x.append([[xy[0][:, 0]]])
        contours_y.append([[xy[0][:, 1]]])

    return (contours_x, contours_y)


@beartype
def prototypical_cells(
    cell_set_file: str,
    sort_by_time: bool = True,
    tolerance: float = 1e-6,
    *,
    only_from_these_cells: Optional[NumpyVector] = None,
) -> tuple:
    """return indicies of cells with max, median and min skew

    Parameters:
    files (Union[str, List[str]]): A single file path or a list of file paths to read.
    sort_by_time (bool): If True (default), sorts cell set series by start time
    tolerance (float): tolerance for potential discrepancies in the sampling
    rate of cell set series
    only_from_these_cells: array of cell indicies that we are
    allowed to choose from. If None, all cells are allowed

    Returns:
    (max_cell, median_cell, min_cell) tuple of indices

    may contain duplicates

    """
    traces = cell_set_to_traces(
        files=cell_set_file,
        sort_by_time=sort_by_time,
        tolerance=tolerance,
    )

    skew = measures.matrix_skew(traces)

    if only_from_these_cells is not None:
        all_cells = np.arange(len(skew))
        bad_cells = np.setdiff1d(all_cells, only_from_these_cells)
        skew[bad_cells] = np.nan

    return (
        np.nanargmax(skew),
        np.nanargmin(np.abs(skew - np.nanmedian(skew))),
        np.nanargmin(skew),
    )
