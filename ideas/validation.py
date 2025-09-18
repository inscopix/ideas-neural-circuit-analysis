import os

import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import List

from ideas.exceptions import IdeasError
from ideas.utils import (
    _extract_footer,
    _sort_isxd_files_by_start_time,
    check_file_extention_is,
)

try:
    import isx
except ImportError as err:
    print(
        """
+-----------------------------------------+
| Could not import isx. You need to have  |
| the isx API installed from IDPS or from |
| pip                                     |
+-----------------------------------------+
"""
    )
    raise err


@beartype
def _check_columns_in_df(
    *,
    df: pd.DataFrame,
    columns: tuple,
) -> None:
    for column in columns:
        if column not in df:
            raise IdeasError(
                f"""Data frame does not contain
the column requested: {column}. 

The columns in this data frame are:
{list(df.columns)}""",
            )


@beartype
def check_file_exists(file: str) -> None:
    """check if a file exists, fail if not"""

    if not os.path.exists(file):
        raise Exception(f"{file} does not exist.")


def movie_series(
    files: List[str],
    sort_by_time: bool = True,
    tolerance: float = 1e-6,
) -> List[str]:
    """function validates a list of ISXD movies
    and returns a re-ordered list if they can form a valid
    series. throws an error otherwise.

    A list of movies is a valid series if:

    - frame sizes are all the same
    - each movie has a unique start time
    - all movies have the same frame rate

    """
    if len(files) == 0:
        return []

    if len(files) == 1:
        return files

    start_times = np.zeros(len(files))
    pixel_shapes = [None] * len(files)
    periods = np.zeros(len(files))

    for i, file in enumerate(files):
        check_file_exists(file)
        check_file_extention_is(file, ext=".isxd")

        # ensure it consists of an isxd MOVIE
        movie = isx.Movie.read(file)

        # read the metadata and ensure that all the pixel shapes are the same
        pixel_shapes[i] = movie.spacing.num_pixels

        # read start time of the movie
        start_times[i] = movie.timing.start._to_secs_since_epoch().secs_float

        # check that frame rates are the same
        periods[i] = movie.timing.period.secs_float

    for i in range(len(files)):
        if not np.isclose(periods[0], periods[i], atol=tolerance):
            raise Exception(
                f"""[INVALID SERIES] The input files do 
            not form a valid movie series. 
            Frame rates are different across these files.
            Differing frame rates are: {periods[0]} which
            is not the same as {periods[i]}.
            """,
            )

    if pixel_shapes.count(pixel_shapes[0]) != len(pixel_shapes):
        raise Exception(
            """[INVALID SERIES] The input files do 
            not form a valid movie series.
            The pixel sizes of the files provided do
            not match.""",
        )

    if len(np.unique(start_times)) != len(start_times):
        raise Exception(
            """[INVALID SERIES] Files in the series
             do not have unique start times"""
        )

    if sort_by_time:
        files = _sort_isxd_files_by_start_time(files)

    return files


@beartype
def cell_set_series(
    files: List[str],
    sort_by_time: bool = True,
    tolerance: float = 1e-6,
) -> List:
    """Validate isxd file paths for existence and cell set format.

    :param files: list of paths to the input isxd cell set files
    :param sort_by_time: if true, returns isxd cell sets sorted by start time
    :param tolerance: tolerance for potential discrepancies in the sampling
    rate of input cell sets
    :return: list of paths to the input isxd cell set
    files ordered by start time.
    """
    if len(files) == 0:
        return []

    if len(files) == 1:
        return files

    for file in files:
        check_file_exists(file)

        check_file_extention_is(file, ext=".isxd")

        # ensure it consists of an isxd CELL SET
        metadata = _extract_footer(file)
        isxd_type = metadata["type"]
        if isxd_type != 1:
            raise Exception(f"{file} is not a ISXD cell set file")

    start_times = np.zeros(len(files))
    cell_lists = [None] * len(files)

    # to be a valid series, all cell sets must have the
    # same status
    cellset = isx.CellSet.read(files[0])
    num_cells = cellset.num_cells
    status0 = [cellset.get_cell_status(i) for i in range(num_cells)]
    periods = np.zeros(len(files))

    for i, file in enumerate(files):
        metadata = _extract_footer(file)

        cellset = isx.CellSet.read(file)
        num_cells = cellset.num_cells
        status = [cellset.get_cell_status(i) for i in range(num_cells)]

        if status != status0:
            raise Exception(
                f"""[INVALID SERIES] {file} and {files[0]} 
    cannot be part of a valid series because they have 
    different cell statuses"""
            )

        cell_lists[i] = metadata["CellNames"]

        start_times[i] = (
            metadata["timingInfo"]["start"]["secsSinceEpoch"]["num"]
            / metadata["timingInfo"]["start"]["secsSinceEpoch"]["den"]
        )

        # check that frame rates are the same
        periods[i] = (
            metadata["timingInfo"]["period"]["num"]
            / metadata["timingInfo"]["period"]["den"]
        )

    for i in range(len(files)):
        if not np.isclose(periods[0], periods[i], atol=tolerance):
            raise Exception(
                f"""[INVALID SERIES] The input files do 
            not form a valid cell set series. 
            Frame rates are different across these files.
            Differing frame rates are: {periods[0]} which
            is not the same as {periods[i]}.
            """,
            )

    # ensure all cell sets contain the same cells
    if cell_lists.count(cell_lists[0]) != len(cell_lists):
        raise Exception(
            """[INVALID SERIES]
            The input files do not form a series. 
            The cell set files do not describe 
            the same cells.""",
        )

    if sort_by_time:
        files = _sort_isxd_files_by_start_time(files)

    return files


@beartype
def vessel_set_series(
    files: List[str],
    sort_by_time: bool = True,
    tolerance: float = 1e-6,
) -> List:
    """Validate isxd file paths for existence and vessel set format.

    :param _files: list of paths to the
    input isxd vessel set files
    :param sort_by_time: if true, returns isxd cell sets sorted by start time
    :param tolerance: tolerance for potential discrepancies in the sampling
    rate of input vessel sets
    :return: list of paths to the input isxd vessel set
    files ordered by start time.
    """
    if len(files) == 0:
        return []

    if len(files) == 1:
        return files

    for file in files:
        check_file_exists(file)

        check_file_extention_is(file, ext=".isxd")

        # ensure it consists of an isxd VESSEL SET
        metadata = _extract_footer(file)
        isxd_type = metadata["type"]
        if isxd_type != 8:
            raise Exception(f"{file} is not a ISXD vessel set file")

    start_times = np.zeros(len(files))
    vessel_lists = [None] * len(files)

    # to be a valid series, all vessel sets must have the
    # same status
    vesselset = isx.VesselSet.read(files[0])
    num_vessels = vesselset.num_vessels
    status0 = [vesselset.get_vessel_status(i) for i in range(num_vessels)]
    periods = np.zeros(len(files))

    for i, file in enumerate(files):
        metadata = _extract_footer(file)

        vesselset = isx.VesselSet.read(file)
        num_vessels = vesselset.num_vessels
        status = [vesselset.get_vessel_status(i) for i in range(num_vessels)]

        if status != status0:
            raise Exception(
                f"""[INVALID SERIES] {file} and {files[0]} 
    cannot be part of a valid series because they have 
    different vessel statuses"""
            )

        vessel_lists[i] = metadata["VesselNames"]

        start_times[i] = (
            metadata["timingInfo"]["start"]["secsSinceEpoch"]["num"]
            / metadata["timingInfo"]["start"]["secsSinceEpoch"]["den"]
        )

        # check that frame rates are the same
        periods[i] = (
            metadata["timingInfo"]["period"]["num"]
            / metadata["timingInfo"]["period"]["den"]
        )

    for i in range(len(files)):
        if not np.isclose(periods[0], periods[i], atol=tolerance):
            raise Exception(
                f"""[INVALID SERIES] The input files do 
            not form a valid vessel set series. 
            Frame rates are different across these files.
            Differing frame rates are: {periods[0]} which
            is not the same as {periods[i]}.
            """,
            )

    # ensure all vessel sets contain the same vessels
    if vessel_lists.count(vessel_lists[0]) != len(vessel_lists):
        raise Exception(
            """[INVALID SERIES]
            The input files do not form a series. 
            The vessel set files do not describe 
            the same vessels.""",
        )

    if sort_by_time:
        files = _sort_isxd_files_by_start_time(files)

    return files


@beartype
def event_set_series(
    files: List[str],
    sort_by_time: bool = True,
) -> List:
    """Order a list of event set files into a event set series

    Warning! This does not perform any validation on the files
    because functionality to read event set files is currently
    lacking
    """
    if len(files) == 0:
        return []

    if len(files) == 1:
        return files

    if sort_by_time:
        files = _sort_isxd_files_by_start_time(files)

    return files
