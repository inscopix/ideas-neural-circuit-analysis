import ast
import json
import logging
import os
import struct
import sys
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Dict, List, Optional, Tuple, Union
from configobj import ConfigObj

from ideas.types import NumpyFloatVector
from ideas.exceptions import IdeasError
from ideas.metadata import read_isxd_metadata

try:
    import isx
except ImportError:
    pass


def _hex_to_rgb(hex_color: str) -> tuple:
    """Converts a hex color to an RGB tuple.

    Args:
        hex_color (str): The hex color to convert.

    Returns:
        tuple: The RGB tuple.
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)


@beartype
def _find_coord_start(
    row: Union[dict, pd.Series], max_poly_points: Union[int, float]
) -> int:
    """
    Find the starting index of the coordinates in the row of a zones file.
    This function assumes that there are an equal number of x and y coordinates, and that the coordinates are formatted as "X 0", "Y 0", "X 1", "Y 1", etc.

    Note that this function is 0-indexed, and +1 is added to the result to account for the 0-indexing of Python.

    Args:
        row (pd.Series): A row of a zones file.
        max_poly_points (int): The maximum number of points in a polygon.

    """
    max_poly_points = int(max_poly_points)
    if max_poly_points < 0:
        raise ValueError("max_poly_points must be a positive integer")

    if isinstance(row, pd.Series):
        try:
            if pd.isnull(row[f"X {max_poly_points}"]):
                return _find_coord_start(row, max_poly_points - 1)
            else:
                return int(max_poly_points)
        except KeyError:
            return _find_coord_start(row, max_poly_points - 1)
    elif isinstance(row, dict):
        if f"X {max_poly_points}" not in row.keys():
            return _find_coord_start(row, max_poly_points - 1)
        else:
            return int(max_poly_points)
    else:
        raise TypeError("Input must be a pandas Series or a dictionary.")


def _in_notebook() -> bool:
    """small function to determine if we are in a notebook

    from https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook

    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


@beartype
def make_tool_command(module: str, func: str) -> str:
    """makes a tool command from a path to a python module and a
    function name"""

    module = module.replace(os.path.sep, "__").replace(".py", "")
    return module + "__" + func


def split_tool_command(tool_key: str) -> (str, str):
    """utility that converts a tool command to the name
    of the function and the containing module"""

    fragments = tool_key.split("__")

    func_name = fragments[-1]

    module_path = os.path.sep.join(fragments[0:-1]) + ".py"
    return module_path, func_name


def get_tool_command_from_info(tool: dict):
    """get a tool command from an entry in the toolbox info spec."""

    # strip the .sh from the command string to just get <module_name>__<function_name>
    return tool["command"][:-3]


@beartype
def isxd_type(file_path: str) -> str:
    """infer ISXD file type"""

    metadata = _extract_footer(file_path)

    isx_datatype_mapping = {
        0: "miniscope_movie",
        1: "cell_set",
        2: "isxd_behavioral_movie",  # not currently supported on IDEAS
        3: "gpio_data",
        4: "miniscope_image",
        5: "neural_events",
        6: "isxd_metrics",  # not currently supported on IDEAS
        7: "imu_data",
        8: "vessel_set",
    }

    if metadata["type"] not in isx_datatype_mapping.keys():
        raise KeyError(
            f"Unknown key: {metadata['type']}. Expected it to be an +ve integer < 9"
        )

    return isx_datatype_mapping[metadata["type"]]


@beartype
def _footer_length(isxd_file: str) -> int:
    """find the length of the footer in bytes"""

    with open(isxd_file, mode="rb") as file:
        file.seek(-8, os.SEEK_END)
        data = file.read()
    footer_length = struct.unpack("ii", data)[0]

    return footer_length


@beartype
def _extract_footer(isxd_file: str) -> dict:
    """extract movie footer from ISXD file"""

    footer_length = _footer_length(isxd_file)

    with open(isxd_file, mode="rb") as file:
        file.seek(-8 - footer_length - 1, os.SEEK_END)
        data = file.read(footer_length)

    footer = data.decode("utf-8")
    return json.loads(footer)


def _get_isxd_times(input_filename: str):
    """Get the timestamps of every sample of an isxd file from its metadata.

    The timestamps are generated by getting the average sampling period
    of the isxd file.

    :param input_filename str: path to the input file (.isxd)
    :return: The timestamps of every sample in the isxd file
    """

    metadata = _extract_footer(input_filename)
    period = (
        metadata["timingInfo"]["period"]["num"]
        / metadata["timingInfo"]["period"]["den"]
    )
    num_times = metadata["timingInfo"]["numTimes"]
    times = np.linspace(
        0,
        (num_times - 1) * period,
        num_times,
    )
    return times


@beartype
def format_p_value(pvalue: float) -> str:
    """
    utility function to prettify printing of p-values


    Args:
        pvalue (float): p-value ∈ [0,1]

    Returns:
        str: "p = .81" or "p < .01"
    """
    if pvalue < 0.001:
        return "p < .001"
    else:
        return "p = " + "{:.3f}".format(pvalue)


@beartype
def rename_file_using_rule(
    *,
    original_name: str,
    options: List[str],
    inputs: dict,
    index: Optional[int] = None,
) -> str:
    """small utility to rename a file based on certain rules

    this is used both in the toolbox creator app and called
    by function_caller, and therefore belongs here.

    - original_name: name of file to be renamed
    - options: list of rename rules
    - inputs: dictionary from inputs.json file
    - index: # index of a multi-file output (None if single output)

    How it works:



    """
    _, original_ext = os.path.splitext(original_name)

    new_name = ""

    for thing in options:
        if thing in inputs.keys():
            input = inputs[thing]
            # If this is a path, get the filename from list
            if isinstance(input, list):
                if index is not None:
                    # this is a multi-file / series output
                    input = input[index]
                else:
                    input = input[0]
                input = os.path.basename(input)
                input, _ = os.path.splitext(input)
            elif not isinstance(input, str):
                input = str(input)
            new_name += input.replace(" ", "_")
        elif thing == "date":
            new_name += datetime.today().strftime("%Y-%m-%d")
        elif thing == "timestamp":
            new_name += datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
        else:
            new_name += thing.replace(" ", "-")

        new_name += "-"

    # remove trailing -
    new_name = new_name[:-1]

    # if this is a multi-file output, then append the index to the renamed file output
    if index is not None:
        new_name += f".{index}"

    new_name += original_ext

    return new_name


def unique_id() -> str:
    """Generate a unique identifier

    This will be different every time you call it.

    """
    return str(uuid.uuid4())


@beartype
def _check_in(thing: str, allowed_values: List[str]) -> None:
    """checks that a string is in a list of strings, and if
    not, print a pretty error"""

    if thing not in allowed_values:
        raise ValueError(f"{thing} is not a member of {allowed_values}")


@beartype
def _sort_isxd_files_by_start_time(
    input_files: List[str],
) -> List:
    """Sort isxd files by their start time.
    :param input_files: list of isxd file paths
    :return: sorted list of isxd file paths
    """
    start_times = []
    for file in input_files:
        isxd_metadata = _extract_footer(file)

        if isxd_metadata["type"] == 5:
            # isxd events
            start_time = (
                isxd_metadata["global times"][0]["secsSinceEpoch"]["num"]
                / isxd_metadata["global times"][0]["secsSinceEpoch"]["den"]
            )
        else:
            # other isxd files
            start_time = (
                isxd_metadata["timingInfo"]["start"]["secsSinceEpoch"]["num"]
                / isxd_metadata["timingInfo"]["start"]["secsSinceEpoch"]["den"]
            )
        start_times.append(start_time)

    sorted_indices = np.argsort(start_times)
    sorted_files = np.array(input_files)[sorted_indices]
    return sorted_files.tolist()


@beartype
def parse_params(
    *,
    inputs_file: str,
    params: List[dict],
) -> dict:
    """given some location to a inputs.sh file, parse
    parameters and return a dictionary with inputs

    params is from the toolbox_info.json for the tool of
    interest

    """

    if not os.path.exists(inputs_file):
        raise FileNotFoundError(f"{inputs_file} not found")

    inputs = ConfigObj(inputs_file)

    data = dict()
    for param in params:
        key = param["key"]
        value = None

        param_type = param["type"]["param_type"]

        if key in inputs.keys():
            if param_type == "ToolPathParam":
                if param["type"]["multiple"]:
                    value = ast.literal_eval(inputs[key])
                else:
                    # this is a single path, so unpack the list
                    # and return just the first item
                    value = ast.literal_eval(inputs[key])[0]

            elif param_type == "ToolStringParam":
                value = inputs[key]

            elif param_type == "ToolFloatRangeParam":
                if key in inputs.keys():
                    value = float(inputs[key])

            elif param_type == "ToolIntRangeParam":
                value = int(inputs[key])

            elif param_type == "ToolBooleanParam":
                value = bool(inputs[key])

            else:
                raise Exception(
                    f"param_type {param['type']['param_type']} not recognized"
                )

        else:
            print(f"{key} not found in inputs.sh")
        if value is not None:
            data[key] = value
    return data


def sh_2_json(sh_file: str, json_file: str):
    """
    Convert a .sh (Shell script) file to a .json file.

    Parameters:
    sh_file (str): Path to the input .sh file.
    json_file (str): Path to the output .json file.

    Returns:
    None
    """
    config = ConfigObj(sh_file)

    for key in config.keys():
        if isinstance(config[key], list):
            continue

        elif config[key].startswith("[") and config[key].endswith("]"):
            # this is a ToolFilePathParam
            config[key] = ast.literal_eval(config[key])
            continue

        # handle booleans
        elif config[key].lower() == "true":
            config[key] = True
            continue
        elif config[key].lower() == "false":
            config[key] = False
            continue

        # try to handle numeric data types
        try:
            config[key] = int(config[key])
        except Exception:
            try:
                config[key] = float(config[key])
            except Exception:
                pass

    with open(json_file, "w") as f:
        json.dump(config, f, indent=4)


def json_2_sh(json_file: str, sh_file: str):
    """
    Convert a .json file to a .sh (Shell script) file.

    Parameters:
    json_file (str): Path to the input .json file.
    sh_file (str): Path to the output .sh file.

    Returns:
    None
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    with open(sh_file, "w") as f:
        for key, value in data.items():
            if isinstance(value, list):
                string = f"{key}='["
                if len(value) == 1:
                    string += f'"{value[0]}"'
                else:
                    for i in value:
                        string += f'"{i}",'
                    string = string[:-1]
                string += "]'\n"
                f.write(string)
            else:
                f.write(f"{key}='{value}'\n")


@beartype
def _add_suffix(names: List, suffix: str) -> List:
    """adds a suffix to a list of filenames, useful for
    interacting with the isx API

    ### Arguments:

    - names: list of string arrays
    - suffix: suffix to add

    ### Returns:

    - list of names with suffix

    """

    names = names.copy()

    for i, thing in enumerate(names):
        name, ext = os.path.splitext(thing)
        names[i] = name + suffix + ext

    return names


@beartype
def _trim_movie(
    movie_file: str,
    *,
    start: int,
    stop: int,
) -> str:
    """trim a movie to some range

    This function exists because I cannot figure out how to
    do this in IDPS.

    ### Arguments

    - movie_file: path to movie to trim
    - start: frame to start from
    - stop: frame to stop at

    ### Returns:

    name of trimmed movie file

    """
    movie = isx.Movie.read(movie_file)

    clean_movie_file = _add_suffix([movie_file], "-trimmed")[0]

    if os.path.exists(clean_movie_file):
        return clean_movie_file

    timing = isx.Timing(
        num_samples=stop - start,
        period=movie.timing.period,
        start=movie.timing.start,
    )

    movie_out = isx.Movie.write(
        clean_movie_file, timing, movie.spacing, movie.data_type
    )

    for i in np.arange(start, stop):
        frame = movie.get_frame_data(i)
        movie_out.set_frame_data(i - start, frame)

    movie_out.flush()

    return clean_movie_file


@beartype
def check_file_extention_is(
    file_name: str,
    *,
    ext: str = ".isxd",
):
    """small util func to check the extention of a file
    and fail otherwise"""

    _, ext_ = os.path.splitext(os.path.basename(file_name))
    if ext_.lower() != ext:
        raise Exception(
            f"{file_name} does not have the extension: {ext}. It instead has {ext_}"
        )


@beartype
def _read_type_annotations(lines: List[str], arg: ast.arg) -> (None, str):
    """small utility to read out the type annotation as a
    string, if annotation is present"""

    if arg.annotation is None:
        return None

    a = arg.annotation.col_offset
    z = arg.annotation.end_col_offset

    # ast returns 1-indexed line numbers,
    # python assumes 0-indexed
    lineno = arg.annotation.lineno - 1
    type_annotation = lines[lineno][a:z]

    return type_annotation


@beartype
def str_to_functions_and_arguments(data: str) -> Tuple[Dict, Dict]:
    """
    Parse a string containing Python code using ast and extract all function definitions, their arguments, and their doc strings.
    Parameters:
    data (str): A string containing Python code.

    Returns:
    dict: A dictionary where the keys are function names and the values are  a dictionary containing "Args", a list of dictionaries representing
          the function's arguments and . Each argument dictionary contains the keys 'name', 'default', and 'type_annotation'.
          and "Doc", the function's doc string.
    """

    # get a list of all expressions in the python file
    m = ast.parse(data)
    exprs = m.body

    lines = data.split("\n")

    # figure out which expressions are func defs
    func_exprs = []
    for expr in exprs:
        if isinstance(expr, ast.FunctionDef):
            func_exprs.append(expr)

    # get names of arguments for each function
    functions = dict()
    docs = dict()
    for func in func_exprs:
        arguments = []

        # we only support default values for kw only arguments
        for arg, kw_defaults in zip(
            func.args.kwonlyargs, func.args.kw_defaults
        ):
            type_annotation = _read_type_annotations(lines, arg)

            argument = dict(
                name=arg.arg,
                default=None,
                type_annotation=type_annotation,
            )

            if kw_defaults is not None and not isinstance(
                kw_defaults,
                (
                    ast.Tuple,
                    ast.List,
                ),
            ):
                argument["default"] = kw_defaults.value

            arguments.append(argument)

        for arg in func.args.args:
            type_annotation = _read_type_annotations(lines, arg)

            argument = dict(
                name=arg.arg,
                default=None,
                type_annotation=type_annotation,
            )

            arguments.append(argument)

        functions[func.name] = arguments
        docs[func.name] = ast.get_docstring(func)

    return functions, docs


def output_manifest_to_toolbox_info(
    output_manifest_file_loc: str,
) -> dict:
    """strips out information from an output manifest file that
    we don't need in the toolbox info, and generates a dict
    that can be inserted directly into the results section of
    a toolbox info

    limitations:

    - this supports only a subset of what is possible in IDEAS
    - specifically, "required": true,
                    "multiple": false,
    - is always set for every group, object and file

    """

    if not os.path.exists(output_manifest_file_loc):
        raise RuntimeError("File not found")

    with open(output_manifest_file_loc, "r") as f:
        groups = json.load(f)

    # this is a list of groups, let's unpack it
    # assuming that we have just one group
    assert (
        len(groups) == 1
    ), "This code assumes that we have only 1 group, and therefore won't work here"

    group = groups[0]

    # remove keys we don't need
    group.pop("group_id", None)

    objects = group["group_data"]

    # remove source objects
    non_source_objects = []
    for obj in objects:
        if obj["object_data"][0]["file_category"] != "source":
            non_source_objects.append(obj)

    # append some things to the non-source objs
    for obj in non_source_objects:
        obj["required"] = True
        obj["multiple"] = False

        for thing in obj["object_data"]:
            thing["file_formats"] = [thing["file_format"]]
            thing.pop("file_format", None)

            thing.pop("parent_ids", None)

            # TODO add help text here dynamically
            thing["help"] = "help text placeholder"

            # TODO create hooks for this
            thing["required"] = True
            thing["multiple"] = False

            # replace the preview field
            # NOTE: this assumes that all preview files
            # are always generated, and we do not (yet)
            # support outputing a variable number of
            # preview files
            if "preview" in thing.keys():
                n_preview_files = len(thing["preview"])

                # TODO make this work for multiple file formats
                file_formats = [thing["preview"][0]["file_format"]]
                thing.pop("preview", None)
                thing["preview"] = dict(
                    file_formats=file_formats, num_files=[n_preview_files]
                )

    group["group_data"] = non_source_objects

    rm_keys = [
        "file_id",
        "file_path",
        "file_type",
        "file_structure",
        "file_category",
    ]

    for obj in group["group_data"]:
        obj.pop("object_id", None)
        for key in rm_keys:
            obj["object_data"][0].pop(key, None)

    # TODO customize this with tool name
    group["group_name"] = "Workflow Output"
    group["group_type"] = "tool_output"
    group["required"] = True
    group["multiple"] = False

    # put it back in a list
    groups = [group]

    return groups


@beartype
def subsample(x: NumpyFloatVector, bin_size: int = 50) -> NumpyFloatVector:
    """
    min-max resampler to make plots smaller
    and to prevent the heat death of your graphics card
    this destroys and alters data, and should only be used
    for plots!
    Args:
        x: a vector
        bin_size: max and min will be computed on this bin size (default 50)
    """

    vector_end = np.floor(x.shape[0] / bin_size).astype(int) * bin_size
    x = x[0:vector_end]
    nbins = int(len(x) / bin_size)

    reshaped_x = np.reshape(x, (nbins, bin_size))
    tops = reshaped_x.max(axis=1)
    bottoms = reshaped_x.min(axis=1)

    reduced_x = np.zeros(tops.shape[0] * 2)
    reduced_x[1::2] = tops
    reduced_x[0::2] = bottoms

    return reduced_x


def remove_files(files: List) -> None:
    """Remove files from disk.
    :param files: list of file paths to remove
    """
    for f in files:
        if os.path.exists(f):
            os.remove(f)


def get_file_size(file_path: str) -> str:
    """Compute and format the size of a file on disk.

    :param file_path: path to the input file
    """
    file_size = abs(os.path.getsize(file_path))
    for unit_prefix in ["", "K", "M", "G", "T", "P", "E", "Z", "Y"]:
        if file_size < 1024.0:
            return f"{file_size:.2f} {unit_prefix}B"
        file_size /= 1024.0
    return f"{os.path.getsize(file_path)} B"


def _set_up_logger():
    """Set up logger for IDEAS


    this code moved from old-style IDEAS toolboxes.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.addFilter(lambda r: r.levelno <= logging.ERROR)
    stdout_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)


def get_num_cells_by_status(cellset_filename: str):
    """Count the number of cells for each cell status"""
    cell_set = isx.CellSet.read(cellset_filename)

    num_accepted_cells = 0
    num_undecided_cells = 0
    num_rejected_cells = 0

    for index in range(cell_set.num_cells):
        cell_status = cell_set.get_cell_status(index)
        if cell_status == "accepted":
            num_accepted_cells += 1
        elif cell_status == "undecided":
            num_undecided_cells += 1
        elif cell_status == "rejected":
            num_rejected_cells += 1

    return num_accepted_cells, num_undecided_cells, num_rejected_cells

def compute_end_time(timing_info):
    """Compute end time of an isxd file.
    :param isxd_timing_info: json object containing isxd timing information
    :return: end time json object of the form
             {
                "secsSinceEpoch": {"num": 123456, "den": 1000}
                "utcOffset": 0
             }
    """
    # extract start time and period contained in the timing info object
    start_time = (
        timing_info["start"]["secsSinceEpoch"]["num"]
        / timing_info["start"]["secsSinceEpoch"]["den"]
    )
    period = timing_info["period"]["num"] / timing_info["period"]["den"]

    # compute end time
    end_time = start_time + (period * timing_info["numTimes"])
    end_time_den = timing_info["start"]["secsSinceEpoch"]["den"]
    end_time_num = round(end_time * end_time_den)

    # return properly formatted end time
    return {
        "secsSinceEpoch": {"den": end_time_den, "num": end_time_num},
        "utcOffset": timing_info["start"]["utcOffset"],
    }

def validate_cellset_series_compatibility(input_files):
    """Validate that input files form a valid series.
    :param input_files: list of paths to isxd cell set files
    :return: True if files form a valid cell set series, False otherwise
    """
    # sort input files by their start time
    input_files = _sort_isxd_files_by_start_time(input_files)

    # extract metadata from first input file
    first_cell_set = isx.CellSet.read(input_files[0])
    first_cell_set_metadata = read_isxd_metadata(input_files[0])
    num_cells = first_cell_set.num_cells
    cell_statuses = [
        first_cell_set.get_cell_status(i) for i in range(num_cells)
    ]
    del first_cell_set

    # ensure cell set contains at least 1 cell
    if num_cells < 1:
        raise IdeasError(
            "No cell found in '{0}'.".format(os.path.basename(input_files[0])),
        )

    # initialize series timing and spacing info to those of first cell set
    series_spacing_info = first_cell_set_metadata["spacingInfo"]
    series_timing_info = first_cell_set_metadata["timingInfo"]
    series_timing_info["end"] = compute_end_time(
        first_cell_set_metadata["timingInfo"]
    )

    # loop over all other cell sets and ensure they can be combined into a series
    for f in input_files[1:]:
        # read isxd cell set and corresponding metadata
        isxd_cell_set = isx.CellSet.read(f)
        isxd_metadata = read_isxd_metadata(f)

        # ensure number of cells match across input files
        if isxd_cell_set.num_cells != num_cells:
            raise IdeasError(
                "The number of cells differ across the input files",
            )

        # ensure cell statuses match across input files
        isxd_cell_statuses = [
            isxd_cell_set.get_cell_status(i)
            for i in range(isxd_cell_set.num_cells)
        ]
        if isxd_cell_statuses != cell_statuses:
            raise IdeasError(
                "Individual cell statuses differ across the input files",
            )

        # verify SPACING info
        # validate that the movies have the same spatial dimensions
        if isxd_metadata["spacingInfo"] != series_spacing_info:
            raise IdeasError(
                "All input movies must have the same spatial dimensions",
            )

        # verify TIMING info
        # ensure current item does not temporally overlap with any other item
        item_start_time = (
            isxd_metadata["timingInfo"]["start"]["secsSinceEpoch"]["num"]
            / isxd_metadata["timingInfo"]["start"]["secsSinceEpoch"]["den"]
        )

        series_end_time = (
            series_timing_info["end"]["secsSinceEpoch"]["num"]
            / series_timing_info["end"]["secsSinceEpoch"]["den"]
        )

        if item_start_time <= series_end_time:
            raise IdeasError(
                "Items that temporally overlap cannot be combined into a series.",
            )

        # validate that the period is nearly identical for all input files
        # TODO

        # validate that all input files share the same microscope identifier
        # TODO: if not, log this as a warning but allow processing to continue
