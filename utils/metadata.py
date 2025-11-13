import os
import isx
import json
from ideas.exceptions import IdeasError
import logging

logger = logging.getLogger()


def read_isxc_metadata(input_filename):
    """Read the metadata of an isxc file as a json-formatted dictionary.

    :param input_filename: path to the input file (.isxc)
    :return: metadata of the isxc file
    """
    _, file_extension = os.path.splitext(input_filename)
    if file_extension.lower() != ".isxc":
        raise IdeasError("Metadata can only be extracted from isxc files")

    try:
        with open(input_filename, "rb") as f:
            # the number of bytes used to represent the variable size_t in c++
            sizeof_size_t = 8

            # location of the frame count in the header (8th field)
            frame_count_location = sizeof_size_t * 7

            # need the size of data descriptors to get to the session offset
            isx_comp_desc_offset = 32

            # multiplied by 2 since there are 2 descriptors (frame and meta data)
            session_offset_location = (
                frame_count_location + isx_comp_desc_offset * 2
            )

            # extract the session offset from the header to get location of the session data
            f.seek(session_offset_location, 0)
            session_offset = int.from_bytes(
                f.read(sizeof_size_t), byteorder="little"
            )

            # move reader to the session data
            f.seek(session_offset, 0)

            # read the session data and decode the string
            session = str(f.read().decode("utf-8"))

            # convert the string into a json file format
            session_json = json.loads(session)
            return session_json
    except Exception as e:
        raise IdeasError(
            "The isxc file metadata cannot be read, it may be missing or corrupted."
        ) from e


def read_isxd_metadata(input_filename):
    """Read the metadata of an isxd file as a json-formatted dictionary.

    :param input_filename: path to the input file (.isxd)
    :return: metadata of the isxd file
    """
    _, file_extension = os.path.splitext(input_filename)
    if file_extension.lower() != ".isxd":
        raise IdeasError("Metadata can only be extracted from isxd files")

    try:
        with open(input_filename, "rb") as f:
            footer_size_offset = 8
            f.seek(-footer_size_offset, 2)
            footer_size = int.from_bytes(
                f.read(footer_size_offset), byteorder="little"
            )
            offset = footer_size + footer_size_offset + 1
            f.seek(-offset, 2)
            metadata = json.loads(f.read(footer_size))
            return metadata
    except Exception as e:
        # error message from IDPS: "Error while seeking to beginning of JSON header at end"
        raise IdeasError(
            "The isxd file metadata cannot be read, it may be missing or corrupted. "
            "File recovery may help recover the data and save it into a new isxd file."
        ) from e


def is_multicolor(metadata, check_interleaved=True):
    """Check if an isxd file is from a multicolor recording

    :param metadata: dictionary containing metadata of isxd file
    :param check_interleaved: if False, only checks if a file originates from
    a multicolor movie and does not check if it has been deinterleaved. If True,
    an additional check is done to ensure that the movie has not been deinterleaved.

    :return: if check_deinterleaved=False: True for multicolor movies and any files derived
    from multicolor movies (e.g. processed movies, cell sets, event set, etc.)
    if check_deinterleaved=True: True only for multicolor movies that have not been deinterleaved
    """
    if (
        "extraProperties" not in metadata
        or metadata["extraProperties"] is None
    ):
        logger.warn(
            "Unable to determine whether the input is a multicolor movie"
        )
        return False

    # return False if we cannot determine if the file is multicolor
    if metadata["extraProperties"] is None:
        return False

    try:
        # check if the movie has the metadata originating from a multicolor movie
        # this does not determine if a movie originates from a multicolor movie
        # but has been deinterleaved
        has_multicolor_metadata = (
            "dualColor" in metadata["extraProperties"]["microscope"]
            and "enabled"
            in metadata["extraProperties"]["microscope"]["dualColor"]
            and metadata["extraProperties"]["microscope"]["dualColor"]["mode"]
            == "multiplexing"
        )

        # check if file has been deinterleaved
        # IDPS adds metadata fields once a movie is deinterleaved, so we can check for that
        if has_multicolor_metadata and check_interleaved:
            is_deinterleaved = (
                "idps" in metadata["extraProperties"]
                and "channel" in metadata["extraProperties"]["idps"]
            )
            return not is_deinterleaved
        return has_multicolor_metadata
    except KeyError:
        return False


def is_multiplane(metadata, check_interleaved=True):
    """Check if an isxd file is from a multiplane recording

    :param metadata: dictionary containing metadata of isxd file
    :param check_interleaved: if False, only checks if a file originates from
    a multiplane movie and does not check if it has been deinterleaved. If True,
    an additional check is done to ensure that the movie has not been deinterleaved.

    :return: if check_deinterleaved=False: True for multiplane movies and any files derived
    from multiplane movies (e.g. processed movies, cell sets, event set, etc.)
    if check_deinterleaved=True: True only for multiplane movies that have not been deinterleaved
    """
    if is_multicolor(metadata):
        return False

    # return False if we cannot determine if the file is multiplane
    if (
        "extraProperties" not in metadata
        or metadata["extraProperties"] is None
    ):
        logger.warn(
            "Unable to determine whether the input is a multiplane movie"
        )
        return False

    has_multiplane_metadata = False
    # check if metadata contains fields originating from multiplane movie
    # this does not determine if a movie originates from a multiplane movie
    # but has been deinterleaved
    try:
        for w in metadata["extraProperties"]["auto"]["waveforms"]:
            if "name" in w and "efocus" in w["name"]:
                has_multiplane_metadata = True
                break
    except KeyError:
        return False

    # check if movie has been deinterleaved
    # IDPS adds metadata fields once a movie is deinterleaved, so we can check for that
    if has_multiplane_metadata and check_interleaved:
        is_deinterleaved = (
            "idps" in metadata["extraProperties"]
            and "efocus" in metadata["extraProperties"]["idps"]
        )
        return not is_deinterleaved

    return has_multiplane_metadata


def get_multiplane_efocus_vals(metadata):
    """Retrieve efocus values from multiplane isxd file metadata

    :param metadata: dictionary containing metadata of multiplane isxd file

    :return: set containing unique efocus values
    """
    # only need unique efocus values
    efocus_vals = set()

    # using initial values from isxPreprocessMovie.cpp getEfocusFromMovieJsonHeader() in IDPS
    v_max = 100
    max_amplitude = 892
    try:
        # get efocus values
        for waveform in metadata["extraProperties"]["auto"]["waveforms"]:
            if (
                "name" in waveform
                and isinstance(waveform["name"], str)
                and "efocus" in waveform["name"]
            ):
                # IDAS > 1.2.1
                if len(waveform["data"]) == 1:
                    for vertices in waveform["data"][0]["vertices"]:
                        efocus_vals.add(vertices["y"])
                # for IDAS <= 1.2.0
                else:
                    # get efocus sensor configurations
                    if metadata["extraProperties"]["adMode"] == "auto":
                        for triggered in metadata["extraProperties"]["auto"][
                            "triggered"
                        ]:
                            if "efocus" in triggered["destination"]["device"]:
                                v_max = triggered["destination"]["vMax"]
                                max_amplitude = triggered["destination"][
                                    "maxAmplitude"
                                ]
                    # get efocus values
                    for data in waveform["data"]:
                        if isinstance(data, str):
                            # need to filter relevant fields. The first 16 bits are used
                            # as an identifier, so right shifting by 16 allows us to see
                            # the identifier only - needs to match 0x3200
                            if int(data, 16) >> 16 == 0x3200:
                                efocus = (
                                    (int(data, 16) & 0xFFFF)
                                    * v_max
                                    / max_amplitude
                                )
                                efocus_vals.add(int(efocus))
        return sorted(list(efocus_vals))
    except KeyError as ke:
        raise IdeasError(
            f"The file metadata is missing required fields: {ke.args}"
        ) from ke


def get_multicolor_efocus_vals(metadata):
    """Retrieve efocus values from multicolor isxd file

    :param metadata: dictionary containing metadata of multicolor isxd file

    :return: set containing unique efocus values
    """
    mode = metadata["extraProperties"]["microscope"]["dualColor"]["mode"]
    green_efocus = metadata["extraProperties"]["microscope"]["dualColor"][
        mode
    ]["green"]["focus"]
    red_efocus = metadata["extraProperties"]["microscope"]["dualColor"][mode][
        "red"
    ]["focus"]
    return [green_efocus, red_efocus]


def get_efocus(input_filename, tmp_dir="/tmp"):
    """Return the efocus value associated with the input ISXD/IMU/GPIO file.
    For multiplane and multicolor data, the list of efocus values will be returned.

    :param input_filename: path to the input file
    :return: efocus value
    """
    # parse input file name
    file_basename, file_extension = os.path.splitext(
        os.path.basename(input_filename)
    )
    file_extension = file_extension.lower()

    # extract file metadata
    if file_extension == ".gpio":
        isx.export_gpio_to_isxd(input_filename, tmp_dir)
        tmp_gpio_isxd_filename = os.path.join(
            tmp_dir, file_basename + "_gpio.isxd"
        )
        metadata = read_isxd_metadata(tmp_gpio_isxd_filename)
        os.remove(tmp_gpio_isxd_filename)
    elif file_extension == ".imu":
        isx.export_gpio_to_isxd(input_filename, tmp_dir)
        tmp_imu_isxd_filename = os.path.join(
            tmp_dir, file_basename + "_imu.isxd"
        )
        metadata = read_isxd_metadata(tmp_imu_isxd_filename)
        os.remove(tmp_imu_isxd_filename)
    elif file_extension == ".isxc":
        metadata = read_isxc_metadata(input_filename)
    elif file_extension == ".isxd":
        metadata = read_isxd_metadata(input_filename)
    else:
        logger.warning(
            f"The efocus value cannot be extracted from a file with file extension '{file_extension}'"
        )
        return None

    # get efocus value from IDPS metadata
    try:
        efocus = metadata["extraProperties"]["idps"]["efocus"]
        if efocus not in [None, ""]:
            return efocus
    except (KeyError, TypeError):
        pass

    # if movie has multiple planes, retrieve all efocus values
    try:
        if is_multiplane(metadata, check_interleaved=True):
            return get_multiplane_efocus_vals(metadata)
        elif is_multicolor(metadata, check_interleaved=True):
            return get_multicolor_efocus_vals(metadata)
    except Exception:
        logger.warning(
            "Could not retrieve multiplane/multicolor efocus values"
        )

    # get efocus value from IDAS metadata
    try:
        efocus = metadata["extraProperties"]["microscope"]["focus"]
        if efocus not in [None, ""]:
            return efocus
    except (KeyError, TypeError):
        pass

    return None
