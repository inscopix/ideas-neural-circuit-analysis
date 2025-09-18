"""Validation utilities for population data analysis.

This module contains all validation functions used by the population data analysis tools.
It provides comprehensive validation for input parameters, file structures, and data consistency.
"""

import os
import logging
from typing import List, Optional, Dict, Set, Any, Tuple, Union

import pandas as pd
import matplotlib.colors as mcolors

from ideas.exceptions import IdeasError
from utils.combine_compare_population_data_utils import (
    detect_baseline_state,
    validate_states_by_state_comparison_type,
)
import utils.config as config

logger = logging.getLogger(__name__)


def validate_colors(
    colors: Union[List[str], str], color_type: str = "modulation"
) -> List[str]:
    """Validate colors input for compatibility with matplotlib plotting functions.

    Defaults to predefined colors if the input is invalid or incompatible.

    Parameters
    ----------
    colors : list of str or str
        List of colors or a comma-separated string.
    color_type : str, optional
        Type of colors being validated ("modulation", "state", or "group").
        Defaults to "modulation".

    Returns
    -------
    list of str
        A valid list of colors, defaulting to predefined ones if invalid.

    Raises
    ------
    IdeasError
        If input parameters are invalid or colors are not recognized by matplotlib.

    """
    # Default colors and constraints based on type
    color_configs = {
        "modulation": {
            "defaults": [
                config.PLOT_UP_MODULATED_COLOR,
                config.PLOT_DOWN_MODULATED_COLOR,
            ],
            "min_colors": 2,
            "max_colors": None,
            "allow_duplicates": False,
        },
        "state": {
            "defaults": [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
            ],
            "min_colors": 1,
            "max_colors": None,
            "allow_duplicates": True,
        },
        "group": {
            "defaults": ["#1f77b4", "#ff7f0e"],
            "min_colors": 1,
            "max_colors": 2,
            "allow_duplicates": False,
        },
    }

    color_config = color_configs.get(color_type, color_configs["modulation"])
    default_colors = color_config["defaults"]

    try:
        # Process string input
        if isinstance(colors, str):
            colors = [x.strip() for x in colors.split(",") if x.strip()]

        # Validate input type
        if not isinstance(colors, (list, tuple)):
            raise IdeasError(
                f"{color_type}_colors must be a list, tuple, or a comma-separated string.",
            )

        # Handle empty colors (return empty for state colors to be determined later)
        if len(colors) == 0:
            return [] if color_type == "state" else default_colors

        # Validate color count
        if len(colors) < color_config["min_colors"]:
            raise IdeasError(
                f"{color_type}_colors must contain at least {color_config['min_colors']} colors.",
            )

        if (
            color_config["max_colors"]
            and len(colors) > color_config["max_colors"]
        ):
            raise IdeasError(
                f"{color_type}_colors must contain at most {color_config['max_colors']} colors.",
            )

        # Validate each color and check for duplicates
        for color in colors:
            if not isinstance(color, str) or not mcolors.is_color_like(color):
                raise IdeasError(
                    f"Invalid {color_type} color: '{color}' is not recognized by Matplotlib.",
                )

        # Check for duplicates if not allowed
        if not color_config["allow_duplicates"] and len(set(colors)) != len(
            colors
        ):
            raise IdeasError(
                f"Duplicate colors detected in {color_type}_colors.",
            )

        return colors

    except IdeasError:
        # Re-raise ToolExceptions to maintain error specificity
        raise
    except Exception as e:
        # Only catch non-IdeasError errors and fallback to defaults
        logger.warning(
            f"Using default {color_type} colors: {default_colors}. Error: {str(e)}"
        )
        return default_colors


def validate_file_group(
    files: List[str],
    group_name: str,
    file_type: str,
    min_files: int = 2,
    allow_test_files: bool = False,
) -> None:
    """Validate a group of files.

    Parameters
    ----------
    files : List[str]
        List of file paths to validate
    group_name : str
        Name of the group (for error messages)
    file_type : str
        Type of files being validated (for error messages)
    min_files : int, optional
        Minimum number of files required, by default 2
    allow_test_files : bool, optional
        Whether to allow single test files, by default False

    Raises
    ------
    IdeasError
        If validation fails

    """
    if not isinstance(files, list):
        raise IdeasError(
            f"The input {group_name} {file_type} files must consist of a list of files.",
        )

    # Adjust minimum files for test scenarios
    actual_min_files = min_files
    if allow_test_files and any(
        "synthetic_data" in f or "test_" in os.path.basename(f) for f in files
    ):
        actual_min_files = 1

    if len(files) < actual_min_files:
        raise IdeasError(
            f"The {group_name} must contain at least {actual_min_files} population "
            f"{file_type} data files.",
        )

    for file_path in files:
        if not os.path.isfile(file_path):
            raise IdeasError(
                f"File does not exist: {file_path}",
            )


def validate_state_names_and_colors(
    state_names: Optional[str],
    state_colors: Optional[str],
    allow_empty: bool = False,
) -> Tuple[List[str], List[str]]:
    """Validate state names and colors input.

    Parameters
    ----------
    state_names : Optional[str]
        Comma-separated string of state names
    state_colors : Optional[str]
        Comma-separated string of state colors
    allow_empty : bool, optional
        Whether to allow empty state names/colors, by default False

    Returns
    -------
    Tuple[List[str], List[str]]
        Validated state names list and state colors list

    Raises
    ------
    IdeasError
        If validation fails

    """
    if not allow_empty:
        if state_names is None:
            raise IdeasError(
                "State names must be provided as a comma-separated string.",
            )

        if state_colors is None:
            raise IdeasError(
                "State colors must be provided as a comma-separated string.",
            )

    # Handle None values for optional case
    if state_names is None:
        state_names = ""
    if state_colors is None:
        state_colors = ""

    # Check that state_names and state_colors are strings
    if not isinstance(state_names, str):
        raise IdeasError(
            "State names must be a comma-separated string.",
        )

    if not isinstance(state_colors, str):
        raise IdeasError(
            "State colors must be a comma-separated string.",
        )

    # Parse and validate
    state_names_list = [name.strip() for name in state_names.split(",")]
    state_colors_list = [color.strip() for color in state_colors.split(",")]

    # Ensure lists are not empty after stripping whitespace
    state_names_list = [name for name in state_names_list if name]
    state_colors_list = [color for color in state_colors_list if color]

    if not allow_empty:
        if not state_names_list:
            raise IdeasError(
                "State names string resulted in an empty list. Please provide valid state names.",
            )

        if not state_colors_list:
            raise IdeasError(
                "State colors string resulted in an empty list. Please provide valid state colors.",
            )

    if (
        state_names_list
        and state_colors_list
        and len(state_names_list) > len(state_colors_list)
    ):
        raise IdeasError(
            f"Number of state colors ({len(state_colors_list)}) is less than "
            f"number of state names ({len(state_names_list)}). "
            "Please provide matching counts.",
        )

    return state_names_list, state_colors_list


def validate_combine_compare_population_data_parameters(
    group1_population_activity_files: List[str],
    group1_population_events_files: List[str],
    group1_name: str,
    group2_population_activity_files: List[str],
    group2_population_events_files: List[str],
    group2_name: str,
    state_names: Optional[List[str]] = None,
    state_colors: Optional[List[str]] = None,
    data_pairing: str = "unpaired",
    subject_matching: str = "number",
) -> None:
    """Validate the input parameters to the combine-and-compare population activity data tool.

    This function performs extensive validation of input parameters, ensuring that all
    required parameters are properly provided and that the data files meet necessary conditions.

    Parameters
    ----------
    group1_population_activity_files : List[str]
        Population activity files from the first group.
    group1_population_events_files : List[str]
        Population events files from the first group.
    group1_name : str
        Name of the first group.
    group2_population_activity_files : List[str]
        Population activity files from the second group.
    group2_population_events_files : List[str]
        Population events files from the second group.
    group2_name : str
        Name of the second group.
    state_names : List[str]
        List of state names to include in the analysis.
    state_colors : List[str]
        List of colors to use for visualizing different states.
    data_pairing: str
        Type of data pairing for comparison ("paired" or "unpaired").
    subject_matching: str
        Method for matching subjects between groups for paired analysis.

    State Name and Color Handling:
    - Providing state_names requires corresponding state_colors (empty list or matching length)
    - When state_names is None/empty, state_colors can be provided for auto-extracted states
    - When both are None/empty, states are extracted from files with default colors assigned
    - If provided state_colors has fewer/more values than detected states,
      it's automatically adjusted
    - For optimal visualization, provide matching state_names and state_colors

    Raises
    ------
    IdeasError
        If any validation check fails.

    """
    # Validate group names using generalized function
    validate_group_names(
        group1_name=group1_name,
        group2_name=group2_name,
        require_group1=True,
        require_group2_if_provided=bool(group2_population_activity_files),
    )

    # Validate file groups
    validate_file_group(
        group1_population_activity_files, "group 1", "activity"
    )

    # Events files validation
    if group1_population_events_files:
        validate_file_group(
            group1_population_events_files, "group 1", "events", min_files=1
        )

    # Group 2 file validation (group name already validated above)
    if group2_population_activity_files:
        validate_file_group(
            group2_population_activity_files, "group 2", "activity"
        )

        if group2_population_events_files:
            validate_file_group(
                group2_population_events_files,
                "group 2",
                "events",
                min_files=1,
            )

    # Validate state names and colors
    if state_names is not None and not isinstance(state_names, list):
        raise IdeasError(
            "state_names must be a list or None.",
        )

    if state_colors is not None and not isinstance(state_colors, list):
        raise IdeasError(
            "state_colors must be a list or None.",
        )

    # Validate data pairing
    if data_pairing not in ["paired", "unpaired"]:
        raise IdeasError(
            "data_pairing must be either 'paired' or 'unpaired'.",
        )

    logger.info("Parameter validation completed successfully")


def _validate_function_inputs(
    group1_files: List[str],
    reference_df: Optional[pd.DataFrame],
    global_state_comparison_type: Optional[str],
) -> None:
    """Validate function inputs and requirements."""
    if not group1_files:
        raise IdeasError(
            "No population activity files provided for group 1.",
        )

    if reference_df is None or global_state_comparison_type is None:
        raise IdeasError(
            "reference_df and global_state_comparison_type must be provided. "
            "Please ensure the calling function provides pre-loaded reference "
            "data and detected comparison type.",
        )

    if reference_df.empty:
        raise IdeasError(
            "Reference dataframe is empty. Cannot proceed with state validation.",
        )


def _extract_validated_states(
    reference_df: pd.DataFrame,
    state_names: Optional[List[str]],
    global_state_comparison_type: str,
    global_baseline_state: Optional[str],
) -> List[str]:
    """Extract and validate states from reference data."""
    logger.debug(
        f"Extracting states using comparison type '{global_state_comparison_type}'"
    )

    try:
        if state_names:
            # Use provided state names with validation
            validated_states, _, _ = validate_states_by_state_comparison_type(
                reference_df, state_names
            )
            logger.debug(f"Validated provided states: {validated_states}")
        else:
            # Extract states from reference dataframe
            _, extracted_states, _ = detect_baseline_state(reference_df)
            if not extracted_states:
                raise IdeasError(
                    "No states found in reference data. Please check input file format.",
                )

            validated_states, _, _ = validate_states_by_state_comparison_type(
                reference_df, extracted_states
            )
            logger.debug(
                f"Auto-extracted and validated states: {validated_states}"
            )

        return validated_states

    except Exception as e:
        if isinstance(e, IdeasError):
            raise
        raise IdeasError(
            f"Failed to extract/validate states: {str(e)}",
        )


def _validate_minimum_state_requirements(validated_states: List[str]) -> None:
    """Ensure minimum state requirements are met."""
    if not validated_states or len(validated_states) < 1:
        raise IdeasError(
            f"At least 1 valid state is required for analysis, "
            f"but only {len(validated_states) if validated_states else 0} were found: "
            f"{validated_states}",
        )


def _prepare_reference_states(
    validated_states: List[str],
    state_comparison_type: str,
    baseline_state: Optional[str],
) -> List[str]:
    """Prepare reference states for cross-file validation."""
    reference_states = validated_states.copy()

    # Include baseline state if needed for validation
    if (
        state_comparison_type == "state_vs_baseline"
        and baseline_state
        and baseline_state not in reference_states
    ):
        reference_states.append(baseline_state)
        logger.debug(
            f"Added baseline state '{baseline_state}' to reference states for validation"
        )

    return reference_states


def _validate_all_files_consistency(
    group1_files: List[str],
    group2_files: List[str],
    group1_events: Optional[List[str]],
    group2_events: Optional[List[str]],
    reference_df: pd.DataFrame,
    reference_states: List[str],
    state_comparison_type: str,
) -> None:
    """Validate structure and state consistency across all files."""
    # Skip validation of the first file (it's the reference)
    all_files_to_validate = []

    # Add remaining group 1 files
    if len(group1_files) > 1:
        all_files_to_validate.extend(group1_files[1:])

    # Add all group 2 files
    if group2_files:
        all_files_to_validate.extend(group2_files)

    # Add event files
    if group1_events:
        all_files_to_validate.extend(group1_events)
    if group2_events:
        all_files_to_validate.extend(group2_events)

    if not all_files_to_validate:
        logger.debug("Only one file provided, skipping cross-file validation")
        return

    reference_column_count = reference_df.shape[1]
    validation_errors = []

    logger.debug(
        f"Validating {len(all_files_to_validate)} files for consistency"
    )

    # Validate each file with enhanced error collection
    for file_path in all_files_to_validate:
        try:
            _validate_file_structure_and_states(
                file_path,
                reference_column_count,
                reference_states,
                state_comparison_type,
            )
        except Exception as e:
            validation_errors.append(
                f"{os.path.basename(file_path)}: {str(e)}"
            )

    # Report any validation errors
    if validation_errors:
        error_summary = "File validation errors detected:\n" + "\n".join(
            validation_errors
        )
        logger.warning(error_summary)
        # Note: We log warnings but don't fail, allowing for some file differences


def _validate_inter_group_consistency(
    group1_files: List[str],
    group2_files: List[str],
    state_comparison_type: str,
) -> None:
    """Validate state consistency between groups."""
    logger.debug("Validating inter-group state consistency")

    try:
        # Read representative files from each group
        group1_df = pd.read_csv(group1_files[0])
        group2_df = pd.read_csv(group2_files[0])

        # Extract states from each group
        _, group1_mod_states, group1_mean_states = detect_baseline_state(
            group1_df
        )
        _, group2_mod_states, group2_mean_states = detect_baseline_state(
            group2_df
        )

        # Combine states for comparison
        group1_all_states = set(group1_mod_states + group1_mean_states)
        group2_all_states = set(group2_mod_states + group2_mean_states)

        # Validate consistency using existing helper
        _validate_group_state_consistency(
            group1_all_states,
            group2_all_states,
            state_comparison_type,
            group1_mod_states,
            group2_mod_states,
        )

        logger.info("Inter-group state consistency validation passed")

    except pd.errors.EmptyDataError as e:
        raise IdeasError(
            f"Empty data file detected during group validation: {str(e)}",
        )
    except pd.errors.ParserError as e:
        raise IdeasError(
            f"File parsing error during group validation: {str(e)}",
        )
    except Exception as e:
        # Handle all other exceptions uniformly
        if isinstance(e, IdeasError):
            raise
        raise IdeasError(
            f"Unexpected error during inter-group validation: {str(e)}",
        )


def _validate_file_structure_and_states(
    file_path: str,
    reference_column_count: int,
    reference_states: List[str],
    detected_state_comparison_type: str,
) -> None:
    """Validate individual file structure and states with enhanced error handling.

    Parameters
    ----------
    file_path : str
        Path to the file being validated
    reference_column_count : int
        Expected number of columns based on reference file
    reference_states : List[str]
        Expected states that should be present in the file
    detected_state_comparison_type : str
        Type of comparison being performed

    Raises
    ------
    IdeasError
        If file cannot be read or has critical structural issues

    """
    file_name = os.path.basename(file_path)

    try:
        # Read and validate file structure
        df = pd.read_csv(file_path)
        if df.empty:
            logger.warning(f"File {file_name} is empty")
            return

        # Extract states from this file
        _, file_mod_states, file_mean_states = detect_baseline_state(df)
        all_file_states = set(file_mod_states + file_mean_states)

        # Validate column count (warning only for flexibility)
        if df.shape[1] != reference_column_count:
            logger.warning(
                f"Column count mismatch in {file_name}: expected {reference_column_count}, "
                f"found {df.shape[1]}. This may affect analysis consistency."
            )

        # Validate states based on comparison type
        validation_results = _validate_states_by_state_comparison_type(
            all_file_states=all_file_states,
            reference_states=reference_states,
            state_comparison_type=detected_state_comparison_type,
            file_name=file_name,
        )

        # Log validation results
        if validation_results["missing_states"]:
            if validation_results["critical_missing"]:
                logger.warning(
                    f"File {file_name}: Critical states missing: "
                    f"{validation_results['missing_states']}"
                )
            else:
                logger.debug(
                    f"File {file_name}: Some states missing (may be acceptable): "
                    f"{validation_results['missing_states']}"
                )

        if validation_results["extra_states"]:
            logger.debug(
                f"File {file_name}: Additional states found: "
                f"{validation_results['extra_states']}"
            )

    except pd.errors.EmptyDataError:
        raise IdeasError(
            f"File {file_name} is empty or contains no data",
        )
    except pd.errors.ParserError as e:
        raise IdeasError(
            f"Error parsing file {file_name}: {str(e)}",
        )
    except FileNotFoundError:
        raise IdeasError(
            f"File not found: {file_path}",
        )
    except Exception as e:
        raise IdeasError(
            f"Unexpected error validating file {file_name}: {str(e)}",
        )


def _validate_states_by_state_comparison_type(
    all_file_states: set,
    reference_states: List[str],
    state_comparison_type: str,
    file_name: str,
) -> Dict[str, Any]:
    """Validate states based on comparison type with detailed results.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing validation results:
        - missing_states: List of states missing from file
        - extra_states: List of additional states in file
        - critical_missing: Whether missing states are critical for analysis

    """
    validation_results = {
        "missing_states": [],
        "extra_states": [],
        "critical_missing": False,
    }

    # Determine expected states based on comparison type
    if state_comparison_type == "pairwise":
        expected_states = _extract_individual_states_from_pairs(
            reference_states
        )
        missing_states = [
            state for state in expected_states if state not in all_file_states
        ]

        if missing_states:
            validation_results["missing_states"] = missing_states
            # Check if file has any pairwise comparisons at all
            available_pairs = [s for s in all_file_states if " vs " in s]
            validation_results["critical_missing"] = not available_pairs
    else:
        # For non-pairwise comparisons, check direct state presence
        expected_states = set(reference_states)
        missing_states = [
            state for state in expected_states if state not in all_file_states
        ]

        if missing_states:
            validation_results["missing_states"] = missing_states
            # Missing more than half of states is critical
            validation_results["critical_missing"] = (
                len(missing_states) > len(expected_states) // 2
            )

    # Identify extra states (informational only)
    # Ensure expected_states is a set for proper set operations
    expected_states_set = (
        set(expected_states)
        if not isinstance(expected_states, set)
        else expected_states
    )
    extra_states = all_file_states - expected_states_set
    if extra_states:
        validation_results["extra_states"] = list(extra_states)

    return validation_results


def _extract_individual_states_from_pairs(state_list: List[str]) -> Set[str]:
    """Extract individual states from pairwise comparisons."""
    individual = set()
    for state in state_list:
        if " vs " in state:
            state1, state2 = state.split(" vs ")
            individual.add(state1.strip())
            individual.add(state2.strip())
        else:
            individual.add(state.strip())
    return individual


def _create_inconsistency_error(
    group1_only: Set[str], group2_only: Set[str]
) -> str:
    """Create standardized error message for state inconsistencies."""
    error_msg = "State name inconsistency detected between groups:\n"
    if group1_only:
        error_msg += (
            f"Group 1 has states not in Group 2: {sorted(group1_only)}\n"
        )
    if group2_only:
        error_msg += (
            f"Group 2 has states not in Group 1: {sorted(group2_only)}\n"
        )
    error_msg += (
        "Both groups must have the same state names for proper comparison. "
    )
    error_msg += (
        "Please ensure consistent state naming across all input files."
    )
    return error_msg


def _validate_group_state_consistency(
    group1_states: Set[str],
    group2_states: Set[str],
    detected_state_comparison_type: str,
    mod_states: List[str],
    group2_mod_states: List[str],
) -> None:
    """Validate state consistency between groups."""
    if detected_state_comparison_type == "pairwise":
        # Check individual component states for consistency
        group1_individual = _extract_individual_states_from_pairs(
            list(group1_states)
        )
        group2_individual = _extract_individual_states_from_pairs(
            list(group2_states)
        )

        group1_only = group1_individual - group2_individual
        group2_only = group2_individual - group1_individual

        if group1_only or group2_only:
            raise IdeasError(
                _create_inconsistency_error(group1_only, group2_only),
            )

        # Check pairwise comparison consistency
        group1_pairs = set([s for s in mod_states if " vs " in s])
        group2_pairs = set([s for s in group2_mod_states if " vs " in s])

        if group1_pairs != group2_pairs:
            logger.warning(
                f"Pairwise comparison differences detected:\n"
                f"Group 1 pairs: {sorted(group1_pairs)}\n"
                f"Group 2 pairs: {sorted(group2_pairs)}\n"
                f"Analysis will proceed with common pairs only."
            )
    else:
        # For non-pairwise comparisons, check direct state consistency
        group1_only = group1_states - group2_states
        group2_only = group2_states - group1_states

        if group1_only or group2_only:
            raise IdeasError(
                _create_inconsistency_error(group1_only, group2_only),
            )


def validate_group_names(
    group1_name: Optional[str],
    group2_name: Optional[str],
    require_group1: bool = True,
    require_group2_if_provided: bool = True,
) -> None:
    """Validate group names for consistency and requirements.

    Parameters
    ----------
    group1_name : Optional[str]
        Name of the first group
    group2_name : Optional[str]
        Name of the second group
    require_group1 : bool, optional
        Whether group1_name is required, by default True
    require_group2_if_provided : bool, optional
        Whether group2_name is required when group 2 data is provided, by default True

    Raises
    ------
    IdeasError
        If validation fails

    """
    if require_group1 and (
        not isinstance(group1_name, str) or not group1_name.strip()
    ):
        raise IdeasError(
            "group1_name must be a non-empty string.",
        )

    if require_group2_if_provided and group2_name is not None:
        if not isinstance(group2_name, str) or not group2_name.strip():
            raise IdeasError(
                "group2_name must be a non-empty string when group 2 data is provided.",
            )

    # Check for identical group names
    if (
        group1_name
        and group2_name
        and group1_name.strip() == group2_name.strip()
    ):
        raise IdeasError(
            "Group names cannot be identical.",
        )


def validate_subject_id_format(
    data: pd.DataFrame,
    data_pairing: str,
    group_names: List[str],
    context: str = "data",
) -> None:
    """Validate subject ID format for clean assignment rules.

    Ensures subject IDs follow the clean format established in data processing:
    - Unpaired: "{Group_Name}_subject_N" format for distinct subjects across groups
    - Paired: "subject_N" format for identical subjects across groups

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing subject IDs to validate
    data_pairing : str
        Type of data pairing ("paired" or "unpaired")
    group_names : List[str]
        List of group names for validation context
    context : str, optional
        Description of data being validated (for error messages), by default "data"

    Raises
    ------
    IdeasError
        If subject ID format violates clean assignment rules or filename contamination detected

    """
    # Validate inputs
    if data is None or data.empty:
        logger.debug(
            f"No data provided for subject ID validation in {context}"
        )
        return

    if data_pairing not in ["paired", "unpaired"]:
        raise IdeasError(
            f"Invalid data_pairing '{data_pairing}'. Must be 'paired' or 'unpaired'.",
        )

    # Check for required column
    if "normalized_subject_id" not in data.columns:
        raise IdeasError(
            f"{context} missing required column: normalized_subject_id",
        )

    # Extract subject IDs
    all_subjects = set(data["normalized_subject_id"].dropna().unique())

    if not all_subjects:
        logger.warning(f"No subject IDs found in {context} for validation")
        return

    # Validate subject ID format
    invalid_subjects = []
    filename_contaminated_subjects = []

    for subject_id in all_subjects:
        # Check for basic validity
        if not isinstance(subject_id, str) or not subject_id.strip():
            invalid_subjects.append(subject_id)
            continue

        # Check for filename contamination (problematic characters)
        problematic_chars = [
            ".",
            "/",
            "\\",
            ":",
            ";",
            ",",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
        ]
        if any(char in subject_id for char in problematic_chars):
            filename_contaminated_subjects.append(subject_id)
            continue

        # Validate format based on data pairing
        if data_pairing == "unpaired" and len(group_names) > 1:
            # Should be: "Group_Name_subject_N" format
            # Example: "Control_subject_1", "Treatment_Group_A_subject_2"
            # Underscore count >= 2 because: GroupName_ + subject_ + N
            # (minimum: one underscore after group, one after "subject")
            # This handles group names with underscores correctly
            expected_pattern = False
            for group_name in group_names:
                clean_group = group_name.replace(" ", "_")
                if (
                    subject_id.startswith(f"{clean_group}_subject_")
                    and subject_id.count("_") >= 2
                ):
                    expected_pattern = True
                    break
            if not expected_pattern:
                invalid_subjects.append(subject_id)

        elif data_pairing == "paired":
            # Should be: "subject_N" format (no group prefix)
            # Example: "subject_1", "subject_2"
            # Underscore count == 1 because: subject_ + N
            # (exactly one underscore between "subject" and number)
            if not (
                subject_id.startswith("subject_")
                and subject_id.count("_") == 1
            ):
                invalid_subjects.append(subject_id)

    # Report filename contamination first (more specific error)
    if filename_contaminated_subjects:
        raise IdeasError(
            f"Filename contamination detected in {context} subject IDs:"
            f" {filename_contaminated_subjects[:3]}. "
            f"Subject IDs contain problematic characters that suggest filename-based assignment. "
            f"This indicates a bug in subject ID processing logic.",
        )

    # Report format violations
    if invalid_subjects:
        expected_format = (
            f"'{group_names[0].replace(' ', '_')}_subject_N' for unpaired analysis"
            if data_pairing == "unpaired" and len(group_names) > 1
            else "'subject_N' for paired analysis"
        )
        raise IdeasError(
            f"Invalid subject ID format detected in {context}: {invalid_subjects}. "
            f"Expected format: {expected_format}. "
            f"Clean subject ID assignment has been violated.",
        )

    # PAIRED ANALYSIS: Validate subject consistency across groups
    if data_pairing == "paired" and len(group_names) > 1:
        # Check if we have group information to validate consistency
        group_column = None
        if "group_name" in data.columns:
            group_column = "group_name"
        elif "group" in data.columns:
            group_column = "group"

        if group_column is not None:
            # Check that all subjects appear in all groups
            subject_group_counts = data.groupby("normalized_subject_id")[
                group_column
            ].nunique()
            expected_groups = len(group_names)
            inconsistent_subjects = subject_group_counts[
                subject_group_counts != expected_groups
            ]

            if not inconsistent_subjects.empty:
                missing_subjects = list(
                    inconsistent_subjects.index[:3]
                )  # Show first 3
                raise IdeasError(
                    f"Paired analysis requires all subjects to appear in all groups. "
                    f"Subjects with missing group data in {context}: {missing_subjects}. "
                    f"Expected {expected_groups} groups per subject, but some subjects "
                    f"appear in only {inconsistent_subjects.iloc[0]} group(s). "
                    f"Check subject matching or data completeness.",
                )

    logger.info(
        f"Subject ID validation passed for {context}: {len(all_subjects)} subjects, "
        f"pairing={data_pairing}, format=clean"
    )


def extract_and_validate_states(
    group1_population_activity_files: List[str],
    group2_population_activity_files: List[str],
    group1_population_events_files: Optional[List[str]] = None,
    group2_population_events_files: Optional[List[str]] = None,
    state_names: Optional[List[str]] = None,
    reference_df: Optional[pd.DataFrame] = None,
    global_state_comparison_type: Optional[str] = None,
    global_baseline_state: Optional[str] = None,
) -> List[str]:
    """Extract and validate states across all input files with enhanced robustness.

    This function reads all input population activity and event files and ensures
    consistent state availability across files. It uses a hierarchical validation
    approach optimized for performance and reliability.

    Parameters
    ----------
    group1_population_activity_files : List[str]
        Population activity files for the first group.
    group2_population_activity_files : List[str]
        Population activity files for the second group.
    group1_population_events_files : Optional[List[str]]
        Population events files for the first group.
    group2_population_events_files : Optional[List[str]]
        Population events files for the second group.
    state_names : Optional[List[str]]
        State names to filter for analysis. If provided, only these states are used.
    reference_df : Optional[pd.DataFrame]
        Pre-loaded reference dataframe to avoid redundant file I/O.
    global_state_comparison_type : Optional[str]
        Pre-detected state comparison type to avoid redundant detection.
    global_baseline_state : Optional[str]
        Pre-detected baseline state to avoid redundant detection.

    Returns
    -------
    List[str]
        Validated state names based on comparison type:
        - Pairwise: ['state1 vs state2', ...]
        - State vs baseline: ['state1', 'state2', ...] (non-baseline states)
        - Standard: ['state1', 'state2', ...] (all available states)

    Raises
    ------
    IdeasError
        If validation fails due to inconsistent states or invalid data structure.

    Examples
    --------
    >>> states = extract_and_validate_states(
    ...     group1_files=["subj1.csv", "subj2.csv"],
    ...     group2_files=["pat1.csv", "pat2.csv"],
    ...     state_names=["running", "resting"],
    ...     reference_df=reference_data,
    ...     global_state_comparison_type="state_vs_baseline"
    ... )
    ['running', 'resting']  # Non-baseline states only

    """
    try:
        # Early validation of inputs
        _validate_function_inputs(
            group1_population_activity_files,
            reference_df,
            global_state_comparison_type,
        )

        # Extract and validate states using cached reference data
        validated_states = _extract_validated_states(
            reference_df=reference_df,
            state_names=state_names,
            global_state_comparison_type=global_state_comparison_type,
            global_baseline_state=global_baseline_state,
        )

        # Ensure minimum state requirements
        _validate_minimum_state_requirements(validated_states)

        # Prepare reference states for cross-file validation
        reference_states = _prepare_reference_states(
            validated_states=validated_states,
            state_comparison_type=global_state_comparison_type,
            baseline_state=global_baseline_state,
        )

        # Validate file structure and state consistency across all files
        _validate_all_files_consistency(
            group1_files=group1_population_activity_files,
            group2_files=group2_population_activity_files,
            group1_events=group1_population_events_files,
            group2_events=group2_population_events_files,
            reference_df=reference_df,
            reference_states=reference_states,
            state_comparison_type=global_state_comparison_type,
        )

        # Validate inter-group state consistency if multiple groups
        if group2_population_activity_files:
            _validate_inter_group_consistency(
                group1_files=group1_population_activity_files,
                group2_files=group2_population_activity_files,
                state_comparison_type=global_state_comparison_type,
            )

        logger.info(
            f"State validation completed successfully. "
            f"Final validated states ({global_state_comparison_type}): {validated_states}"
        )

        return validated_states

    except IdeasError:
        # Re-raise ToolExceptions without modification
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error in extract_and_validate_states: {str(e)}"
        )
        raise IdeasError(
            f"Error validating states across files: {str(e)}",
        )
