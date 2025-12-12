import copy
import re

# from jsonschema import validate
import numpy as np

# from ideas_commons.constants import FileCategory


def compare_float_dataframes(df1, df2, atol=1e-08):
    """Compare the contents of two float data frames and their types

    :param df1: first pandas dataframe to be compared
    :param df2: second pandas dataframe to be compared
    :return: True if the contents of the dataframes are the same, False otherwise
    """
    # compare indices, column names, and dtypes
    if (
        len(df1) != len(df2)
        or df1.index.tolist() != df2.index.tolist()
        or df1.columns.tolist() != df2.columns.tolist()
        or df1.dtypes.tolist() != df2.dtypes.tolist()
    ):
        return False

    # compare numeric values
    df1_num = df1.select_dtypes(include=np.number)
    df2_num = df2.select_dtypes(include=np.number)
    if not np.allclose(df1_num.values, df2_num.values, atol=atol, equal_nan=True):
        return False

    # compare non-numeric values
    for c in df1.columns:
        if c not in df1_num:
            if (df1[c] != df2[c]).all():
                return False

    return True


def compare_metadata_dictionaries(expected_dict, actual_dict):
    """Compare expected and actual metadata dictionaries.
    - The 'group_id' field is ignored since uuid's are unique to each run.

    :param expected_dict: expected metadata dictionary
    :param actual_dict: actual metadata dictionary
    """
    exp = copy.deepcopy(expected_dict)
    act = copy.deepcopy(actual_dict)
    del exp["ideas"]["dataset"]["group_id"]
    del act["ideas"]["dataset"]["group_id"]
    return exp == act


# def validate_json_schema(data, schema, load_data_from_file=True):
#     """Validate json-formatted data against a pre-defined json schema.

#     :param data: json-formatted data file
#     :param schema: json schema file
#     """
#     # read schema
#     with open(schema, "r") as f:
#         schema = json.load(f)

#     # read data
#     if load_data_from_file:
#         with open(data, "r") as f:
#             data = json.load(f)

#     validate(instance=data, schema=schema)


def clean_dict(obj, func=lambda key: re.match("^.+_id[s]?$", key)):
    """Clean dictionaries by removing every key for which callable func returns True.

    :param obj: dict or list of dicts to clean
    :param func: callable that takes a key and returns True for each key to delete
    """
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if func(key):
                del obj[key]
            else:
                clean_dict(obj[key], func)
    elif isinstance(obj, list):
        for i in reversed(range(len(obj))):
            clean_dict(obj[i], func)
    # else: neither a dict nor a list, do nothing


def retrieve_dict_values(obj, target_key):
    """Recursively retrieve all values associated with
    the specified field in a given dictionary.

    :param obj: dict or list of dicts
    :param k: key for which we want to extract values
    """
    values = []
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if key == target_key:
                values.append(obj[key])
            else:
                values.extend(retrieve_dict_values(obj[key], target_key))
    elif isinstance(obj, list):
        for i in reversed(range(len(obj))):
            values.extend(retrieve_dict_values(obj[i], target_key))

    return values
