# Helpers for local file storage
import pandas as pd

# import pyreadr
import shutil
import os
import pickle
import pathlib
import yaml


# General functions
def to_pickle(py_object, filename):
    "Save python object as a pickle file."
    with open(filename, "wb") as file:
        pickle.dump(py_object, file)


def read_pickle(filename):
    "Load a pickle file."
    with open(filename, "rb") as file:
        return pickle.load(file)


def read_yaml(path):
    with open(path) as file:
        documents = yaml.full_load(file) #enables yaml tags
        return documents
    
def save_yaml(obj, path):
    with open(path, 'w') as file:
        documents = yaml.dump(obj, file)


def convert_df_format(source_path, target_path, source_format, target_format, keep_original=False):
    """
    Converts dataframe into another format.
    Supported types:
    -pickle
    -parquet
    -feather
    -rds
    -csv

    Parameters
    ---------
    source_path: str
        path of original dataframe. Includes directory and file extension
    target_path: str
    source_format: str
    target_format: str
    remove_original: bool
    """

    if source_format == "parquet":
        source_df = pd.read_parquet(source_path)

    elif source_format == "pickle":
        source_df = pd.read_pickle(source_path)

    elif source_format == "feather":
        source_df = pd.read_feather(source_path)

    elif source_format == "csv":
        source_df = pd.read_csv(source_path)

    elif source_format.lower() == "rds":
        source_df = pyreadr.read_r(source_path)[None]

    else:
        raise Exception("Source format not supported")

    if target_format == "parquet":
        source_df.to_parquet(target_path)

    elif target_format == "pickle":
        source_df.to_pickle(target_path)

    elif target_format == "feather":
        source_df.to_feather(target_path)

    elif target_format == "csv":
        source_df.to_csv(target_path)

    elif target_format.lower() == "rds":
        pyreadr.write_rds(target_path, source_df, compress="gzip")

    if not keep_original:
        try:
            if source_format == "parquet":
                shutil.rmtree(source_path)
            else:
                os.remove(source_path)
        except FileNotFoundError:
            print(f"Original source not deleted: {source_path}")


def remove_file_or_directory(local_file_path, verbose=False):

    try:
        if verbose:
            print(f"Trying to remove {local_file_path}")
        shutil.rmtree(local_file_path)
    except:
        if verbose:
            print(f"{local_file_path} file not found")

    # Remove as local directory
    try:
        if verbose:
            print(f"Trying to remove directory {local_file_path}")
        os.rmdir(local_file_path)
    except:
        if verbose:
            print(f"{local_file_path} directory not found")


def identify_large_files(dir_name="/home/", unit="G", exclude_paths=[]):
    """
    Identify large files
    
    Parameters
    ----------
    dir_name: str
        Directory path
    unit: str
        'T','G','M','K'
    exclude_paths: list of str
        Paths in `dir_name` to exlude (any file in this path)
    """
    # List all file path
    list_of_files = list(filter(os.path.isfile, [str(fileref) for fileref in pathlib.Path(dir_name).glob("**/*")]))

    # Compute scale
    unit = unit[0].lower()
    bytes_in_unit = {
        "k": 10 ** 3,  # 1KB = 1000 bytes; 1KiB = 1024 bytes
        "m": 10 ** 6,
        "g": 10 ** 9,
        "t": 10 ** 12,
    }
    unit_scaler = bytes_in_unit.get(unit, 1)
    sizes_in_bytes = [os.stat(x).st_size / unit_scaler for x in list_of_files]

    size_df = pd.DataFrame(dict(path=list_of_files, file_size=sizes_in_bytes))
    size_df["unit"] = unit.upper() + "B"
    size_df.sort_values("file_size", ascending=False, inplace=True)

    # Remove excluded paths
    excluded_paths = [f"{dir_name}/{p}" for p in exclude_paths]  # Prepend dir_name

    def _is_excluded(path, excluded_paths):
        for excluded_path in excluded_paths:
            if path.startswith(excluded_path):
                return True
        return False

    size_df["excluded"] = size_df["path"].map(lambda x: _is_excluded(x, excluded_paths))
    return size_df[~size_df.excluded].drop(columns=["excluded"])

