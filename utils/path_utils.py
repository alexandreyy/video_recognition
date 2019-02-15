"""
Path utils.
"""

from glob import glob
import os


def get_files_in_directory(dir_path, extensions=['mp4', 'avi', 'mpg']):
    """
    Get a list of files which are inside the given list
    """

    file_paths = []
    for filename in glob(os.path.join(dir_path, '*')):
        ext = os.path.basename(filename).split(".")[-1]

        if ext.lower() in extensions:
            file_paths.append(filename)

    return file_paths


def get_file_name(file_path):
    """
    Get file name from file path.
    """

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    return file_name
