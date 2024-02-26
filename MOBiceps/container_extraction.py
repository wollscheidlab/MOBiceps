#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:00:18 2022
3ed step in DIA Image creation
@author: Jens Settelmeier
"""
import os
import glob
import argparse
import numpy as np
from tqdm import tqdm


def ms1extraction(container):
    """

     Parameters
     ----------
    container : npy
        Is a n x m matrix which contains in the first column the ms1 acquisitions.
        In all other columns ms2 acquisitions. n is the number o cycles,
        defined by the number of ms1 scans.
        Note this differs from the definition that a cycle is completed if
        on ms2 level the whole mz range is sampled.

     Returns
     -------
     ms1_cont : npy
         n x m array,  containing ms1 acquisitions.

    """
    ms1_cont = container[:, 0]
    return ms1_cont


def ms2extraction(container, reshape_format=None):
    """

     Parameters
     ----------
    container : npy
        Is a n x m matrix which contains in the first column the ms1 acquisitions.
        In all other columns ms2 acquisitions. n is the number o cycles,
        defined by the number of ms1 scans.
        Note this differs from the definition that a cycle is completed if
        on ms2 level the whole mz range is sampled.
     reshape_format : list of two integers, optional
         The first integer specifies the number of rows, the second the number
         of columns. The default is None.
     n : integer, optional
         Corresponds to the number of ms1 acquisitions within one cycle,
         where a cycle is completet if the whole mz range is sampled on
         ms2 lvl. The default is 3.

     Returns
     -------
     ms2_mat : npy
         n x m array,  containing ms2 acquisitions.

    """
    ms2_mat = container[:, 1:]
    first_precursor_window = ms2_mat[0, 0].precursor_window
    for i, row in enumerate(ms2_mat[1:, :]):
        if row[0].precursor_window == first_precursor_window:
            n = i + 1
            break
    if (
        n > 1
    ):  # could be removed, but if n=1 then the for loop can be saved -> faster execution time
        last_precusor_window = ms2_mat[i, -1].precursor_window

        for k, j in enumerate(np.flip(np.arange(ms2_mat.shape[0]))):
            if ms2_mat[j, -1].precursor_window == last_precusor_window:
                break
        if k > 0:
            ms2_mat = ms2_mat[:-k, :]

    if reshape_format is None:
        if n is not None:
            reshape_format = [int(ms2_mat.shape[0] / n), int(ms2_mat.shape[1] * n)]
    ms2_mat = ms2_mat.reshape(reshape_format)
    return ms2_mat


def level_extraction(container, ms_lvl, reshape_format=None):
    """
     Parameters
     ----------
    container : npy
        Is a n x m matrix which contains in the first column the ms1 acquisitions.
        In all other columns ms2 acquisitions. n is the number o cycles,
        defined by the number of ms1 scans.
        Note this differs from the definition that a cycle is completed if
        on ms2 level the whole mz range is sampled.
     ms_lvl : int or list of integers
         Works only with 1,2 or [1,2] and specifies the ms level acquistiion
         which should be extracted.
     reshape_format : list of two integers, optional
         The first integer specifies the number of rows, the second the number
         of columns. The default is None.
     n : integer, optional
         Corresponds to the number of ms1 acquisitions within one cycle,
         where a cycle is completet if the whole mz range is sampled on
         ms2 lvl. The default is 3.

     Raises
     ------
     ValueError
         If a meaningless ms_lvl is supported.

     Returns
     -------
     list
         The first component of the list is the npy array with the ms1 level,
         the second one with the ms2 level, if [1,2] was provided as ms_lvl.
         Otherwise the list only contains the npy array of the provided ms_lvl.

    """

    if ms_lvl == 1:
        return [ms1extraction(container)]
    elif ms_lvl == 2:
        return [ms2extraction(container, reshape_format)]
    elif ms_lvl == [1, 2]:
        return [ms1extraction(container), ms2extraction(container, reshape_format)]
    else:
        raise ValueError("Only ms level 1 and 2 or [1,2] are supported")


def level_extraction_wrapper(
    path_to_input, path_to_output, ms_lvl, container_name_prefix="container"
):
    """

    Parameters
    ----------
    path_to_input : str
        Path to the npy container(s) containing the ms1 and ms2 acquisitions of
        a DIA.
    path_to_output : str
        Path to output folder.
    ms_lvl : integer or list of integers
        Works only with 1,2 or [1,2] and specifies the ms level acquistiion
        which should be extracted.
    container_name_prefix : str, optional
        prefix to find the container files in a folder.
        The default is 'container'.

    Returns
    -------
    None. The output is written to disk.

    """
    os.chdir(path_to_input)
    filenames = glob.glob(f"{container_name_prefix}*.npy")
    fails = []
    for file in tqdm(filenames):
        print(f"##### process file {file} #####\n")
        try:
            container = np.load(os.path.join(path_to_input, file), allow_pickle=True)
            extractions = level_extraction(container, ms_lvl)
            for extraction, lvl in zip(extractions, np.array(ms_lvl).reshape(-1)):
                np.save(
                    os.path.join(path_to_output, "ms_" + str(lvl) + file), extraction
                )
        except Exception as e:
            print(f"An error occured in level_extraction_wrapper: {e}")
            fails.append(file)
    with open(f"fails_{ms_lvl}.txt", "w") as output:
        output.write(str(fails))
    print("done!")
    return


def parse_args():
    """

    Returns
    -------
    args : TYPE
        Default argument parser for console execution.

    """
    parser = argparse.ArgumentParser(
        description="Extracts ms1 and ms2 level from a general DIA npy container."
    )
    parser.add_argument(
        "--i",
        "--path_to_input",
        type=str,
        default=os.getcwd(),
        help="path to the folder containing all files to be converted",
    )
    parser.add_argument(
        "--o",
        "--path_to_output",
        type=str,
        default=os.getcwd(),
        help="Output path to save images",
    )
    parser.add_argument(
        "--ml",
        "--ms_lvl",
        type=int,
        default=[1, 2],
        help="levels which should be extracted",
    )
    parser.add_argument(
        "--cp",
        "--container_name_prefix",
        type=str,
        default="container",
        help="prefix to identify container npy files",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    level_extraction_wrapper(args.i, args.o, args.ml, args.cp)
