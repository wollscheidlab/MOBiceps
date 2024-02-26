#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 22:29:47 2023

@author: Jens Settelmeier
"""

import os
import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np
import glob
from joblib import Parallel, delayed


def my_func2(cycle, window, min_mz, max_mz, container):
    """

    Parameters
    ----------
    cycle_step_size : int, optional
        The cycle step size which allows to skipp acquisition cycles.
        The default is 1.
    window : int
        Specifies the acquistion window to be converted to img. Needed for
        ms2 lvl.
    min_mz : float
        This will be the new zero mz value.
    max_mz : float
        This will be the max mz value considered.
    container : npy
        n x m array,  containing ms1 or ms2 acquisitions.

    Returns
    -------
    image_vec : npy
        one row of the construced image.

    """

    if len(container.shape) == 1:
        container = container.reshape(-1, 1)
    x_dim_image = int(np.round(max_mz)) - int(np.round(min_mz))
    test_vec_mz = container[cycle, window].mz - min_mz
    test_vec_inten = container[cycle, window].intensity

    rounded_test_vec_mz = np.round(test_vec_mz)
    unique_test_vec_vals = np.unique(rounded_test_vec_mz)

    comparison_mat = abs(np.subtract.outer(rounded_test_vec_mz, unique_test_vec_vals))
    comparison_mat[comparison_mat != 0] = np.nan
    row_index, col_index = np.nonzero(comparison_mat == 0)
    comparison_mat = pd.DataFrame(comparison_mat)
    counts = comparison_mat.count()

    image_vec = np.zeros(x_dim_image)
    for i in np.unique(col_index):
        image_vec[int(unique_test_vec_vals[i]) - 1] = (
            sum(test_vec_inten[col_index == i]) / counts[i]
        )
    return image_vec


def msCont2Image(
    container,
    window,
    filename,
    path_to_output,
    min_mz=1,
    max_mz=4000,
    cycle_step_size=1,
    log_transf=False,
):
    """

    Parameters
    ----------
    container : npy
        n x m array,  containing ms1 or ms2 acquisitions.
    window : int
        Specifies the acquistion window to be converted to img. Needed for
        ms2 lvl.
    min_mz : float, optional
        This will be the new zero mz value. The default is 250.
    max_mz : float, optional
        This will be the max mz value considered. The default is 2000.
    cycle_step_size : int, optional
        The cycle step size which allows to skipp acquisition cycles.
        The default is 1.

    Raises
    ------
    ValueError
        Consisty check, if all scans in the container have the same ms lvl.
        Makes sure not to mix ms1 and ms2 acquisitions.

    Returns
    -------
    image : npy
        Returns the gray value DIA image for a specific window or ms1 lvl.

    """
    # check consistency of the container
    if len(container.shape) == 1:
        container = container.reshape(-1, 1)

    ms_level = container[0, 0].ms_lvl
    for entry in container:
        if entry[0].ms_lvl != ms_level:
            raise ValueError("Not all entries have same ms level!")

    result2 = []

    for cycle in np.arange(len(container), step=cycle_step_size):
        result2.append(my_func2(cycle, window, min_mz, max_mz, container))
    image = np.array(result2)
    if log_transf is True:
        image = np.log(image + 1)

    np.save(os.path.join(path_to_output, f"img_{filename}_{window}.npy"), image)
    return image


def msCont2ImageWrapper(
    path_to_input,
    path_to_output,
    windows=None,
    cycle_step_size=1,
    min_mz=1,
    max_mz=4000,
    ms_lvl=[1, 2],
    log_transf=False,
):
    """

    Parameters
    ----------
    path_to_input : str
        Path to the input data (npy ms container).
    path_to_output : str
        Path to the output folder.
    window : int
        Specifies the acquistion window to be converted to img. Needed for
        ms2 lvl.
    cycle_step_size : int, optional
        The cycle step size which allows to skipp acquisition cycles.
        The default is 1.
    min_mz : float, optional
        This will be the new zero mz value. The default is 250.
    max_mz : float, optional
        This will be the max mz value considered. The default is 2000.
    ms_lvl : int or list of integers, optional
        Works only with 1,2 or [1,2] and specifies the ms level acquistiion
        which should be extracted. The default is [1,2].
    log_transf: boolean
        Default is True. If True the log transform x=log(x+1) is applied.

    Returns
    -------
    None. _build_img saves the results to the path_to_output path.

    """
    os.chdir(path_to_input)
    ms_lvl = np.array(ms_lvl).reshape(-1)
    if 1 in ms_lvl:
        filenames_ms1 = glob.glob("ms_1container*.npy")
        print("build ms1 images")
        _build_img(
            1,
            filenames_ms1,
            path_to_input,
            path_to_output,
            min_mz,
            max_mz,
            log_transf,
            windows=0,
        )
        print("finished ms1 images")
    if 2 in ms_lvl:
        filenames_ms2 = glob.glob("ms_2container*.npy")
        print("build ms2 images")
        _build_img(
            2,
            filenames_ms2,
            path_to_input,
            path_to_output,
            min_mz,
            max_mz,
            log_transf,
            windows,
        )
        print("finished ms2 images")
    return


def _build_img(
    ms_lvl,
    filenames,
    path_to_input,
    path_to_output,
    min_mz,
    max_mz,
    log_transf,
    windows=None,
):
    """

    Parameters
    ----------
    ms_lvl : int or list of integers
        Works only with 1,2 or [1,2] and specifies the ms level acquistiion
        which should be extracted.
    filenames : list
        list of filenames that are going to processed.
    windows : list of int
        Specifies the acquistion windows to be converted to imgs. Needed for
        ms2 lvl.
    path_to_input : str
        Path to the input data (npy ms container).
    path_to_output : str
        Path to the output folder.
    min_mz : float
        This will be the new zero mz value.
    max_mz : float
        This will be the max mz value considered.
    log_transf: boolean
        Default is True. If True the log transform x=log(x+1) is applied.

    Returns
    -------
    None. Saves the results to the path_to_output path.

    """
    fails = []

    for file in tqdm(filenames):
        try:
            print(f"##### process file {file} #####\n")
            container = np.load(os.path.join(path_to_input, file), allow_pickle=True)
            if windows is None:
                windows = np.arange(container.shape[1])
            else:
                windows = np.array(windows).reshape(-1)
            Parallel(n_jobs=-1, backend="loky", verbose=1)(
                delayed(msCont2Image)(
                    container,
                    window,
                    filename=file,
                    path_to_output=path_to_output,
                    min_mz=min_mz,
                    max_mz=max_mz,
                    cycle_step_size=1,
                    log_transf=log_transf,
                )
                for window in windows
            )
        except Exception as e:
            print(f"An error occured in _build_img: {e}")
            fails.append(file)
        with open(f"fails_{ms_lvl}.txt", "w") as output:
            output.write(str(fails))
    return


def parse_args():
    """

    Returns
    -------
    args : TYPE
        Default argument parser for console execution.

    """
    parser = argparse.ArgumentParser(
        description="converts a ms1 or ms2 container to a image in npy format."
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
        "--css",
        "--cycle_step_size",
        type=int,
        default=1,
        help="cycle step size sampling for image creation",
    )
    parser.add_argument(
        "--minmz", "--min_mz", type=int, default=1, help="minimum mz value"
    )  # default 250
    parser.add_argument(
        "--maxmz", "--max_mz", type=int, default=4000, help="maximum mz value"
    )  # default 2000
    parser.add_argument(
        "--ml",
        "--ms_lvl",
        type=int,
        default=[1, 2],
        help="levels which should be converted to images",
    )
    parser.add_argument(
        "--win",
        "--windows",
        type=int,
        default=None,
        help="windows of the scan which should be converted to image",
    )
    parser.add_argument(
        "--l",
        "--log_transf",
        type=bool,
        default=False,
        help="Apply log transform x=log(x+1)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    msCont2ImageWrapper(
        args.i, args.o, args.win, args.css, args.minmz, args.maxmz, args.ml
    )
