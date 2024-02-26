#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 15:38:12 2022
5th (optional) step in DIA Image creation
@author: Jens Settelmeier
"""
import glob
import argparse
import os
import numpy as np
from tqdm import tqdm
from skimage import exposure
from matplotlib import pyplot as plt


def npy2png(
    file, path_to_output, img_format="png", max_respected_mz=3000, hist_eq=True
):
    """

    Parameters
    ----------
    file : str
        file name of the npy DIA image.
    path_to_output : str
        Folder where to save the result.
    img_format: str
        Specifies which file format the image should have. Default is png.
        Other options are: jpeg

    Returns
    -------
    None. Saves the DIA npy image to the specified fileformat

    """
    npy_dia_img = np.load(file, allow_pickle=True)
    npy_dia_img = npy_dia_img[:, :max_respected_mz]

    if hist_eq is True:
        img_eq = exposure.equalize_hist(npy_dia_img)
    else:
        img_eq = npy_dia_img

    plt.imsave(
        os.path.join(path_to_output, f"png_{file}.{img_format}"), img_eq, cmap="gray"
    )
    return


def npy2pngWrapper(path_to_input, path_to_output, max_respected_mz=3000, hist_eq=True):
    """

    Parameters
    ----------
    path_to_input : str
        Folder to the npy DIA images.
    path_to_output : str
        Folder where to save the result.

    Raises
    ------
    Warning
        If a npy file is not a DIA image, it can not be converted and will be
        ignored.

    Returns
    -------
    None. Saves the DIA npy images to png images.

    """
    os.chdir(path_to_input)
    filenames = glob.glob("*.npy")
    fails = []
    for file in tqdm(filenames):
        print(f"##### process file {file} #####\n")
        try:
            npy2png(
                file, path_to_output, max_respected_mz=max_respected_mz, hist_eq=hist_eq
            )
        except Exception as e:
            print(
                f"{file} is not convertable to png format. An error occured in npy2pngWrapper: {e}"
            )
            fails.append(file)
    with open("fails.txt", "w") as output:
        output.write(str(fails))
    return


def parse_args():
    """

    Returns
    -------
    args : TYPE
        Default argument parser for console execution.

    """
    parser = argparse.ArgumentParser(description="Convert npy DIA images to png images")
    parser.add_argument(
        "--i",
        "--path_to_input",
        type=str,
        default=os.getcwd(),
        help="path to the folder containing all npy DIA images",
    )
    parser.add_argument(
        "--o",
        "--path_to_output",
        type=str,
        default=os.getcwd(),
        help="Output path to save images in png",
    )
    parser.add_argument(
        "--h",
        "--hist_eq",
        type=bool,
        default=True,
        help="Specifies if a histogram gray value equalization is applied or not.",
    )
    parser.add_argument(
        "--m",
        "--max_respected_mz",
        type=int,
        default=3000,
        help="Specifies the x resolution of the image, which corresponds to the maximum considered mz value. The original signal is cutted away after this value.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    npy2pngWrapper(args.i, args.o, args.h, args.m)
