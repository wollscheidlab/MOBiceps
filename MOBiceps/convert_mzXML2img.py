#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 15:19:14 2022

@author: Jens Settelmeier
"""

import os
import argparse
import time
from MOBiceps.mzXML2npyContainer import mzXML2npy
from MOBiceps.container_extraction import level_extraction_wrapper
from MOBiceps.container2img import msCont2ImageWrapper
from MOBiceps.npy2png import npy2pngWrapper


def mzXML2DIAimg(path_to_mzXML_folder, output_path=None, ms_lvl=1):
    """

    Parameters
    ----------
    path_to_mzXML_folder
    output_path
    ms_lvl

    Returns
    -------

    """

    start = time.time()
    path2_DIA_npy_Container_folder = os.path.join(
        path_to_mzXML_folder, f"output_DIA_npy_container_MS{ms_lvl}"
    )
    os.makedirs(path2_DIA_npy_Container_folder)
    mzXML2npy(
        path_to_input=path_to_mzXML_folder,
        path_to_output=path2_DIA_npy_Container_folder,
    )

    path2_spectrum_npy_folder = os.path.join(
        path2_DIA_npy_Container_folder, "output_spectrum_npy_files"
    )
    os.makedirs(path2_spectrum_npy_folder)
    level_extraction_wrapper(
        path_to_input=path2_DIA_npy_Container_folder,
        path_to_output=path2_spectrum_npy_folder,
        ms_lvl=ms_lvl,
    )

    path2_spectrum_image_npy_folder = os.path.join(
        path2_spectrum_npy_folder, "output_spectrum_img_npy_container"
    )
    os.makedirs(path2_spectrum_image_npy_folder)
    msCont2ImageWrapper(
        path_to_input=path2_spectrum_npy_folder,
        path_to_output=path2_spectrum_image_npy_folder,
        ms_lvl=[ms_lvl],
    )

    if output_path is None:
        path2__spectrum_images_folder = os.path.join(
            path2_spectrum_image_npy_folder, "output_spectrum_imgs"
        )
    else:
        path2__spectrum_images_folder = output_path
    os.makedirs(path2__spectrum_images_folder)
    npy2pngWrapper(
        path_to_input=path2_spectrum_image_npy_folder,
        path_to_output=path2__spectrum_images_folder,
    )

    end = time.time()
    print("Execution time in seconds:", end - start)
    # ToDo
    # delete all other files that were created in between if "delete_tmp_files is True"
    return


def parse_args():
    """

    Returns
    -------
    args : TYPE
        Default argument parser for console execution.

    """
    parser = argparse.ArgumentParser(
        description="Converts all DIA mzXML files in a folder to png images for a specific ms level."
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
        default=None,
        help="Output path to save images",
    )
    parser.add_argument(
        "--l",
        "--ms_lvl",
        type=int,
        default=1,
        help="ms level to be converted to images",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    mzXML2DIAimg(args.i, args.o, args.l)
