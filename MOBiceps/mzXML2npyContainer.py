#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 13:09:50 2022
2nd step in DIA Image creation
@author: Jens Settelmeier
"""

import os
import argparse
from tqdm import tqdm
from pyopenms import MSExperiment, MzXMLFile
import collections
import numpy as np
import glob
import re
from MOBiceps.dataClasses import MSScan
from pyopenms import Normalizer, GaussFilter, PeakPickerHiRes


def mzXML2npy(path_to_input, path_to_output, cycle_step_size=1, preprocessing=False):
    """
    Parameters
    ----------
    path_to_input : str
        Absolute path to the mzXML files which will be converted.
    path_to_output : str
        Absolute path to the output folder.
    cycle_step_size : int, optional
        The cycle step size which allows to skipp acquisition cycles.
        The default is 1.

    Raises
    ------
    ValueError
        checks consitency of cycles.

    Returns
    -------
    container : npy
        Is a n x m matrix which contains in the first column the ms1 acquisitions.
        In all other columns ms2 acquisitions. n is the number of cycles,
        defined by the number of ms1 scans.
        Note this differs from the definition that a cycle is completed if
        on ms2 level the whole mz range is sampled.

    """

    filenames_with_path = glob.glob(os.path.join(path_to_input, "*.mzXML"))
    filenames = [os.path.basename(f) for f in filenames_with_path]
    print("#### filenames #####", filenames)
    fails = []
    print("##### start processing #####\n")
    for filename in tqdm(filenames):  # iterate through the mzxml files in the folder

        print(f"##### process file {filename} #####\n")
        try:

            exp_0 = MSExperiment()  # initilize pyopenms MSExperiment object
            # load mzxml file into variable exp_0
            MzXMLFile().load(os.path.join(path_to_input, filename), exp_0)
            # perform some usual spectrum preprocessing steps (Hannes Röst)
            if preprocessing is True:
                # smoothing
                gf = GaussFilter()
                param = gf.getParameters()
                param.setValue("gaussian_width", 1.0)  # needs wider width
                gf.setParameters(param)
                gf.filterExperiment(exp_0)

                # centroiding
                exp = MSExperiment()
                PeakPickerHiRes().pickExperiment(exp_0, exp)

                # normalizing
                normalizer = Normalizer()
                param = normalizer.getParameters()
                param.setValue("method", "to_one")
                normalizer.setParameters(param)

                normalizer.filterPeakMap(exp)
            else:
                exp = exp_0  # no preprocessing performed

            plane_txt = open(os.path.join(path_to_input, filename), "r")

            # figure out, how many acquisition windows do we have.
            # catch out the casses, where different amount of acquisition windows were used. Will the once with a last falsy cycle also excluded? -No
            MS1_level_idxs = []
            potential_window_sizes = []
            for i, scan in enumerate(exp):
                if scan.getMSLevel() == 1:
                    MS1_level_idxs.append(i)
                    if i > 0:
                        potential_window_sizes.append(i - MS1_level_idxs[-2])
            counter = collections.Counter(potential_window_sizes)
            if (
                len(counter) > 1
            ):  # in this case, we have different aquisition windows in between MS1 acquisitions and the file is not suitable as image
                print(
                    f"##### different amount of acquisition windows ({counter}) in file {filename}. File is not suitable as image. #####\n"
                )
                fails.append(filename)
                continue
            else:
                windows = list(counter)[0] - 1

            # catch the cases where aquisition stopped with an MS1 step.
            nr_spectra = exp.getNrSpectra()
            while nr_spectra % (windows + 1) != 0:
                tmp_idx_1 = nr_spectra - 1
                while exp[tmp_idx_1].getMSLevel() != 2:
                    tmp_idx_1 -= 1

                tmp_idx_2 = tmp_idx_1
                while exp[tmp_idx_2].getMSLevel() == 2:
                    tmp_idx_2 -= 1

                if (tmp_idx_1 - tmp_idx_2) % (windows) == 0:
                    nr_spectra = tmp_idx_1 + 1
                else:
                    nr_spectra = (
                        tmp_idx_2 + 1
                    )  # here also +1? -yes bc nr. of spectra = index +1, since we start counting from 0 on.

            circle_amount = int(
                nr_spectra / (windows + 1)
            )  # ms_lvls = np.tile(np.hstack([1,np.ones(windows)*2]),circle_amount)
            scan_circles = np.reshape(
                np.tile(np.arange(circle_amount), windows + 1),
                [windows + 1, circle_amount],
            ).T.reshape(-1)

            # check if scan_circles construction workes as expected
            for i in range(circle_amount):
                if scan_circles[scan_circles == i].shape[0] != windows + 1:
                    raise ValueError(f"cycle {i} has not {windows +1} columns!")

            # another check: It should hold: circle_amount \times (windows+1) = nr_spectra
            if circle_amount * (windows + 1) != nr_spectra:
                raise ValueError(
                    "The formular circle_amount * (windows+1) = nr_spectra is not fulfilled!"
                )

            # extract precoursor windows
            xml_lines = []
            patrn = "</precursorMz>"
            for line in plane_txt:
                if re.search(patrn, line):
                    xml_lines.append(line)

            patrn2 = ">.*<"
            precursor_masses_lines = []
            for line in xml_lines:
                if re.search(patrn2, line):
                    precursor_masses_lines.append(re.search(patrn2, line))

            precursor_masses_list = []
            for line in precursor_masses_lines:
                precursor_masses_list.append(float(line[0][1:-1]))
            plane_txt.close()

            precursor_masses_idx = 0

            tmp_cont = []
            for k in np.arange(nr_spectra):
                scan = exp[int(k)]
                ms_lvl = scan.getMSLevel()
                if ms_lvl == 2:
                    precursor_masses = precursor_masses_list[precursor_masses_idx]
                    precursor_masses_idx += 1
                else:
                    precursor_masses = None

                mz, intensity = scan.get_peaks()
                mat_obj = MSScan(
                    ms_lvl=ms_lvl,
                    circle_no=scan_circles[k],
                    precursor_window=precursor_masses,
                    retention_time=scan.getRT(),
                    mz=mz,
                    intensity=intensity,
                )

                tmp_cont.append(mat_obj)
            container = np.array(tmp_cont).reshape([circle_amount, windows + 1])
            np.save(
                os.path.join(path_to_output, f"container_{filename}.npy"), container
            )
        except Exception as e:
            print(f"An error occrured in the function mzXML2npy : {e}")
            fails.append(filename)
    with open("fails.txt", "w") as output:
        output.write(str(fails))
    try:
        return container
    except Exception as e:
        print(f"Container could not be extracted. Error in mzXML2npy : {e}")
        return None


def parse_args():
    """

    Returns
    -------
    args : TYPE
        Default argument parser for console execution.

    """
    parser = argparse.ArgumentParser(
        description="Converts a DIA mzXML file to a npy container."
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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    container_out = mzXML2npy(args.i, args.o)
