#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 18:16:15 2022

@author: Jens Settelmeier
"""

import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class DIAImageDataset(Dataset):
    def __init__(self, meta_df, filenames_with_path, transform=None, class_to_num=None):
        """
        Args:
            meta_df (string): Path to the csv file with annotations.
            filenames_with_path (string): All paths to the images.
            transform (callable, optional): Optional transform to be applied
        """
        if type(meta_df) == str:
            self.meta_data = pd.read_csv(meta_df).set_index("files")
            if "Unnamed: 0" in self.meta_df.columns:
                self.meta_data = self.meta_df.drop("Unnamed: 0", axis=1)
        else:
            self.meta_data = meta_df

        self.filenames = filenames_with_path
        self.transform = transform

        # Get a list of unique class names
        class_names = self.meta_data["class"].unique()
        # Create a dictionary mapping class names to integers
        if class_to_num is None:
            self.class_to_num = {class_names[i]: i for i in range(len(class_names))}
        else:
            self.class_to_num = class_to_num

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = self.filenames[idx]
        image = Image.open(filename)

        label = None
        for j in self.meta_data.index:
            # we assume the images have a file name like: png_img_ms_1container_sgoetze_A1803_040.mzXML.npy_0.npy_part_1.png
            # and the idx is sgoetze_A1803_040
            if j + "." in filename:  # renaming files will crash this :(
                label = self.meta_data.loc[j, "class"]
                break
        # If label is not found, return None -> no match in meta data.
        if label is None:
            # raise ValueError(f"Meta data for the image {filename} not found.")
            print(f"Meta data for the image {filename} not found.")
            return None

        # Convert label from string to numerical using the dictionary
        label = self.class_to_num[label]
        # Convert label to a torch tensor
        label = torch.tensor(label)

        if self.transform:
            image = self.transform(image)

        sample = {
            "image": np.array(image),
            "label": label,
            "filename": filename,
        }  # np array necessary for tensorflow. Pytorch and sklearn also works with pil object.

        return sample

    def get_class_to_num_mapping(self):
        return self.class_to_num


class MSScan:
    def __init__(
        self,
        ms_lvl,
        circle_no,
        retention_time,
        precursor_window,
        mz,
        intensity,
        precursor_charge=None,
    ):
        self.ms_lvl = ms_lvl
        self.cicle_no = circle_no
        self.retention_time = retention_time
        self.precursor_window = precursor_window
        self.mz = mz
        self.intensity = intensity
        self.precursor_charge = (
            precursor_charge  # DIA files don't have precursor charges
        )
