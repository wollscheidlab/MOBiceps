#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 15:49:53 2022

@author: Jens Settelmeier
"""

import pandas as pd
import os
import numpy as np
import argparse


def mod_col_name(col):
    out = os.path.split(col)[1]
    return out


def table_preprocessing(df):
    df.columns = df.columns.to_series().apply(mod_col_name)
    df.set_index(df.columns[0], inplace=True)
    return df


def reliable_prots_filter(df, percentage):
    """
    caution! Makes only sense to apply on a dataframe which corresponds to one biological condition!
    """
    NaN_stats = df.isnull().sum(axis=1) / df.shape[1]
    reliable_prots = np.sort(NaN_stats[NaN_stats <= percentage].index.to_list())
    print(len(reliable_prots[reliable_prots == "nan"]))
    reliable_prots = reliable_prots[reliable_prots != "nan"]
    df = df.loc[reliable_prots]
    return df, reliable_prots


def clean_feature_expression_table(
    df, percentage, imputation="none", noise_scaling=0.1
):

    df = table_preprocessing(df)
    # exclude iRTs
    df = df[df.index != "Biognosys|iRT-Kit_WR_fusion GN=iRTKit"].copy()

    df, selected_prots = reliable_prots_filter(
        df, percentage
    )  # not necessary anymore in xgb version.

    if imputation == "zero":
        df.fillna(
            0, inplace=True
        )  # here instead of just filling with zero, other smarter imputation strategies could help
    elif imputation == "gaussian":
        df = gaussian_imputation(df, noise_scaling)
    elif imputation == "median":
        df = median_imputation(df)
    elif imputation == "mean":
        df = mean_imputation(df)
    elif imputation == "none":
        df = df
    return df


def mean_imputation(df):
    for i in range(df.shape[1]):
        df.iloc[:, i][df.isnull().iloc[:, i]] = np.mean(
            df.iloc[:, i][~df.isnull().iloc[:, i]]
        )
    return df


def median_imputation(df):
    for i in range(df.shape[1]):
        df.iloc[:, i][df.isnull().iloc[:, i]] = np.median(
            df.iloc[:, i][~df.isnull().iloc[:, i]]
        )
    return df


def gaussian_imputation(df, noise_scaling=0.1):
    for i in range(df.shape[1]):
        df.iloc[:, i][df.isnull().iloc[:, i]] = (
            np.random.normal(
                scale=noise_scaling
                * np.ones(df.iloc[:, i][df.isnull().iloc[:, i]].shape[0])
                * (df.iloc[:, i].std())
            )
            + df.iloc[:, i].min()
        )
    return df


def cleanTable(input_path, percentage=0.95, sep=None, zero_imput=False):
    if not os.path.isfile(input_path):
        raise ValueError(f"File {input_path} does not exist.")
    df2 = None
    if input_path[-3:] == "csv":
        if sep is None:
            sep = ","
        df = pd.read_csv(input_path, sep=sep)
    else:
        if sep is None:
            sep1 = "\t"
            sep2 = ","
        try:
            df = pd.read_table(input_path, sep=sep1)
        except Exception as e:
            print(f"read with sep={sep1} didn't work. Exception: {e}")
        try:
            df2 = pd.read_table(input_path, sep=sep2)
        except Exception as e:
            print(f"read with sep={sep2} didn't work. Exception: {e}")
        if df2 is not None:
            try:
                if df2.shape[1] > df.shape[1]:
                    df = df2.copy()
            except AttributeError:
                pass
    df = clean_feature_expression_table(df, percentage, imputation=zero_imput)
    return df


def parse_args():
    """
    Returns
    -------
    args : TYPE
        Default argument parser for console execution.

    """
    parser = argparse.ArgumentParser(
        description="Cleans an expression table constructed with iq."
    )
    parser.add_argument(
        "--i",
        "--input_path",
        type=str,
        default=os.getcwd(),
        help="path to the tsv file with the feature expressions.",
    )
    parser.add_argument(
        "--p",
        "--percentage",
        type=str,
        default=0.95,
        help="The minimum fraction of samples the features have to be expressed.",
    )
    parser.add_argument(
        "--s", "--seperator", type=str, default=None, help="Seperator for the table."
    )
    parser.add_argument(
        "--m",
        "--missing_vals",
        type=str,
        default="none",
        help='Supported are "none", "zero", "gaussian", "median", "mean". If you apply a ML method '
        "on top, make always sure not to have a information leakage by accident.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cleanTable(args.i, args.p, args.s, args.m)
