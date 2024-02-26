"""
File: expression_table.py
Author: Jens Settelmeier
Created: 6/1/23
"""
import argparse
import os
import numpy as np
import re
import pandas as pd

from MOBiceps.create_expression_table import build_expression_table
from MOBiceps.clean_expression_table import cleanTable


def remove_file_extension(filename):
    return os.path.splitext(filename)[0]


def post_process_feature_table(feature_expression_table):
    y = feature_expression_table["class"]
    y.index = y.index.to_series().apply(remove_file_extension)
    feature_expression_table.drop("class", axis=1, inplace=True)
    X = feature_expression_table.copy()
    X.index = X.index.to_series().apply(remove_file_extension)
    return X, y


def table_utility(search_software, df, class_annotations):
    if search_software == "spectronaut":
        updated_filenames = []
        for i in range(class_annotations.shape[0]):
            updated_filenames.append(re.split(".mzML", class_annotations.index[i])[0])
        class_annotations.index = updated_filenames
    elif search_software == "diann" or search_software == "ion_quant":
        print(f"{search_software} output detected, no column names adjust necessary...")
    else:
        print(
            "Please provide a valid search software name. Currently supported are spectronaut and diann"
        )
    idx1 = set(list(class_annotations.index))
    idx2 = set(list(df.columns))
    idx = list(idx1.intersection(idx2))
    class_annotations = class_annotations.loc[idx].copy()
    df = df.reindex(np.sort(df.columns), axis=1).copy()
    class_annotations = class_annotations.reindex(
        np.sort(class_annotations.index), axis=0
    ).copy()
    feature_expression_table = pd.concat([df.T, class_annotations], axis=1)
    return feature_expression_table


def create_rfe_expression_table(
    path_to_search_output_file,
    path_to_class_annotation_file,
    path_to_output,
    imputation="none",
    feature_lvl="protein",
    manifest_file_path=None,
    percentage=None,
):
    """
    Parameters
    ----------
    feature_lvl
    imputation
    path_to_search_output_file
    path_to_class_annotation_file
    path_to_output

    Returns
    -------
    feature_expression_table for RFE++
    """

    # 1) Load class annotations
    class_annotations = pd.read_csv(path_to_class_annotation_file)
    class_annotations.rename(
        columns={class_annotations.columns[0]: "files"}, inplace=True
    )
    class_annotations.rename(
        columns={class_annotations.columns[1]: "class"}, inplace=True
    )
    class_annotations = class_annotations.set_index("files")

    number_of_classes = len(class_annotations["class"].unique())
    if percentage is None:
        percentage = (
            1 / number_of_classes
        )  # still applying in xgboost verions? actually, not necessary...

    search_software = build_expression_table(
        path_to_search_output_file, path_to_output, feature_lvl, manifest_file_path
    )
    iq_expression_table_path = os.path.join(path_to_output, "iq_expression_table.tsv")
    df = cleanTable(
        iq_expression_table_path, percentage=percentage, zero_imput=imputation
    )  # ToDo zero_imput is missleading name and shuld be changed to imputation or impute

    feature_expression_table = table_utility(search_software, df, class_annotations)
    feature_expression_table.index.name = "files"
    feature_expression_table.to_csv(
        os.path.join(path_to_output, "RFE_PP_feature_expression_table.csv")
    )

    return feature_expression_table


def parse_args():
    """

    Returns
    -------
    args : TYPE
        Default argument parser for console execution.

    """
    parser = argparse.ArgumentParser(description="Prepare input for RFE++")
    parser.add_argument(
        "--s",
        "--path_to_search_output_file",
        type=str,
        default=os.getcwd(),
        help="Path to search output. Currently spectronaut and diann output is suppoerted.",
    )
    parser.add_argument(
        "--c",
        "--path_to_class_annotation_file",
        type=str,
        default=os.getcwd(),
        help="Path to search annotation file.",
    )
    parser.add_argument(
        "--o", "--path_to_output", type=str, default=os.getcwd(), help="Output path"
    )
    parser.add_argument(
        "--m",
        "--imputation",
        type=str,
        default=os.getcwd(),
        help='Currently "mean", "median", "zero",' ' "gaussian" are supported.',
    )
    parser.add_argument(
        "--f",
        "--feature_lvl",
        type=str,
        default="peptide",
        help='Supported are "peptide" and "protein".',
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    feature_expression_table = create_rfe_expression_table(
        args.s, args.c, args.o, args.m, args.f
    )
