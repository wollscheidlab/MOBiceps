"""
File: rfePlusPlusWF.py
Author: Jens Settelmeier
This is the pipeline for the RFE++
Created: 5/26/23
"""
import argparse
import os
import sys
import json
import pandas as pd
import numpy as np

from MOBiceps.featureSelector import (
    robust_crossvalidated_rfe4,
    capture_sig_high_correlated_features,
)
from MOBiceps.expression_table import (
    create_rfe_expression_table,
    post_process_feature_table,
)
from MOBiceps.expression_table import remove_file_extension


def execute_rfePP(
    path_to_search_file,
    path_to_class_annotation_file,
    path_to_output,
    path_to_manifest_file=None,
    path_to_sample_annotation_file=None,
    bootstrapping_augmentation=False,
    feature_lvl="protein",
    gpu=False,
    noisy_augmentation=False,
    force_selection=True,
    phenotype_class=None,
):
    """

    Parameters
    ----------


    Returns
    -------

    """
    f = open(os.path.join(path_to_output, "rfePlusPlusLog.txt"), "w")
    # Save the original standard output
    original_stdout = sys.stdout
    # Set the standard output to the file we just opened
    sys.stdout = f
    print("Start RFE++...")
    # 1) Create expression table
    if "expression_table" in path_to_search_file:
        print("Use provided expression table for RFE++...")
        feature_expression_table = pd.read_csv(path_to_search_file).set_index("files")
        if "Unnamed: 0" in feature_expression_table.columns:
            feature_expression_table = feature_expression_table.drop(
                "Unnamed: 0", axis=1
            )
    else:
        print("Build expression table from search results for RFE++...")
        feature_expression_table = create_rfe_expression_table(
            path_to_search_file,
            path_to_class_annotation_file,
            path_to_output,
            imputation="none",
            feature_lvl=feature_lvl,
            manifest_file_path=path_to_manifest_file,
            percentage=1,  # for XGB we don't filter out nan values.
        )
    print("#### considered classes:", phenotype_class)
    if phenotype_class is not None:
        feature_expression_table = feature_expression_table[
            feature_expression_table["class"].isin(phenotype_class)
        ]
    copy_of_expression_table_w_classes = feature_expression_table.copy()
    X, y = post_process_feature_table(feature_expression_table)
    # 2) Apply RFE++
    if path_to_sample_annotation_file is not None:
        print(
            "columns of sample_annotation are:",
            pd.read_csv(path_to_sample_annotation_file).columns,
        )
        sample_annotation = pd.read_csv(path_to_sample_annotation_file)
        sample_annotation["files"] = sample_annotation["files"].apply(
            remove_file_extension
        )  # was "file"
        sample_annotation.set_index("files", inplace=True)  # was "file"
        sample_annotation = sample_annotation.loc[list(X.index)].copy()
    else:
        sample_annotation = None

    os.chdir(path_to_output)
    most_contributing_features = robust_crossvalidated_rfe4(
        X,
        y,
        n_features_to_select=None,
        patient_file_annotations=sample_annotation,
        boot_strapping=bootstrapping_augmentation,
        gpu=gpu,
        noisy_augmentation=noisy_augmentation,
        do_selection=force_selection,
    )
    print("Most classification relevant features:", most_contributing_features)
    np.save("golden_features.npy", most_contributing_features)
    golden_features = np.load("golden_features.npy")
    golden_features_df = pd.DataFrame(golden_features, columns=["feature", "voting"])
    golden_features_df.to_csv("golden_features.csv")

    key_features = golden_features_df["feature"].tolist()
    # compute Kendall correlation between golden features and other features in expression matrix
    class_specific_results = capture_sig_high_correlated_features(
        copy_of_expression_table_w_classes, key_features
    )

    # Convert each DataFrame to a JSON string or a nested dictionary
    json_dict = {
        key: value.to_json(orient="split")
        for key, value in class_specific_results.items()
    }

    # Save the dictionary to a JSON file
    with open(
        "class_specific_dataframes_of_highly_relevant_features.json", "w"
    ) as file:
        json.dump(json_dict, file)

    # Assuming you already have the df_combined DataFrame
    df_combined = pd.concat(
        [df for df in class_specific_results.values()], ignore_index=True, axis=0
    )

    # Create a temporary column for the absolute values of the 'Correlation' column
    df_combined["Abs_Correlation"] = df_combined["Correlation"].abs()
    # First, sort within each block
    df_combined.sort_values(
        by=["Key_Feature", "Other_Feature", "Abs_Correlation"],
        ascending=[True, True, False],
        inplace=True,
    )

    # Group by 'Key_Protein' and 'Other_Protein', and find the max 'Abs_Correlation' for each group
    group_max = (
        df_combined.groupby(["Key_Feature", "Other_Feature"])["Abs_Correlation"]
        .max()
        .reset_index()
    )

    # Sort the groups by 'Abs_Correlation' in descending order
    group_max.sort_values(by="Abs_Correlation", ascending=False, inplace=True)

    # Merge to get the sorted blocks with their respective rows
    df_sorted = pd.merge(
        group_max[["Key_Feature", "Other_Feature"]],
        df_combined,
        on=["Key_Feature", "Other_Feature"],
    )

    # Drop duplicates and the temporary 'Abs_Correlation' column
    df_sorted.drop_duplicates(subset=["Key_Feature", "Other_Feature"], inplace=True)
    df_sorted.drop("Abs_Correlation", axis=1, inplace=True)

    # Save to CSV
    df_sorted.to_csv("sorted_high_correlated_features_of_golden_features.csv")

    # Reset the standard output to its original value
    sys.stdout = original_stdout

    f.close()
    print("Finish!")
    return most_contributing_features


def parse_args():
    """

    Returns
    -------
    args : TYPE
        Default argument parser for console execution.

    """
    parser = argparse.ArgumentParser(
        description="Takes search output and performs RFE++"
    )
    parser.add_argument(
        "--i",
        "--path_to_search_file",
        type=str,
        default=os.getcwd(),
        help="path to the folder containing the search output of spectronaut or diann",
    )
    parser.add_argument(
        "--c",
        "--path_to_class_annotation_file",
        type=str,
        default=None,
        help="path to the class annotation file",
    )
    parser.add_argument(
        "--s",
        "--path_to_sample_annotation_file",
        type=str,
        default=None,
        help="path to the sample annotation file",
    )
    parser.add_argument(
        "--o", "--path_to_output", type=str, default=os.getcwd(), help="Output path"
    )
    parser.add_argument(
        "--m",
        "--path_to_manifest_file",
        type=str,
        default=None,
        help="Path to manifest file from Fragpipe",
    )
    parser.add_argument(
        "--b",
        "--bootstrapping_augmentation",
        type=bool,
        default=False,
        help="Use bootstrapping augmentation",
    )

    parser.add_argument(
        "--f",
        "--feature_lvl",
        type=str,
        default="peptide",
        help='Supported are "peptide" and "protein".',
    )
    parser.add_argument(
        "--g", "--gpu", type=bool, default=False, help="Support for GPU if True."
    )
    parser.add_argument(
        "--n",
        "--noisy_augmentation",
        type=bool,
        default=False,
        help="Bootstrapping with noisy resampling.",
    )
    parser.add_argument(
        "--h",
        "--force_selection",
        type=bool,
        default=True,
        help="Force the reduction to a handable amount of features.",
    )
    parser.add_argument(
        "--p",
        "--phenotype_class",
        nargs="*",
        default=None,
        help="List of classes that should be considered in the analysis.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    execute_rfePP(
        args.i,
        args.c,
        args.s,
        args.o,
        args.m,
        args.b,
        args.f,
        args.g,
        args.n,
        args.h,
        args.p,
    )
    # imputation is not used currently. 'none' is the default.
