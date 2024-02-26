"""
File: create_expression_table.py
Author: Jens Settelmeier

Created: 5/25/23
"""
import argparse
import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_boxplots(df, title, save_path):
    # Selecting only the relevant columns (excluding the first column)
    data_to_plot = df.iloc[:, 1:]

    # Creating a boxplot
    plt.figure(figsize=(20, 20))
    boxplot = data_to_plot.boxplot(
        showmeans=True,
        meanline=True,
        meanprops={"linestyle": "-", "linewidth": "1.5", "color": "red"},
    )

    plt.title(title)
    plt.xlabel("Columns")
    plt.ylabel("Values")
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.savefig(save_path, format="pdf")
    plt.close()


def feature_median_normalize(df, save_path):

    df.iloc[:, 1:] = np.log2(
        df.iloc[:, 1:] + 1
    )  # + 1 will make zero values 0 after log transform.
    df_copy = df.copy()
    df_copy.replace(0, np.nan, inplace=True)  # log2 cut offs for median computations.

    filename_w_path1 = os.path.join(
        save_path, "pre_normalisation_feature_distributions.pdf"
    )
    plot_boxplots(df, title="pre normalisation", save_path=filename_w_path1)

    medians = df_copy.iloc[:, 1:].median()  # m # ignores np.nan values by default.
    mean = np.mean(medians)
    f = mean - medians
    ff = np.array(f).reshape([1, len(f)])
    df.iloc[:, 1:] = df.iloc[:, 1:] + ff
    normalized_df = df
    filename_w_path2 = os.path.join(
        save_path, "post_normalisation_feature_distributions.pdf"
    )
    plot_boxplots(normalized_df, title="post normalisation", save_path=filename_w_path2)
    return normalized_df


def ion_quant_code(
    path_to_input,
    path_to_output,
    feature_lvl,
    manifest_file_path,
    median_normalization=True,
):

    save_path = path_to_output
    path_to_output = os.path.join(path_to_output, "iq_expression_table.tsv")
    # check ion quant output depending on feature_lvl.
    if feature_lvl == "peptide":
        expected_file_name = "combined_ion.tsv"
    elif feature_lvl == "protein":
        expected_file_name = "combined_protein.tsv"
    else:
        raise ValueError("Invalid feature level. Choose 'peptide' or 'protein'.")

    # Check if the expected file name is a substring of the path_to_input
    if expected_file_name not in path_to_input:
        raise ValueError(
            "The provided path does not seem to be an output of the ion_quant package."
        )

    # load tsv file
    input_file = pd.read_table(path_to_input, sep="\t")
    manifest_file = pd.read_table(manifest_file_path, sep="\t", header=None)
    manifest_dict = {
        f"{row[1]}_{row[2]}": row[0] for index, row in manifest_file.iterrows()
    }

    if feature_lvl == "peptide":
        expression_data = input_file.iloc[:, 17:].T
        number_of_samples = int(expression_data.shape[0] / 2)
        expression_data = expression_data.iloc[-number_of_samples:, :]
        # build feature column
        rows = input_file["Modified Sequence"]
        charges = input_file["Charge"]
        features = [a + str(b) for a, b in zip(rows, charges)]
        expression_data.columns = features
        expression_data.index = expression_data.index.str.split().str[0]
        expression_data = expression_data.rename(index=manifest_dict)

        expression_data = expression_data.T
        expression_data.index.name = "Precursor.Id"
        expression_data.reset_index(inplace=True)

    elif feature_lvl == "protein":
        expression_data = input_file.iloc[:, 14:-1].T
        number_of_samples = int(expression_data.shape[0] / 5)
        expression_data = expression_data.iloc[-number_of_samples:, :]
        features = input_file["Protein ID"]
        expression_data.columns = features
        expression_data.index = expression_data.index.str.split().str[0]
        expression_data = expression_data.rename(index=manifest_dict)

        expression_data = expression_data.T
        expression_data.index.name = "Protein.Group"
        expression_data.reset_index(inplace=True)

    else:
        print("not reachable")

    if median_normalization:
        output_expression_matrix = feature_median_normalize(expression_data, save_path)
    else:
        output_expression_matrix = expression_data
    output_expression_matrix.to_csv(path_to_output, index=False)
    return output_expression_matrix


def diann_r_code(path_to_input, path_to_output, feature_lvl):
    os.chdir(path_to_output)
    r_output_log = os.path.join(path_to_output, "r_output_log.txt")
    path_to_output = os.path.join(path_to_output, "iq_expression_table.tsv")
    if feature_lvl == "peptide":
        r_code = """
        # Your R code here
        sink("{r_output_log}")
        library(iq)
        process_long_format("{path_to_input}",
                            output_filename = "{path_to_output}",
                            filter_double_less = c("Q.Value" = "0.01", "Global.Q.Value" = "0.01"),
                            primary_id = "Precursor.Id")
        sink()
        """.format(
            r_output_log=r_output_log,
            path_to_input=path_to_input,
            path_to_output=path_to_output,
        )
    elif feature_lvl == "protein":
        r_code = """
        # Your R code here
        sink("{r_output_log}")
        library(iq)
        process_long_format("{path_to_input}",
                            output_filename = "{path_to_output}",
                            filter_double_less = c("Q.Value" = "0.01", "Global.Q.Value" = "0.01",
                            "PG.Q.Value" = "0.01", "Protein.Q.Value"="0.01", "Global.PG.Q.Value"="0.01")
                            )
        sink()
        """.format(
            r_output_log=r_output_log,
            path_to_input=path_to_input,
            path_to_output=path_to_output,
        )
    else:
        raise ValueError(
            f"feature_lvl {feature_lvl} is not supported. Please use 'peptide' or 'protein'."
        )
    full_command = f"Rscript -e '{r_code}'"
    print(
        "Running R code to build expression table based on diann search output: "
        + full_command
    )
    subprocess.run(full_command, shell=True, check=True)

    if not os.path.isfile(path_to_output):
        raise ValueError(f"iq output file {path_to_output} was not created.")


def spectronaut_r_code(path_to_input, path_to_output, feature_lvl):
    os.chdir(path_to_output)
    r_output_log = os.path.join(path_to_output, "r_output_log.txt")
    path_to_output = os.path.join(path_to_output, "iq_expression_table.tsv")
    if feature_lvl == "peptide":
        r_code = """
        # Your R code here
        sink("{r_output_log}")
        library(iq)
        sample_id  <- "R.FileName"

        #secondary_id <- c("EG.Library", "FG.Id", "FG.Charge", "F.FrgIon", "F.Charge", "F.FrgLossType")
        secondary_id <- c("R.Condition","PG.ProteinGroups", "EG.ModifiedSequence", "FG.Charge",
                          "F.FrgIon", "F.Charge", "F.PeakArea", "PG.Genes", "PG.ProteinNames")

        #annotation_columns <- c("PG.Genes", "PG.ProteinNames")

        iq_dat <- iq::fast_read("{path_to_input}",
                                sample_id  = sample_id,
                                primary_id = "EG.PrecursorId",  ##### Peptide/Precusor #####
                                secondary_id = secondary_id,
                                filter_double_less = c("PG.Qvalue" = "0.01", "EG.Qvalue" = "0.01"),
                                intensity_col = "F.PeakArea",
                                filter_string_equal = c("F.ExcludedFromQuantification" = "False"),
        )

        iq_norm_data <- iq::fast_preprocess(iq_dat$quant_table)
        result_faster <- iq::fast_MaxLFQ(iq_norm_data)

        # repair column names (files) and row names (features e.g. proteins/peptides)
        df <- result_faster$estimate
        cols <- iq_dat$sample
        rows <- iq_dat$protein[,1]
        colnames(df)<-cols
        rownames(df)<-rows[1:dim(df)[1]]
        write.csv(df,"{path_to_output}")
        sink()
        """.format(
            r_output_log=r_output_log,
            path_to_input=path_to_input,
            path_to_output=path_to_output,
        )
    elif feature_lvl == "protein":
        r_code = """
        # Your R code here
        sink("{r_output_log}")
        library(iq)
        sample_id  <- "R.FileName"

        #secondary_id <- c("EG.Library", "FG.Id", "FG.Charge", "F.FrgIon", "F.Charge", "F.FrgLossType")
        secondary_id <- c("R.Condition","PG.ProteinGroups", "EG.ModifiedSequence", "FG.Charge",
                          "F.FrgIon", "F.Charge", "F.PeakArea", "PG.Genes", "PG.ProteinNames")

        #annotation_columns <- c("PG.Genes", "PG.ProteinNames")

        iq_dat <- iq::fast_read("{path_to_input}",
                                sample_id  = sample_id,
                                primary_id = "PG.ProteinGroups", ##### Protein #####
                                secondary_id = secondary_id,
                                filter_double_less = c("PG.Qvalue" = "0.01", "EG.Qvalue" = "0.01"),
                                intensity_col = "F.PeakArea",
                                filter_string_equal = c("F.ExcludedFromQuantification" = "False"),
        )

        iq_norm_data <- iq::fast_preprocess(iq_dat$quant_table)
        result_faster <- iq::fast_MaxLFQ(iq_norm_data)

        # repair column names (files) and row names (features e.g. proteins/peptides)
        df <- result_faster$estimate
        cols <- iq_dat$sample
        rows <- iq_dat$protein[,1]
        colnames(df)<-cols
        rownames(df)<-rows[1:dim(df)[1]]
        write.csv(df,"{path_to_output}")
        sink()
        """.format(
            r_output_log=r_output_log,
            path_to_input=path_to_input,
            path_to_output=path_to_output,
        )
    else:
        raise ValueError(
            f"feature_lvl {feature_lvl} is not supported. Please use 'peptide' or 'protein'."
        )
    full_command = f"Rscript -e '{r_code}'"
    print(
        "Running R code to build expression table based on spectronaut search output: "
        + full_command
    )
    subprocess.run(full_command, shell=True, check=True)
    if not os.path.isfile(path_to_output):
        raise ValueError(f"iq output file {path_to_output} was not created.")


def is_txt_or_tsv(filename):
    _, ext = os.path.splitext(filename)
    if ext == ".txt":
        return "spectronaut"
    elif ext == ".tsv":
        # check if the data is ion_quant or diann data.
        if "diann" in filename:
            return "diann"
        elif "combined_ion" in filename or "combined_protein" in filename:
            return "ion_quant"
        else:
            print(
                f"Filename {filename} with .tsv extension does not match expected patterns for diann or ion_quant."
            )
            return None
    else:
        print(
            f"File extension {ext} is not supported. Please use .txt for spectronaut and .tsv for DIANN/Fragpipe."
        )
        return None


def build_expression_table(
    path_to_input, path_to_output, feature_lvl, manifest_file_path=None
):
    """

    Parameters
    ----------


    Returns
    -------

    """
    # check first if it is diann or spectronaut output
    filetyp = is_txt_or_tsv(path_to_input)

    if filetyp == "spectronaut":
        spectronaut_r_code(path_to_input, path_to_output, feature_lvl)
    elif filetyp == "diann":
        diann_r_code(path_to_input, path_to_output, feature_lvl)
    elif (
        filetyp == "ion_quant"
    ):  # actually we can also check if manifest_file_path is not none...
        ion_quant_code(path_to_input, path_to_output, feature_lvl, manifest_file_path)
    else:
        print("No valid input file")

    return filetyp


def parse_args():
    """

    Returns
    -------
    args : TYPE
        Default argument parser for console execution.

    """
    parser = argparse.ArgumentParser(
        description="Compute feature expression table from spectronaut or diann output"
    )
    parser.add_argument(
        "--i",
        "--path_to_input",
        type=str,
        default=os.getcwd(),
        help="path to the folder containing all input files",
    )
    parser.add_argument(
        "--o", "--path_to_output", type=str, default=os.getcwd(), help="Output path"
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
    search_software = build_expression_table(args.i, args.o, args.f)
