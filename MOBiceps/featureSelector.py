#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 12:14:16 2022
@author: Jens Settelmeier
"""

import shap
import umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
import itertools as it
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.stats.multitest import multipletests
from collections import Counter, OrderedDict
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    RandomizedSearchCV,
    GridSearchCV,
)
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn import preprocessing

from MOBiceps.modifiedYellowbrick import RFECV
from MOBiceps.modifiedYellowbrick_ROCAUC import ROCAUC
from MOBiceps.foldsConstruction import construct_folds


def adjusted_decision_threshold_prediction(roc_data_train, model, X):
    # Identify the optimal threshold
    # Calculate the distance of each point on the ROC curve from the top-left corner
    roc_data_train_copy = roc_data_train.copy()
    roc_data_train_copy["Distance"] = np.sqrt(
        (1 - roc_data_train_copy["TPR"]) ** 2 + roc_data_train_copy["FPR"] ** 2
    )
    optimal_idx = roc_data_train_copy["Distance"].idxmin()
    optimal_threshold = roc_data_train_copy.loc[optimal_idx, "Thresholds"]
    # Predict probabilities
    y_prediction_prob = model.predict_proba(X)[:, 1]
    # Adjust predictions according to the optimal threshold
    y_prediction = (y_prediction_prob >= optimal_threshold).astype(int)
    return y_prediction


def save_roc_data(visualizer, save_name="roc_data.csv"):
    # Extract FPR, TPR, and thresholds
    fpr = visualizer.fpr["micro"]
    tpr = visualizer.tpr["micro"]
    thresholds = visualizer.thresholds["micro"]

    # Create a DataFrame and save it to a CSV file
    roc_data = pd.DataFrame({"FPR": fpr, "TPR": tpr, "Thresholds": thresholds})
    roc_data.to_csv(save_name, index=False)
    return roc_data


def calculate_correlations(expression_data, key_proteins):
    # Prepare a list to hold the results
    results_list = []

    # Iterate over key proteins only if they are in the DataFrame
    for key_protein in [kp for kp in key_proteins if kp in expression_data.columns]:
        for other_protein in expression_data.columns.drop(key_protein):
            # Calculate Kendalltau's correlation and p-value
            correlation, p_value = stats.kendalltau(
                expression_data[key_protein], expression_data[other_protein]
            )

            # Append results to list
            results_list.append(
                {
                    "Key_Feature": key_protein,
                    "Other_Feature": other_protein,
                    "Correlation": correlation,
                    "P_Value": p_value,
                }
            )

    # Convert results list to DataFrame in one go
    results = pd.DataFrame(results_list)

    # Apply Benjamini-Hochberg correction
    corrected_p_values = multipletests(results["P_Value"], method="fdr_bh")[1]
    results["Adjusted_P_Value"] = corrected_p_values

    return results


def capture_sig_high_correlated_features(
    expression_mat, key_proteins, p_value_threshold=0.05, corr_threshold=0.5
):
    classes = expression_mat["class"].unique()
    classes = list(classes)
    classes.append("ALL")  # could break, if class ALL already exists
    class_specific_results = {}
    df = expression_mat.copy()
    for clasS in classes:
        if clasS == "ALL":
            df_class = df.copy()
        else:
            df_class = df[df["class"] == clasS].copy()

        # remove last column which are the classes and fill NaN with 0
        expression_data = df_class.iloc[:, :-1].fillna(
            0
        )  # could be done outside the loop?

        # compute correlations of key_proteins (new) with all other proteins in the expression matrix
        correlation_results = calculate_correlations(expression_data, key_proteins)

        # only consider corrleations above or equal to an absolut value of 0.5
        filtered_ana = correlation_results[
            abs(correlation_results.Correlation) >= corr_threshold
        ].copy()

        # only trust correlations with a adjusted p-value below 0.05
        filtered_ana = filtered_ana[
            filtered_ana["Adjusted_P_Value"] <= p_value_threshold
        ]
        class_specific_results[clasS] = filtered_ana

    return class_specific_results


def visualize_rfe_selected_features(
    X, y, feature_counter, class_col="class", filename="rfe_selected_features_votes.pdf"
):
    """Visualizes the features selected using crossvalidated_rfe

    Args:
      X (pd.DataFrame): the intensity matrix
      y (pd.DataFrame): the response vector
      feature_counter (Counter): the Counter object returned by crossvalidated_rfe
      Author: Patrick Pedrioli

    Parameters
    ----------
    class_col
    """
    # Print out the most voted features
    c = feature_counter
    print("Highest voted features:")
    for feature in c.most_common(20):
        print(feature)

    def save_barplot(data, x="feature", y="votes", filename="_barplot.pdf"):
        """
        Generates a bar plot from the given data and saves it as a .pdf file.

        Parameters:
        -----------
        data : iterable
            The input data to create the bar plot. Each inner list or tuple
            must contain two elements, corresponding to `x` and `y` values.

        x : str, optional
            The label for the x-axis of the plot. Also defines the column name for
            x-axis values in the pandas DataFrame created from `data`.
            Defaults to 'feature'.

        y : str, optional
            The label for the y-axis of the plot. Also defines the column name for
            y-axis values in the pandas DataFrame created from `data`.
            Defaults to 'votes'.

        filename : str, optional
            The name of the output .pdf file. Defaults to '_barplot.pdf'.

        Returns:
        --------
        None

        """
        plt.figure(figsize=(20, 20))
        g = sns.catplot(x=x, y=y, data=pd.DataFrame(data, columns=[x, y]), kind="bar")
        g.set_xticklabels(rotation=90)
        plt.savefig(filename, format="pdf")
        plt.close()

    def save_boxplot(
        data, x="feature", y="value", hue=class_col, filename="_boxplot.pdf"
    ):
        """
        Generates a box plot from the given data and saves it as a .pdf file.

        Parameters:
        -----------
        x : str
            The label for the x-axis of the plot. Also defines the column name for
            x-axis values in the pandas DataFrame.

        y : str
            The label for the y-axis of the plot. Also defines the column name for
            y-axis values in the pandas DataFrame.

        hue : str
            Variable in `data` to map plot aspects to different colors.

        data : DataFrame
            Input data structure that should be a pandas DataFrame where
            variables are assigned to either the x, y or hue semantic roles.

        filename : str, optional
            The name of the output .pdf file. Defaults to '_boxplot.pdf'.

        Returns:
        --------
        None
        """
        plt.figure(figsize=(20, 20))
        g = sns.boxplot(x=x, y=y, hue=hue, data=data)
        _ = g.set_xticklabels(g.get_xticklabels(), rotation=90)
        plt.savefig(filename, format="pdf")
        plt.close()

    # Save the barplot
    save_barplot(c.most_common(10), filename=filename + "_barplot.pdf")

    # Prepare the data for the boxplot
    subset_features = [name for name, count in c.most_common(10)]
    subset_mtx = X[subset_features]
    subset_mtx = subset_mtx.merge(y, right_index=True, left_index=True)
    subset_mtx_long = subset_mtx.melt(id_vars=[class_col], var_name="feature")

    # Save the boxplot
    save_boxplot(subset_mtx_long, hue=class_col, filename=filename + "_boxplot.pdf")


def dfMat22ColumnRep(df, col_1_name="col1", col_2_name="col2"):
    """
    Transforms the input DataFrame to a 2-column DataFrame where the first column
    contains the values from the original DataFrame and the second column
    contains the corresponding column names.

    Parameters:
    -----------
    df : DataFrame
        The input DataFrame to be transformed.

    col_1_name : str, optional
        The name for the first column in the output DataFrame which contains
        the values from the original DataFrame. Defaults to 'Column1'.

    col_2_name : str, optional
        The name for the second column in the output DataFrame which contains
        the column names from the original DataFrame. Defaults to 'Column2'.

    Returns:
    --------
    Df : DataFrame
        The transformed DataFrame with two columns - one with the values and
        the other with the column names from the original DataFrame.
    """
    m, n = df.shape
    values1col = df.to_numpy().T.flatten()
    col_names_array = df.columns.to_numpy()
    second_col = list()
    for j in range(n):
        second_col.extend([i for i in it.repeat(col_names_array[j], m)])

    Df = pd.DataFrame({col_1_name: values1col, col_2_name: second_col})

    return Df


def has_nan(df: pd.DataFrame) -> bool:
    """
    Check if a DataFrame has any NaN values.

    Parameters:
    df (pd.DataFrame): DataFrame to check for NaN values

    Returns:
    bool: True if there is at least one NaN value in the DataFrame, False otherwise
    """
    return df.isnull().any().any()



def expression_heatmap(df, labels, label_encoding_dic=None, filename="heatmap.pdf"):
    """
    Generates a clustered heatmap from the given DataFrame, annotates rows with class colors,
    performs hierarchical clustering, and saves it as a .pdf file.

    Parameters:
    -----------
    df : DataFrame
        The input DataFrame from which the heatmap is generated.

    labels : list or Series
        List or Pandas Series containing class labels for each row in the DataFrame.

    filename : str, optional
        The name of the output .pdf file. Defaults to 'heatmap.pdf'.

    method : str, optional
        Linkage method to use for clustering ('average', 'single', 'complete', etc.).

    metric : str, optional
        Distance metric to use for clustering ('euclidean', 'correlation', 'cityblock', etc.).

    Returns:
    --------
    ax : AxesSubplot
        An AxesSubplot object representing the generated heatmap plot.
    """
    if len(labels) != len(df):
        raise ValueError("Length of `labels` must match the number of rows in `df`.")

    # Create a color palette and map labels to colors
    sample_labels = labels.to_dict()

    labels_list = [sample_labels[x] for x in df.index]
    if label_encoding_dic is not None:
        labels_list = [label_encoding_dic[x] for x in labels_list]

    palette = sns.color_palette("hsv", len(set(labels_list)))
    color_dict = {label: color for label, color in zip(set(labels_list), palette)}
    row_colors = [color_dict[label] for label in labels_list] # Map labels to colors for each row
    # sns.set_context("talk")
    # Create the heatmap with hierarchical clustering
    g = sns.clustermap(df, row_colors=row_colors)

    # Access the position of the heatmap within the figure
    heatmap_pos = g.ax_heatmap.get_position()

    # Extract the top-right corner of the heatmap
    top_right = (heatmap_pos.x1 + 0.1, heatmap_pos.y1 + 0.15)

    # Add x-axis label for features
    g.ax_heatmap.set_xlabel('features')

    # Add title to the color bar
    colorbar = g.cax
    colorbar.set_ylabel('Feature Intensity')

    # Create a legend
    legend_title = "Sample Classes"
    legend_handles = [mpatches.Patch(color=color, label=label) for label, color in color_dict.items()]

    # Add legend relative to the top-right corner of the heatmap
    g.fig.legend(handles=legend_handles, title=legend_title, loc='upper left', bbox_to_anchor=top_right)
    g.fig.suptitle('                  Clustered Feature Expression Heatmap')
    g.fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.1, wspace=0.1)


    # Save the plot
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()

    return g.ax_heatmap


def plot_feature_correlation(X_train, most_common_features, sim, method=None):
    """
    Plots a Spearman or Kandall -correlation matrix depending on sample size
    for the most common features in the training set.
    With more than 29 samples we use spearman correlation, otherwise kendall.
    Saves the resulting cluster map as a .pdf file.

    Parameters:
    -----------
    X_train : DataFrame
        The input DataFrame containing the training data.

    most_common_features : iterable
        An iterable of the most common features in the training set.
        Each feature should be a column name in X_train.

    sim : str or int
        A specifier used to distinguish the output .pdf file.
        It is appended to the output file's name.

    method : {'spearman', 'kendall', 'pearson'}, optional

    Returns:
    --------
    tuple : (AxesSubplot, DataFrame), or None
        If more than one feature is selected, it returns a tuple containing:
            - An AxesSubplot object representing the generated clustermap plot.
            - A DataFrame of the calculated correlations between features.
        If only one feature is selected, returns None, because a correlation matrix is not meaningful.
    """
    print("### Plotting feature correlation matrix ... ###\n")
    indexes = list(np.unique(most_common_features))
    # plot correlation matrix
    print("indexes:", indexes)
    if len(indexes) > 1:
        if method is None:
            if X_train.shape[0] >= 30:
                method = "spearman"
            else:
                method = "kendall"
        print(f"Using {method} correlation method.")
        df_corr = X_train[indexes][:-1].corr(method=method)
        cluster_plot = sns.clustermap(
            df_corr,
            figsize=(25, 25),
            z_score=None,
            row_cluster=True,
            col_cluster=True,
            method="ward",
            cmap="coolwarm",
            vmax=1,
            vmin=-1,
            robust=False,
            annot=True,
            annot_kws={"size": 13},
            cbar_kws={"label": f"{method}\ncorrelation"},
        )
        cluster_plot.savefig(f"clustermap_relevant_features_{sim}.pdf", format="pdf")
        plt.close()
        return cluster_plot, df_corr
    else:
        print("Only one feature selected. Correlation matrix not meaningful.")
        return None


# add volcano tab to MOAgent. Work in progress...
def volcano_plot_wrappe(
    df_expression_data,
    class_annotation=None,
    imputation=None,
    title=None,
    test="ttest",
    log_transformed=True,
):
    # check if class annotation file is available and check if df_expression_data has class column
    # Then create A and B dataframes
    # Check if one of my imputation strategy should be applied.
    # if title is none, create meaning full title.
    # apply volcanp_plot()
    return


def volcano_plot(
    A,
    B,
    costum_title,
    FC_lo2_threshold=0.5,
    sig_niveau=0.05,
    test="ttest",
    filename="volcano_plot.pdf",
    already_log2_expressions=True,
):
    """
    Creates a volcano plot of the given data and saves it as a .pdf file.

    Parameters:
    -----------
    A, B : DataFrame
        Two DataFrames, representing two conditions, which will be compared for their difference
        in expression values.

    costum_title : str
        A string to be used in the title of the plot.

    FC_lo2_threshold : float, optional
        The log2 fold change threshold for considering a protein as significantly changed.
        Defaults to 0.5.

    sig_niveau : float, optional
        The significance level to consider for the statistical test. Defaults to 0.05.

    test : {'ttest', 'utest'}, optional
        The statistical test to use for comparing the groups. Defaults to 'ttest'.

    filename : str, optional
        The name of the output .pdf file. Defaults to 'volcano_plot.pdf'.

    Returns:
    --------
    tuple : (Series, Series)
        A tuple containing two pandas Series:
            - The -log10 transformed p-values from the statistical test.
            - The log2 fold changes.
    """
    # https://thecodingbiologist.com/posts/Making-volcano-plots-in-python-in-Google-Colab

    if already_log2_expressions:
        A = A.applymap(lambda x: np.power(2, x)).copy()
        B = B.applymap(lambda x: np.power(2, x)).copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    if test == "ttest":
        statistics, p_values = stats.ttest_ind(A.T, B.T, equal_var=False)
    elif test == "utest":
        statistics, p_values = stats.mannwhitneyu(A.T, B.T)

    _, corrected_p_vals, _, bon_fernoni_alpha = multipletests(
        p_values, alpha=0.05, method="fdr_bh"
    )
    # compute Volcano plot A to B:
    mean_A = A.apply(np.mean, axis=1)
    mean_B = B.apply(np.mean, axis=1)
    # compute fold changes. set 1 as min value
    constant_to_add = min(
        A.min().min(), B.min().min()
    )  # same strategy as msstats, to avoid devision by zero in FC computations.
    FC_A_to_B = np.log2((mean_A + constant_to_add) / (mean_B + constant_to_add))

    log_10_p = -np.log10(corrected_p_vals)

    bon_fernoni_threshold = -np.log10(
        sig_niveau
    )  # Should be renamed to adjusted threshold, otherwise miss leading.

    x_axis_p1, y_axis_p1 = (
        -2 * max(abs(FC_A_to_B.max()), abs(FC_A_to_B.min())),
        bon_fernoni_threshold,
    )
    x_axis_p2, y_axis_p2 = (
        2 * max(abs(FC_A_to_B.max()), abs(FC_A_to_B.min())),
        bon_fernoni_threshold,
    )

    x_axis_FCl1, y_axis_FCl1 = -FC_lo2_threshold, 0
    x_axis_FCl2, y_axis_FCl2 = -FC_lo2_threshold, 10

    x_axis_FCr1, y_axis_FCr1 = FC_lo2_threshold, 0
    x_axis_FCr2, y_axis_FCr2 = FC_lo2_threshold, 10

    my_colors = [
        "orange"
        if x >= bon_fernoni_threshold and abs(y) >= FC_lo2_threshold
        else "black"
        for x, y in zip(log_10_p, FC_A_to_B)
    ]

    counter = 0
    for noted_color in my_colors:
        if noted_color == "orange":
            counter += 1
    print(f"Number of enriched and significantly features: {counter}")

    if counter == 0:
        print(
            "No enriched and significantly features found. Focus on significant features."
        )
        my_colors = [
            "yellow" if x >= bon_fernoni_threshold else "black" for x in log_10_p
        ]

    ax1.scatter(FC_A_to_B, log_10_p, c=my_colors)
    ax1.axline((x_axis_p1, y_axis_p1), (x_axis_p2, y_axis_p2), c="red")
    ax1.axline((x_axis_FCl1, y_axis_FCl1), (x_axis_FCl2, y_axis_FCl2), c="gray")
    ax1.axline((x_axis_FCr1, y_axis_FCr1), (x_axis_FCr2, y_axis_FCr2), c="gray")

    feature_assignment_info = []
    for i, color in zip(range(log_10_p.shape[0]), my_colors):
        if color == "orange" or color == "yellow":
            ax1.text(
                x=FC_A_to_B[i],
                y=log_10_p[i],
                s=i,
                fontdict=dict(color="red", size=10),
                bbox=dict(facecolor="yellow", alpha=0.2),
            )
            feature_assignment_info.append(f"{i}: {A.index[i]}")
    textstr = "\n".join(f"{i}" for i in feature_assignment_info)
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax2.text(
        0.05,
        0.95,
        textstr,
        transform=ax2.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    # Hide axes and ticks
    ax2.axis("off")

    ax1.set_title(f"{test} Volcano Plot for {len(p_values)} Features. {costum_title}")
    ax1.set_xlabel("Fold Changes in log2 scale")
    ax1.set_ylabel("adjusted p values in -log10 scale")

    if counter == 0:
        one = mlines.Line2D(
            [], [], color="yellow", marker="o", ls="", label="significant features"
        )
    else:
        one = mlines.Line2D(
            [], [], color="orange", marker="o", ls="", label="Enriched and sig features"
        )
    two = mlines.Line2D(
        [], [], color="black", marker="o", ls="", label="not enriched features"
    )

    four = mlines.Line2D([], [], color="gray", marker="o", ls="", label="FC threshold")
    five = mlines.Line2D([], [], color="red", marker="o", ls="", label="FDR threshold")
    ax1.legend(handles=[one, two, four, five])
    plt.savefig(filename, format="pdf")
    plt.close()
    volcano_data = pd.DataFrame(
        {"feature": A.index, "adj_log10_p_val": log_10_p, "log2_fc": FC_A_to_B}
    )
    return volcano_data


def print_fstatistics(X_train, y_train, most_common_features):
    """
    Computes the ANOVA F-statistic for the most common features in the training set and prints the p-values.

    Parameters:
    -----------
    X_train : DataFrame
        The input DataFrame containing the training data.

    y_train : iterable
        An iterable containing the target values for the training set. Each value should correspond to a row in X_train.

    most_common_features : iterable
        An iterable of the most common features in the training set. Each feature should be a column name in X_train.

    Returns:
    --------
    sortedPvals : list of tuple
        A list of tuples where each tuple contains a feature name and its corresponding p-value,
        sorted in ascending order of p-value.
    """
    print("### F-statistic is computed and p-values will be plotted ###")
    # print p_values of features for fstatistic
    X_train = X_train[most_common_features]
    _, pvalues = f_classif(X_train, y_train)

    Xpval = {X_train.columns[i]: v for i, v in enumerate(pvalues)}
    sortedPvals = sorted(Xpval.items(), key=lambda x: x[1], reverse=False)

    print("features ANOVA-F scores (p-values):")
    for feature, pval in sortedPvals:
        print("\t", feature, ":", pval)
    return sortedPvals


def apply_logistic_regression(
    X_train,
    X_test,
    y_train,
    y_test,
    most_common_features,
    print_confusion_mat,
    random_state,
    sim,
):
    """
    Fits a Logistic Regression model to the training data, makes predictions on the train and test sets,
    and calculates various performance metrics. It optionally saves the confusion matrix plots.

    Parameters:
    -----------
    X_train, X_test : DataFrame
        The training and testing data.
        Each should be a DataFrame where each row is an observation and each column is a feature.

    y_train, y_test : Series
        The target labels for the training and testing sets. Each should be a pandas Series.

    most_common_features : iterable
        An iterable of the most common features in the training set. Each feature should be a column name in X_train.

    random_state : int
        The random seed for the logistic regression classifier.

    print_confusion_mat : bool
        Whether to print and save the confusion matrix plots.

    sim : str or int
        A specifier used to distinguish the output .pdf file. It is appended to the output file's name.

    Returns:
    --------
    tuple : (LogisticRegression, array)
        A tuple containing:
            - The fitted LogisticRegression model.
            - The predictions made by the model on the test set.
    """
    X_train_without_nan = X_train.fillna(0)
    X_test_without_nan = X_test.fillna(0)
    print("### Logistic regression as a (generalized) linear classifier is applied ###")
    # Fit logisticRegression on most common features and calculate test scores
    scaler = StandardScaler()  # (X-mean(X))/std(X)
    scaler.fit(X_train_without_nan)
    X_train_scaled = scaler.transform(X_train_without_nan)
    X_test_scaled = scaler.transform(X_test_without_nan[most_common_features])
    lr = LogisticRegression(class_weight="balanced", random_state=random_state)
    lr.fit(X_train_scaled, y_train)

    y_train_pred_prob_lr = lr.predict_proba(X_train_scaled)
    if y_train_pred_prob_lr.shape[1] == 2:
        y_train_pred_prob_lr = y_train_pred_prob_lr[:, 1]
    auc_train = roc_auc_score(y_train, y_train_pred_prob_lr, multi_class="ovo")
    print("LR ROC AUC train score:", auc_train)

    y_test_pred_prob_lr = lr.predict_proba(X_test_scaled)
    if y_test_pred_prob_lr.shape[1] == 2:
        y_test_pred_prob_lr = y_test_pred_prob_lr[:, 1]
    auc_test = roc_auc_score(y_test, y_test_pred_prob_lr, multi_class="ovo")
    print("LR ROC AUC test score:", auc_test)

    y_train_pred = lr.predict(X_train_scaled)
    y_test_pred = lr.predict(X_test_scaled)
    print(
        "LR classification train report:\n{}".format(
            classification_report(y_train, y_train_pred)
        )
    )
    print(
        "LR classification test report:\n{}".format(
            classification_report(y_test, y_test_pred)
        )
    )

    if print_confusion_mat:
        print("LR confusion matrix for train data:\n")
        disp3 = ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
        plt.savefig(f"confusion_matrix_train_{sim}.pdf", format="pdf")
        plt.close()

        print("LR confusion matrix for test data:\n")
        disp4 = ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred)
        plt.savefig(f"confusion_matrix_test_{sim}.pdf", format="pdf")
        plt.close()
    return lr, y_test_pred


def noisy_bootstrapping(X, y, print_out=True, times=3, noisy=True, prefix=None):
    """
    Perform noisy bootstrapping to increase the sample size of a dataset by duplicating existing samples with
    or without adding noise.

    Parameters:
    -----------
    X : DataFrame
        The input DataFrame containing the features.

    y : Series
        The input Series containing the target labels.

    print_out : bool, optional
        Whether to print out the number of samples after applying bootstrapping. Defaults to True.

    times : int, optional
        The number of times the original dataset should be duplicated. Defaults to 3.

    noisy : bool, optional
        Whether to perform noisy augmentation by adding noise to the bootstrapped samples. Defaults to True.

    Returns:
    --------
    tuple : (DataFrame, Series, list)
        A tuple containing:
            - The bootstrapped feature DataFrame.
            - The corresponding target Series.
            - A list containing the bootstrapped samples for each class.
    """
    new_class_samples_cont = []
    X_orig = X.copy()
    y_orig = y.copy()

    for i in range(times):
        new_class_samples = noise_aug(X_orig, y_orig, i, noise=noisy, prefix=prefix)
        new_class_samples_cont.append(new_class_samples)
        for clas in new_class_samples:
            XX = clas[list(clas.keys())[0]]
            yy = clas[list(clas.keys())[1]]

            X = pd.concat([X, XX], axis=0)
            y = pd.concat([y, yy], axis=0)
    if print_out is True:
        print("Number of samples after applying boot strapping: {}".format(X.shape[0]))
    return X, y, new_class_samples_cont


def noise_aug(X, y, i, noise, prefix=None):
    """
    Perform data augmentation by adding normally-distributed noise to the input data.

    Parameters:
    -----------
    X : DataFrame
        The input DataFrame containing the features.

    y : Series
        The input Series containing the target labels.

    i : int
        An index to append to the column names of the new noisy samples.

    Returns:
    --------
    list
        A list of dictionaries, where each dictionary corresponds to a class and contains:
            - The class name as the key, with the corresponding DataFrame of noisy samples as the value.
            - A 'label' key, with the corresponding Series of labels as the value.
            - A 'parent_samples' key, with the corresponding DataFrame of original (non-noisy) samples as the value.
    """
    if noise is True:
        noise = 1
    else:
        noise = 0
    classes = np.unique(y)
    class_noises = []
    new_class_samples = []
    for pc in classes:
        X_tmp = X[y == pc]
        mean = X_tmp.mean()
        var = X_tmp.std().apply(np.sqrt)
        noise_cont = []
        for feat_mean, feat_var in zip(mean, var):
            noise_cont.append(
                np.random.normal(feat_mean, feat_var, X_tmp.shape[0]) * noise
            )
        class_noises.append(noise_cont)
        new_samples = X_tmp.to_numpy() + np.array(noise_cont).T
        X_tmp2 = X_tmp.copy()
        X_tmp2.iloc[:, :] = new_samples.copy()
        X_tmp2 = X_tmp2.T.add_prefix(f"{prefix}_n{i + 1}_").T.copy()  # n for noisy
        y_tmp2 = y[y == pc].copy()
        y_tmp2 = y_tmp2.T.add_prefix(f"{prefix}_n{i + 1}_").T.copy()
        new_class_samples.append({pc: X_tmp2, "label": y_tmp2, "parent_samples": X_tmp})
    return new_class_samples


def cv_bootstrapping_wrapper(
    X, y, times=2, cv_folds=5, noisy_augmentation=True, prefix=None
):
    """
    Perform bootstrapping and noisy augmentation on data, then split the data into cross-validation folds.

    Parameters:
    -----------
    X : DataFrame
        The input DataFrame containing the features.

    y : Series
        The input Series containing the target labels.

    times : int, optional
        Number of times to apply the bootstrapping process. Default is 2.

    cv_folds : int, optional
        Number of cross-validation folds to split the data into. Default is 5.

    noisy_augmentation : bool, optional
        If True, apply noisy augmentation to the data during the bootstrapping process. Default is True.

    Returns:
    --------
    DataFrame
        A DataFrame of the features after bootstrapping and augmentation.

    Series
        A Series of the target labels after bootstrapping and augmentation.

    list
        A list of tuples, where each tuple corresponds to a cross-validation fold and contains:
            - The indices of the training data in the first element.
            - The indices of the testing data in the second element.
    """
    # make sure each class is represented at least twice (so with two samples)!
    X_orig = X.copy()
    y_orig = y.copy()
    X, y, out = noisy_bootstrapping(
        X, y, times=times, noisy=noisy_augmentation, prefix=prefix
    )
    y_sorted = y.sort_index()
    X_hat = [X_orig.sort_index()]
    for augmentation in out:
        curr_class = augmentation[0]["label"][0]
        X_prime = augmentation[0][curr_class]
        number_of_classes = len(augmentation)
        for pheno_class_idx in range(1, number_of_classes):
            next_class = augmentation[pheno_class_idx]["label"][0]
            next_X_prime = augmentation[pheno_class_idx][next_class]
            X_prime = pd.concat([X_prime, next_X_prime], axis=0)
        X_prime = X_prime.sort_index().copy()
        X_hat.append(X_prime)

    rng = np.random.default_rng()
    X_idxs = rng.choice(X_orig.shape[0], size=X_orig.shape[0], replace=False)
    y_skf = y_orig[X_idxs].copy()
    skf = StratifiedKFold(n_splits=cv_folds)
    cv_array = []
    for train_index, test_index in skf.split(X_idxs, y_skf):
        X_train, X_test = X_idxs[train_index], X_idxs[test_index]
        train_test_tuple = [X_train, X_test]
        cv_array.append(train_test_tuple)

    number_of_orig_samples = X_orig.shape[0]
    number_augmentations = len(out)
    bootstrapped_Folds = []
    for fold_idx in range(len(cv_array)):
        Train_fold_idxs = cv_array[fold_idx][0]
        Test_fold_idxs = cv_array[fold_idx][1]
        for augmentation_idx in range(number_augmentations):
            Train_fold_idxs = np.concatenate(
                [
                    Train_fold_idxs,
                    cv_array[fold_idx][0]
                    + number_of_orig_samples * (augmentation_idx + 1),
                ]
            )
            Test_fold_idxs = np.concatenate(
                [
                    Test_fold_idxs,
                    cv_array[fold_idx][1]
                    + number_of_orig_samples
                    * (
                        augmentation_idx + 1
                    ),  # verschiebung um die artificial samples zu finden.
                ]
            )
        bootstrapped_Folds.append((Train_fold_idxs, Test_fold_idxs))

    X_new = pd.concat(X_hat, axis=0)  # is equal to X.sort_index()
    return X_new, y_sorted, bootstrapped_Folds


def perform_RFECV(
    model,
    X,
    y,
    cv,
    step=0.1,
    min_features_to_select=1,
    scoring="roc_auc_ovo_weighted",
    index=1,
    sim=1,
):
    """
    Fit a Recursive Feature Elimination with Cross-Validation (RFECV) selector on the data, then save and close a
    plot of the cross-validation score.

    Parameters:
    -----------
    model : estimator object
        The supervised learning estimator to be fitted on the data.

    X : DataFrame
        The input DataFrame containing the features.

    y : Series
        The input Series containing the target labels.

    cv : int, cross-validation generator or an iterable
        Determines the cross-validation splitting strategy.

    step : int or float, optional
        If greater than or equal to 1, then `step` corresponds to the (integer) number of features to remove at
        each iteration. If within (0.0, 1.0), then `step` corresponds to the percentage (rounded down) of features
        to remove at each iteration. Default is 0.1.

    min_features_to_select : int, optional
        The minimum number of features to be selected. This number of features will always be scored, even if the
        difference between the current number of features and `step` is less than `step`. This parameter is ignored
        if `step` is a float. Default is 1.

    scoring : string, callable, list/tuple or dict, optional
        A single string (see :ref:`scoring_parameter`) or a callable (see :ref:`scoring`) to evaluate the predictions
        on the test set. Default is 'roc_auc_ovo_weighted'.

    index : int, optional
        The index number used to generate the name of the output plot file. Default is 1.

    sim : int, optional
        The simulation number used to generate the name of the output plot file. Default is 1.

    Returns:
    --------
    selector : RFECV object
        The fitted RFECV object.
    """
    selector = RFECV(
        model,
        step=step,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        min_features_to_select=min_features_to_select,
    )
    selector = selector.fit(X, y)
    selector.show(outpath=f"rfecv_iter_{index}_sim_{sim}_cv_{scoring}.pdf")
    plt.close()
    return selector


def optimal_fs(
    model,
    X,
    y,
    costum_folds,
    threshold=-0.1,
    counter_limits=2,
    scoring="roc_auc_ovo_weighted",
    sim=1,
):
    """
    Select the optimal number of features by performing recursive feature elimination with cross-validation (RFECV)
    and return the resulting DataFrame and its columns.

    Parameters:
    -----------
    model : estimator object
        The supervised learning estimator to be fitted on the data.

    X : DataFrame
        The input DataFrame containing the features.

    y : Series
        The input Series containing the target labels.

    costum_folds : iterable
        The custom folds for cross-validation.

    threshold : float, optional
        The threshold difference in the cross-validation score between iterations. The feature selection process
        will stop if the difference in score is below this threshold for `counter_limits` number of times. Default
        is -0.1.

    counter_limits : int, optional
        The number of iterations to tolerate for the cross-validation score difference to be below the threshold or
        for the number of selected features to not change. Default is 2.

    scoring : string, callable, list/tuple or dict, optional
        A single string (see :ref:`scoring_parameter`) or a callable (see :ref:`scoring`) to evaluate the predictions
        on the test set. Default is 'roc_auc_ovo_weighted'.

    sim : int, optional
        The simulation number used to generate the name of the output plot file in `perform_RFECV`. Default is 1.

    Returns:
    --------
    X : DataFrame
        The DataFrame after recursive feature elimination.

    columns : Index
        The columns of the resulting DataFrame.
    """
    # add minimum_features
    current_number_of_features = X.shape[1]
    if current_number_of_features <= 2:
        print(
            "#### There are only two or less features. Performing optimal_fs does not make sense. Optimal_fs returns "
            "input as output. ####\n"
        )
        return X, X.columns
    else:
        print(
            f"#### There are currently {current_number_of_features} many input features which will be reduced to the "
            f"optimal number of features. ####\n"
        )
        stopping = False
        last_score = 0
        last_selector = 0
        last_number_of_features = current_number_of_features
        counter = 0
        counter2 = 0
        selector = None
        print("#### Beginn features reduction ....\n")
        rfecv_i = 0
        while stopping is False:
            selector = perform_RFECV(
                model, X, y, costum_folds, scoring=scoring, index=rfecv_i, sim=sim
            )
            cv_max_score = max(selector.cv_scores_.mean(axis=1))
            number_features = X.iloc[:, selector.support_].shape[
                1
            ]  # selector.n_features_ would also work

            if number_features < 2:
                ranks = selector.ranking_
                ranks[ranks == 2] = 1
                ranks[ranks != 1] = 0
                X = X.iloc[:, ranks == 1].copy()
                print(
                    "### 0) There is only one features remaining. Continuing optimal_fs does not make sense and X "
                    "with the only rank 1 feature and all rank 2 features will be returned ####\n"
                )
                stopping = True
            else:
                print(
                    "score:", cv_max_score, "and number of features:", number_features
                )
                if cv_max_score - last_score < threshold:
                    print(
                        f"#### 1) Performance Score did not improve sufficiently (current score: {cv_max_score}"
                        f" vs last score {last_score}), reject current selector and use previous selector #####\n"
                    )
                    selector = last_selector
                    counter2 += 1
                    if counter2 == counter_limits:
                        print(
                            "#### 2) Stopped, because selector score did  not improve anymore #####\n"
                        )
                        stopping = True
                else:
                    counter2 = 0
                    if number_features == last_number_of_features:
                        print(
                            f"#### 3) Number of features unchanged (current number of features: {number_features} vs "
                            f"last number of features {last_number_of_features}), reject current selector and use "
                            f"previous selector #####\n"
                        )
                        selector = last_selector
                        counter += 1
                        if counter == counter_limits:
                            print(
                                "#### 4) Stopped, because number of selector features did not change anymore #####\n"
                            )
                            stopping = True
                    else:
                        print(
                            "#### 5) Number of features was reduced or performance of the selector increased ####\n"
                        )
                        print(
                            f"#### 5.1) Current number of features {number_features}, last number of features"
                            f"{last_number_of_features} ####\n"
                        )
                        print(
                            f"#### 5.2) Current score {cv_max_score}, last score {last_score} ####\n"
                        )
                        print("#### 5.3) Updating the selected features...")
                        counter = 0
                        X = X.iloc[:, selector.support_].copy()
                        last_selector = selector
                        last_score = cv_max_score
                        last_number_of_features = number_features
            rfecv_i += 1

        print("X shape before return: ", X.shape)
        return X, X.columns


def auc_crossvalidated_rfe(
    X_train,
    y_train,
    model,
    n_splits=6,
    n_repeats=2,
    random_state=42,
    n_features_to_select=30,
    step=100,
    observe_training=False,
    custome_folds=None,
    auc_threshold=0.7,
    sim=0,
):
    """
    Perform recursive feature elimination (RFE) on the model with cross-validation,
    and return a Counter object tracking the number of times each feature is selected.

    Parameters:
    -----------
    X_train : DataFrame
        The input DataFrame containing the training features.

    y_train : Series
        The input Series containing the training labels.

    model : estimator object
        The supervised learning estimator to be fitted on the training data.

    n_splits : int, optional
        The number of splits for cross-validation. Default is 6.

    n_repeats : int, optional
        The number of times to repeat cross-validation. Default is 2.

    random_state : int, optional
        The seed of the pseudo random number generator. Default is 42.

    n_features_to_select : int, optional
        The number of features to select in RFE. Default is 30.

    step : int or float, optional
        The number (or percentage if less than 1) of features to remove at each iteration. Default is 100.

    observe_training : bool, optional
        If True, generate classification reports and confusion matrices during the process. Default is False.

    custome_folds : iterable of (train, test) pairs, optional
        The custom folds for cross-validation. If None, a RepeatedStratifiedKFold object will be created. Default is None.

    auc_threshold : float, optional
        The threshold for the area under the ROC curve (AUC) score to consider the selected features. Default is 0.7.

    sim : int, optional
        The simulation number used to generate the names of the output plot files. Default is 0.

    Returns:
    --------
    c : Counter
        The Counter object tracking the number of times each feature is selected.
    """
    # further filtering
    if auc_threshold is None:
        auc_threshold = 1 / len(np.unique(y_train)) * np.sqrt(3)

    c = Counter()
    round = 0
    cv_auc = []

    iterator_folds = custome_folds
    try:
        iter_test = iter(
            custome_folds
        )  # doesn't need to be asigned. If that works, the for loop will work. If not except will save the day.
        print("Using custome folds")
    except TypeError:
        print("Using RepeatedStratifiedKFold to build folds (no costum folds provided)")
        rskf = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
        iterator_folds = rskf.split(X_train, y_train)

    for train_index, test_index in iterator_folds:
        round += 1
        print("Round: {} of {}".format(round, n_splits * n_repeats))
        X_train_rskf, y_train_rskf = X_train.iloc[train_index], y_train[train_index]

        X_test_rskf, y_test_rskf = X_train.iloc[test_index], y_train[test_index]

        selector = RFE(
            estimator=model, n_features_to_select=n_features_to_select, step=step
        )

        selector = selector.fit(X_train_rskf, y_train_rskf)

        y_test_pred_prob_sel = selector.predict_proba(X_test_rskf)

        if (
            y_test_pred_prob_sel.shape[1] == 2
        ):  # necessary for binary classification since roc_auc_score expects (n_sample,), if len(selector.classes_)==2
            y_test_pred_prob_sel = y_test_pred_prob_sel[:, 1]
        auc = roc_auc_score(y_test_rskf, y_test_pred_prob_sel, multi_class="ovo")

        cv_auc.append(auc)

        if observe_training is True:
            y_train_pred_selector = selector.predict(X_train_rskf)
            y_test_pred_selector = selector.predict(X_test_rskf)
            # classification report
            print(
                "Selector train classification report\n",
                classification_report(y_train_rskf, y_train_pred_selector),
            )
            print(
                "Selector test classification report\n",
                classification_report(y_test_rskf, y_test_pred_selector),
            )
            # confusion matrix
            print("Selector confusion matrix for train data:\n")
            disp = ConfusionMatrixDisplay.from_predictions(
                y_train_rskf, y_train_pred_selector
            )
            plt.savefig(f"confusion_matrix_selector_train_{sim}.pdf", format="pdf")
            plt.close()
            print("Selector confusion matrix for test data:\n")
            disp2 = ConfusionMatrixDisplay.from_predictions(
                y_test_rskf, y_test_pred_selector
            )
            plt.savefig(f"confusion_matrix_selector_test_{sim}.pdf", format="pdf")
            plt.close()

        print("ROC AUC: {}".format(auc))
        if auc > auc_threshold:
            c.update(X_train.columns[selector.support_])

    print(
        "Mean ROC AUC over {} rounds: {} (+/-{}):".format(
            round, np.mean(cv_auc), np.std(cv_auc)
        )
    )
    return c


def applyUMAP(X, y, runs_proteins, label_encoding_df=None, title="Final"):
    if label_encoding_df is not None:
        class_dict = label_encoding_df.set_index("class")["label"].to_dict()

    number_classes = len(np.unique(y))
    unique_classes = np.unique(y)

    print("### Plotting UMAP ###\n")
    colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
    if len(colors) < number_classes:
        print("Number of classes exceeds number of distinct colors!")

    # UMAP limited to selected features
    feature_space_dim = len(np.unique(runs_proteins))
    umap_model = umap.UMAP(n_components=2)
    X_umap = umap_model.fit_transform(X[np.unique(runs_proteins)])

    # Save the 2D coordinates from UMAP for selected features
    X_umap_df = pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2"])
    X_umap_df.to_csv(f"2D_coordinates_umap_selected_features_{title}.csv")

    color_array = []
    for label in y:
        for i, clas in enumerate(unique_classes):
            if label == clas:
                color_array.append(colors[i])
                break

    plt.figure(figsize=(20, 20))
    plt.title(
        f"{title}: UMAP of expression data with the {feature_space_dim} phenotype relevant features"
    )
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    legend = []
    for element in range(number_classes):
        if label_encoding_df is not None:
            class_names_in_legend = class_dict[element]
        else:
            class_names_in_legend = unique_classes[element]
        legend.append(
            mlines.Line2D(
                [],
                [],
                color=colors[element],
                marker="o",
                ls="",
                label=class_names_in_legend,
            )
        )
    plt.legend(handles=legend)
    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=color_array)
    plt.savefig(f"umap_selected_features_{title}.pdf", format="pdf")
    plt.close()

    if feature_space_dim != X.shape[1]:
        # UMAP taking all features into account
        feature_space_dim = X.shape[1]
        umap_model2 = umap.UMAP(n_components=2)
        X_umap2 = umap_model2.fit_transform(X)

        # Save the 2D coordinates from UMAP for all features
        X_umap2_df = pd.DataFrame(X_umap2, columns=["UMAP1", "UMAP2"])
        X_umap2_df.to_csv(f"2D_coordinates_umap_all_features_{title}.csv")

        color_array = []
        for label in y:
            for i, clas in enumerate(unique_classes):
                if label == clas:
                    color_array.append(colors[i])
                    break
        plt.figure(figsize=(20, 20))
        plt.title(
            f"{title}: UMAP of expression data with all {feature_space_dim} features"
        )
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        legend = []
        for element in range(number_classes):
            if label_encoding_df is not None:
                class_names_in_legend = class_dict[element]
            else:
                class_names_in_legend = unique_classes[element]
            legend.append(
                mlines.Line2D(
                    [],
                    [],
                    color=colors[element],
                    marker="o",
                    ls="",
                    label=class_names_in_legend,
                )
            )
        plt.legend(handles=legend)
        plt.scatter(X_umap2[:, 0], X_umap2[:, 1], c=color_array)
        plt.savefig(f"umap_all_features_{title}.pdf", format="pdf")
        plt.close()
    else:
        print("UMAP not applied since all of X available features are selected!")


def applyPCA(X, y, runs_proteins, label_encoding_df=None, title="Final"):
    if label_encoding_df is not None:
        class_dict = label_encoding_df.set_index("class")["label"].to_dict()
    number_classes = len(np.unique(y))
    unique_classes = np.unique(y)
    feature_names_selected = np.unique(runs_proteins)

    print("### Plotting PCAs ###\n")
    # modularize the pca part!
    colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
    if len(colors) < number_classes:
        print("Number of classes exceeds number of distinct colors!")

    # PCA limited to selected features
    feature_space_dim = len(np.unique(runs_proteins))
    pca = PCA(n_components=2, svd_solver="full")
    X_pca = pca.fit_transform(X[feature_names_selected])

    # Save the PCA components for selected features
    pca_components_df_selected = pd.DataFrame(
        pca.components_.T, index=feature_names_selected, columns=["PC1", "PC2"]
    )
    pca_components_df_selected.to_csv(f"PCA_loadings_selected_features_{title}.csv")
    # Save the PCA PC1 and PC2 coordinates from the plot.
    X_pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    X_pca_df.to_csv(f"2D_coordinates_selected_features_{title}.csv")

    color_array = []
    for label in y:
        for i, clas in enumerate(unique_classes):
            if label == clas:
                color_array.append(colors[i])
                break
    plt.figure(figsize=(20, 20))
    plt.title(
        f"{title}: PCA of expression data with the {feature_space_dim} phenotype relevant features"
    )
    plt.xlabel(
        f"PC1, total variance contribution {round(pca.explained_variance_ratio_[0], 2)}"
    )
    plt.ylabel(
        f"PC2 total variance contribution {round(pca.explained_variance_ratio_[1], 2)}"
    )
    legend = []
    for element in range(number_classes):
        if label_encoding_df is not None:
            class_names_in_legend = class_dict[element]
        else:
            class_names_in_legend = unique_classes[element]

        legend.append(
            mlines.Line2D(
                [],
                [],
                color=colors[element],
                marker="o",
                ls="",
                label=class_names_in_legend,
            )
        )
    plt.legend(handles=legend)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=color_array)
    plt.savefig(f"pca_selected_features_{title}.pdf", format="pdf")
    plt.close()

    if feature_space_dim != X.shape[1]:
        # PCA taking all features into account
        feature_space_dim = X.shape[1]
        pca2 = PCA(n_components=2, svd_solver="full")
        X_pca2 = pca2.fit_transform(X)

        # Save the PCA components for all features
        feature_names_all = X.columns
        pca2_components_df = pd.DataFrame(
            pca2.components_.T, index=feature_names_all, columns=["PC1", "PC2"]
        )
        pca2_components_df.to_csv(f"PCA_loadings_all_features_{title}.csv")
        # Save the PCA components for all features
        X_pca2_df = pd.DataFrame(X_pca2, columns=["PC1", "PC2"])
        X_pca2_df.to_csv(f"2D_coordinates_all_features_{title}.csv")

        color_array = []
        for label in y:
            for i, clas in enumerate(unique_classes):
                if label == clas:
                    color_array.append(colors[i])
                    break
        plt.figure(figsize=(20, 20))
        plt.title(
            f"{title}: PCA of expression data with all {feature_space_dim} features"
        )
        plt.xlabel(
            f"PC1, total variance contribution {round(pca2.explained_variance_ratio_[0], 2)}"
        )
        plt.ylabel(
            f"PC2 total variance contribution {round(pca2.explained_variance_ratio_[1], 2)}"
        )
        legend = []
        for element in range(number_classes):
            if label_encoding_df is not None:
                class_names_in_legend = class_dict[element]
            else:
                class_names_in_legend = unique_classes[element]
            legend.append(
                mlines.Line2D(
                    [],
                    [],
                    color=colors[element],
                    marker="o",
                    ls="",
                    label=class_names_in_legend,
                )
            )
        plt.legend(handles=legend)
        plt.scatter(X_pca2[:, 0], X_pca2[:, 1], c=color_array)
        plt.savefig(f"pca_all_features_{title}.pdf", format="pdf")
        plt.close()
    else:
        print("PCA not applied since all of X available features are selected!")


# for now only median imputation is available, but it could be an additional parameter, such that mean can be used.
def _imputationForStandardStatisticalTests(X, y, class_nan_impute="dataset_min"):
    df = X.copy()
    df["class"] = y.copy()
    unique_classes = np.unique(y)
    cols_to_impute = df.columns.difference(["class"])
    constant_impute = df[cols_to_impute].min().min()
    for cls in unique_classes:
        mean_values = df.loc[df["class"] == cls, cols_to_impute].mean()
        if mean_values.isna().any():
            # consider to set the value to the minimum feature value of the feature accross all classes or min value of the whole dataset.

            if class_nan_impute == "dataset_min":
                print(
                    f"No non-NaN values in class {cls} for columns {mean_values.index[mean_values.isna()]}. Set values "
                    f"to minimum value of the whole dataset."
                )
                mean_values[mean_values.isna()] = constant_impute
            elif class_nan_impute == "feature_min":  # not yet tested, maybe buggy.
                print(
                    f"No non-NaN values in class {cls} for columns {mean_values.index[mean_values.isna()]}. Set values "
                    f"to minimum feature value of the whole dataset."
                )
                mean_values[mean_values.isna()] = df.loc[
                    mean_values.index[mean_values.isna()]
                ].min()
            elif class_nan_impute == "zero":
                print(
                    f"No non-NaN values in class {cls} for columns {mean_values.index[mean_values.isna()]}. Set values "
                    f"to zero."
                )
                mean_values[mean_values.isna()] = 0
            else:
                raise ValueError(
                    f"Unknown class_nan_impute parameter {class_nan_impute}."
                )
        df.loc[df["class"] == cls, cols_to_impute] = (
            df[df["class"] == cls].fillna(mean_values).copy()
        )

    return df


class CustomPipeline(Pipeline):
    """Custom pipeline class to enable feature importance extraction for example in RFECV"""

    @property
    def feature_importances_(self):
        return self.named_steps["classifier"].feature_importances_


# modified feature selection function
def robust_crossvalidated_rfe4(
    X,
    y,
    patient_file_annotations=None,
    max_number_of_features=15,
    n_splits=6,  # default 6
    n_repeats=2,  # default 2
    random_state=None,
    n_features_to_select=30,
    step=100,
    boot_strapping=False,  # default False
    test_size=0.2,
    observe_training=False,
    n_iter_grid=20,  # default 15
    experiment_simulations=5,  # default 3
    print_confusion_mat=True,
    perform_add_LR=False,
    print_pca=True,
    print_umap=True,
    print_p_values=True,
    print_expression_heatmap=True,
    print_corr_hp=True,
    print_feature_votes=True,
    print_volcanos=True,
    compute_shap=True,
    test="utest",
    do_selection=True,
    use_fstatistic=False,  # deaultf false
    scoring="roc_auc_ovo_weighted",
    imputation="mean",
    gpu=False,
    noisy_augmentation=True,
    do_optimal_selection=True,
):
    """Crossvalidate Recursive Feature Elimination.
    Authors: Patrick Pedrioli, Jens Settelmeier
    robust for class imbalanced data
    applies boot strapping for little data samples
    Uses a Repeated Stratified KFold to crossvalidate features selected via RFE.

    Args: X (pd.DataFrame): the intensity matrix. y (pd.DataFrame): the response vector. n_splits (int): n_splits for
    RepeatedStratifiedKFold. n_repeats (int): n_repeats for RepeatedStratifiedKFold. random_state (float):
    random_state for RepeatedStratifiedKFold. estimator (sklearn.Estimator): the estimator to be used in by RFE:
    Extra Trees or Random Forest are possible n_features_to_select (int): n_features_to_select for RFE. step (int):
    step for RFE. accuracy_threshold (float): minimum round accuracy that needs to be achieved in order to include
    the selected features in the Counter object. boot_strapping: if True, boot strapping is applied by recursively
    appending the dataframe to itself int(np.log2(512/X.shape[0])) times (where X.shape[0] is the number of samples)

    Return:
      Counter object of selected features.
    """
    print(
        "#### XGBoost requires features which do not contain [, ] or < which is the case for "
        "modification annotations. So we replace the brackets with { and }"
    )
    X.columns = X.columns.str.replace("[", "{")
    X.columns = X.columns.str.replace("]", "}")

    plt.switch_backend("Agg")
    le = preprocessing.LabelEncoder()
    y[:] = le.fit_transform(y)
    y = y.astype(int)
    label_encoding_df = pd.DataFrame(
        {"label": le.classes_, "class": le.transform(le.classes_)}
    )
    label_encoding_df.to_csv("label_encoding.csv", index=False)

    print("Start robust crossvalidated RFE core function...")

    print("Number of total features: {}".format(X.shape[1]))
    if max_number_of_features is None:
        max_number_of_features = X.shape[1]
    unique_classes = np.unique(y)
    number_classes = len(unique_classes)

    # class dependent imputation for traditional statistical analysis
    df = _imputationForStandardStatisticalTests(X, y)
    df_y = y.copy()
    print("#### df after imputation", df.isna().any().any())

    print(f"####### boot_strapping: {boot_strapping}")

    _, _, X, y, X_test_final, y_test_final, _ = construct_folds(
        X,
        y,
        patient_file_annotations,
        random_state=random_state,
        test_size=test_size,
        n_splits=n_splits,
        n_repeats=n_repeats,
    )

    if patient_file_annotations is not None:
        patient_file_annotations = patient_file_annotations[
            patient_file_annotations.index.isin(X.index)
        ].copy()
    else:
        print(
            "### (Patient) sample annotations not available. Assuming information leakage due to replicates is "
            "not applicable ... ###\n"
        )

    X_train_final = X.copy()
    y_train_final = y.copy()

    runs_auc_values = []
    runs_model = []
    runs_proteins = []
    runs_classification_reports = []
    for sim in range(experiment_simulations):
        # make sure we do not have a bias by the order of features in the columns and shuffle the columns of X
        # Note the sample order will be shuffled during constructions of the folds.

        cols_to_shuffle = (
            X.columns
        )  # do not shuffle the first column, which is the filename column
        shuffled_cols = np.random.permutation(cols_to_shuffle)

        # Assign back to dataframe
        X = X.loc[:, list(shuffled_cols)]

        print(f"### Start simulation {sim + 1} of {experiment_simulations}. ###\n")
        # make sure all files corresponding to a patient are in train, or test split.

        (
            _,
            folds,
            X_train,
            y_train,
            X_test,
            y_test,
            patient_file_annotations,
        ) = construct_folds(
            X,
            y,
            patient_file_annotations,
            random_state=random_state,
            test_size=test_size,
            n_splits=n_splits,
            n_repeats=n_repeats,
        )

        print("Number of samples used for training: {}".format(X_train.shape[0]))
        print(
            "Class representation in training set:\n{}".format(y_train.value_counts())
        )

        print(
            "Number of samples to compute final test score: {}".format(X_test.shape[0])
        )
        print(
            "Class representation in final test set:\n{}".format(y_test.value_counts())
        )

        if boot_strapping is True:
            print(
                f"### Apply boot strapping with noise = {noisy_augmentation} within the data folds to augmentate data ###\n"
            )

            updated_folds_dfs_X_train = []
            updated_folds_dfs_y_train = []
            updated_folds_dfs_X_val = []
            updated_folds_dfs_y_val = []
            for f, fold in enumerate(folds):
                X_train_tmp = X_train.iloc[fold[0]]
                y_train_tmp = y_train.iloc[fold[0]]

                X_train_pre, y_train_pre, bootstrapped_Folds = cv_bootstrapping_wrapper(
                    X_train_tmp,
                    y_train_tmp,
                    noisy_augmentation=noisy_augmentation,
                    prefix=f + 1,
                )  # we ignore the default bootstrapped folds and adjust the indexes of the original folds.
                y_train_pre = y_train_pre.loc[X_train_pre.index]
                updated_folds_dfs_X_train.append(X_train_pre)
                updated_folds_dfs_y_train.append(y_train_pre)
                updated_folds_dfs_X_val.append(X_train.iloc[fold[1]])
                updated_folds_dfs_y_val.append(y_train.iloc[fold[1]])

            X_train_updated = pd.concat(updated_folds_dfs_X_train)
            X_train_updated_reset = X_train_updated.reset_index()
            X_train_updated_no_duplicates = X_train_updated_reset.drop_duplicates(
                subset="files", keep="first"
            )
            X_train_updated_final = X_train_updated_no_duplicates.set_index("files")
            X_train_updated = X_train_updated_final

            y_train_updated = pd.concat(updated_folds_dfs_y_train)
            y_train_updated_reset = y_train_updated.reset_index()
            y_train_updated_no_duplicates = y_train_updated_reset.drop_duplicates(
                subset="files", keep="first"
            )
            y_train_updated_final = y_train_updated_no_duplicates.set_index("files")
            y_train_updated = y_train_updated_final["class"]

            updated_folds = []
            for f, fold in enumerate(
                zip(updated_folds_dfs_X_train, updated_folds_dfs_X_val)
            ):
                train_samples = fold[0]
                val_samples = fold[1]

                # Get the matching indices
                matching_indices_train = X_train_updated.index.intersection(
                    train_samples.index
                )
                matching_locations_train = [
                    X_train_updated.index.get_loc(idx) for idx in matching_indices_train
                ]

                matching_indices_val = X_train_updated.index.intersection(
                    val_samples.index
                )
                matching_locations_val = [
                    X_train_updated.index.get_loc(idx) for idx in matching_indices_val
                ]

                updated_folds.append([matching_locations_train, matching_locations_val])

            folds = updated_folds
            X_train = X_train_updated
            y_train = y_train_updated

        y_encoded = y_train  # since last update, I already encode labels from beginning on, sinc XGBboost need encoded labels and also other classifiers need it if we have only 2 classes for the AUC.

        X_train_of_sim_all_features = X_train.copy()

        if gpu is True:
            model = XGBClassifier(tree_method="gpu_hist")
        else:
            model = XGBClassifier()

        if do_optimal_selection is True:
            X_train, support = optimal_fs(
                model, X_train, y_encoded, folds, scoring=scoring, sim=sim
            )
            # decision threshold based on roc auc

            print(
                "Number of optimal features in the sense to be the smallest subset of all features with at most "
                "marginal compromiss in loosing F1 and AUC performance:",
                X_train.shape[1],
            )
            print("write to file...")  # write these features to file

            optimal_features = (
                X_train.columns
            )  # how well will all optimal features perform? # should be same to "support"
            of_df = pd.DataFrame(
                optimal_features, columns=["optimal features by RFECV"]
            )
            of_df.to_csv(f"optimal_features_{sim}.csv")

        most_common_features = X_train.columns

        if do_selection is True:

            c = auc_crossvalidated_rfe(
                X_train, y_encoded, model, custome_folds=folds, sim=sim
            )

            most_common_features = []

            cc = c.most_common(max_number_of_features)
            for feature in cc:
                most_common_features.append(feature[0])
            most_common_features_df = pd.DataFrame(
                most_common_features,
                columns=["stage_2_forced_reduced_features_with_best_performance"],
            )
            most_common_features_df.to_csv(
                f"stage_2_forced_reduced_features_with_best_performance{sim}.csv"
            )
            print("rfe features", most_common_features)
            X_train = X_train[
                most_common_features
            ]  # should maybe be renamed to X_train_most_common_features...

        if perform_add_LR is True:
            apply_logistic_regression(
                X_train,
                X_test,
                y_train,
                y_test,
                most_common_features,
                print_confusion_mat=True,
                random_state=42,
                sim=sim,
            )

        model2 = XGBClassifier()
        grid_values_tree = {
            "n_estimators": [100, 200, 500],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 6],
            # 'subsample': [0.6, 1],
            # 'colsample_bytree': [0.6, 0.7, 0.8, 1],
            # 'gamma': [0, 0.1, 0.2],
            # 'min_child_weight': [1, 5, 10],
            "reg_alpha": [0, 0.1, 1],
            "reg_lambda": [0, 0.1, 1],
        }

        print(
            "### Hyper-parameter grid for non linear classifier:",
            grid_values_tree,
        )

        grid2 = RandomizedSearchCV(
            cv=folds,
            n_iter=n_iter_grid,
            estimator=model2,
            n_jobs=-1,
            param_distributions=grid_values_tree,
            refit=scoring,
            scoring=scoring,
        )

        grid2.fit(X_train, y_train)
        X_test_most_common_features = X_test[most_common_features].copy()

        # show train ROC AUC
        print("Train ROC \n")
        # modularize this as create_ROC_curves(fit_input_X, fit_input_y, score_input_X, score_input_Y, label_encoder, model, sim, mode)
        visualizer = ROCAUC(
            grid2.best_estimator_, is_fitted=True, encoder=le, classes=le.classes_
        )  # , classes=[-1,1]) # does it work?
        visualizer.fit(X_train, y_train)
        visualizer.score(X_train, y_train)
        visualizer.show(outpath=f"grid2_train_ROC_AUC_relevant_features_{sim}.pdf")
        roc_data_train = save_roc_data(
            visualizer, save_name=f"grid2_train_ROC_AUC_data_{sim}.csv"
        )
        plt.close()

        print("Test ROC \n")
        visualizer = ROCAUC(
            grid2.best_estimator_, is_fitted=True, encoder=le, classes=le.classes_
        )
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test_most_common_features, y_test)
        visualizer.show(outpath=f"grid2_test_ROC_AUC_relevant_features_{sim}.pdf")
        roc_data_test = save_roc_data(
            visualizer, save_name=f"grid2_test_ROC_AUC_data_{sim}.csv"
        )
        plt.close()

        y_train_pred_grid2 = adjusted_decision_threshold_prediction(
            roc_data_train, grid2, X_train
        )
        test_pred = adjusted_decision_threshold_prediction(
            roc_data_train, grid2, X_test_most_common_features
        )

        print("Grid2 best AUC cv score on training set: {}".format(grid2.best_score_))
        print(
            "Grid2 train classification report\n",
            classification_report(y_train, y_train_pred_grid2),
        )
        print(
            "Grid2 test classification report\n",
            classification_report(y_test, test_pred),
        )

        y_train_pred_prob_grid2 = grid2.predict_proba(X_train)
        if (
            y_train_pred_prob_grid2.shape[1] == 2
        ):  # necessary for binary classification since roc_auc_score expects (n_sample,), if len(selector.classes_)==2
            y_train_pred_prob_grid2 = y_train_pred_prob_grid2[:, 1]

        auc_train = roc_auc_score(y_train, y_train_pred_prob_grid2, multi_class="ovo")
        print("Grid2 ROC AUC train score:", auc_train)

        y_test_pred_prob_grid2 = grid2.predict_proba(X_test_most_common_features)
        if (
            y_test_pred_prob_grid2.shape[1] == 2
        ):  # necessary for binary classification since roc_auc_score expects (n_sample,), if len(selector.classes_)==2
            y_test_pred_prob_grid2 = y_test_pred_prob_grid2[:, 1]
        auc_test = roc_auc_score(y_test, y_test_pred_prob_grid2, multi_class="ovo")
        print("Grid2 ROC AUC test score:", auc_test)

        # here we need to impute also X_test?? -no
        if compute_shap is True:
            explainer = shap.TreeExplainer(grid2.best_estimator_)
            shap_values_train = explainer.shap_values(X_train)
            print("SHAP values for train data:\n")
            shap.summary_plot(
                shap_values_train, X_train, max_display=min(20, X_train.shape[1])
            )
            plt.savefig(f"shap_train_{sim}.pdf", format="pdf")
            plt.close()

            shap_values_test = explainer.shap_values(X_test_most_common_features)
            print("SHAP values for test data:\n")
            shap.summary_plot(
                shap_values_test,
                X_test_most_common_features,
                max_display=min(20, X_test_most_common_features.shape[1]),
            )
            plt.savefig(f"shap_test_{sim}.pdf", format="pdf")
            plt.close()

        # Note: this confusion matrix is computed on the default decision threshold of the model which might not be
        # optimal. The optimal decision threshold is usually depending on the specific context of the use case
        # A general trade-off could be the used decision threshold that achieved best trade off True Positive Rate
        # and False Positive Rate - which is the most top left point on the roc.
        if print_confusion_mat is True:
            class_dict = label_encoding_df.set_index("class")["label"].to_dict()
            class_names = [class_dict[i] for i in sorted(class_dict.keys())]

            print("Grid2 confusion matrix for train data:\n")
            disp5 = ConfusionMatrixDisplay.from_predictions(
                y_train, y_train_pred_grid2, display_labels=class_names
            )
            plt.savefig(f"confusion_matrix_Grid2_train_{sim}.pdf", format="pdf")
            plt.close()
            print("Grid2 confusion matrix for test data:\n")
            disp6 = ConfusionMatrixDisplay.from_predictions(
                y_test, test_pred, display_labels=class_names
            )
            plt.savefig(f"confusion_matrix_Grid2_test_{sim}.pdf", format="pdf")
            plt.close()

        #### Apply classical statistical approaches which require imputation methods.

        df_within_sim_tmp = _imputationForStandardStatisticalTests(
            X_train_of_sim_all_features, y_train
        )
        X_train_of_sim_all_features = df_within_sim_tmp.drop("class", axis=1).copy()
        df_within_sim_tmp = _imputationForStandardStatisticalTests(X_test, y_test)
        X_test = df_within_sim_tmp.drop("class", axis=1).copy()
        df_within_sim_tmp = _imputationForStandardStatisticalTests(X_train, y_train)
        X_train = df_within_sim_tmp.drop("class", axis=1).copy()

        if print_p_values is True:
            print_fstatistics(X_train, y_train, most_common_features)

        if print_corr_hp is True:
            plot_feature_correlation(
                X_train, most_common_features, sim, method="kendall"
            )

        if do_selection is True:
            noone_should_use_counters = []
            for t in cc:
                for prot in most_common_features:
                    if t[0] == prot:
                        for i in range(t[1]):
                            noone_should_use_counters.append(prot)
        else:
            noone_should_use_counters = list(most_common_features)

        # apply PCA to check how linear separable the data is using the most common features
        applyPCA(
            X_train_of_sim_all_features,
            y_train,
            noone_should_use_counters,
            label_encoding_df=label_encoding_df,
            title=f"PCA_train_Sim_{sim}",
        )
        applyPCA(
            X_test,
            y_test,
            noone_should_use_counters,
            label_encoding_df=label_encoding_df,
            title=f"PCA_test_Sim_{sim}",
        )  # note X_test has all features, X_train is already reduced to key features. That's why a copy X_train_of_sim_all_features exists.
        # apply UMAP to check how (non-linear) separable the data is using the most common features
        applyUMAP(
            X_train_of_sim_all_features,
            y_train,
            noone_should_use_counters,
            label_encoding_df=label_encoding_df,
            title=f"UMAP_train_Sim_{sim}",
        )
        applyUMAP(
            X_test,
            y_test,
            noone_should_use_counters,
            label_encoding_df=label_encoding_df,
            title=f"UMAP_test_Sim_{sim}",
        )

        runs_proteins.extend(noone_should_use_counters)
        runs_classification_reports.append(
            classification_report(y_test, test_pred, output_dict=True)
        )
        runs_auc_values.append(auc_test)
        runs_model.append(grid2)

    print("Simulations completed...")

    ### Outside of simulation loop  ###
    ccc = Counter(runs_proteins).most_common()

    X = df.drop(
        "class", axis=1
    ).copy()  # traditional approaches can not handle missing values ### use gaussian imputation instead?
    y = df_y.copy()

    if print_feature_votes is True:
        print(
            f"### Bar plot of selected features within {experiment_simulations}, where each simulation performs a"
            f"{n_repeats}-repeated stratified {n_splits}-cross validation for the selections. ###\n"
        )
        prot_tmp = []
        for el in ccc:
            for f in range(el[1]):
                prot_tmp.append(el[0])
        visualize_rfe_selected_features(
            X_test,
            y_test,
            Counter(prot_tmp),
            filename="relevant_feature_votes_final.pdf",
        )

    if print_volcanos is True:
        print(
            f"### Plotting {number_classes} many Volcano plots, each showing one specific vs rest ###\n"
        )
        for phenotype_class in np.unique(y):
            label_4_phenotype_class = label_encoding_df[
                label_encoding_df["class"] == phenotype_class
            ]["label"].iloc[0]
            A = X[y == phenotype_class][np.unique(runs_proteins)].T.copy()
            B = X[y != phenotype_class][np.unique(runs_proteins)].T.copy()
            costum_title = f"Class {phenotype_class}: {label_4_phenotype_class} vs rest"
            volcano_data = volcano_plot(
                A,
                B,
                costum_title,
                test=test,
                filename=f"volcano_{label_4_phenotype_class}.pdf",
                already_log2_expressions=True,
            )

            volcano_data.to_csv(
                f"volcano_data_{label_4_phenotype_class}_vs_rest.csv", index=False
            )

    if print_corr_hp is True:
        print(
            "### Plotting pearson correlation using most relevant features and all data ###\n"
        )
        plot_feature_correlation(
            X,
            np.unique(runs_proteins),
            sim="most_relevant_features_final_all_data",
            method="kendall",
        )

    if print_pca is True:
        applyPCA(X, y, runs_proteins, label_encoding_df=label_encoding_df)

    if print_umap is True:
        applyUMAP(X, y, runs_proteins, label_encoding_df=label_encoding_df)

    final_features = np.unique(runs_proteins)
    print(f"Final {len(final_features)} features:", final_features)

    X_train_final = X_train_final[final_features]
    X_test_final = X_test_final[final_features]

    final_XGB_model = XGBClassifier()
    grid_final_model_parameter = {
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 6],
        # 'subsample': [0.6, 1],
        # 'colsample_bytree': [0.6, 0.7, 0.8, 1],
        # 'gamma': [0, 0.1, 0.2],
        # 'min_child_weight': [1, 5, 10],
        "reg_alpha": [0, 0.1, 1],
        "reg_lambda": [0, 0.1, 1],
    }
    grid_final_model = GridSearchCV(
        estimator=final_XGB_model,
        param_grid=grid_final_model_parameter,
        scoring=scoring,
        refit=scoring,
        cv=5,
    )
    grid_final_model.fit(X_train_final, y_train_final)

    final_XGB_visualizer = ROCAUC(
        grid_final_model.best_estimator_,
        is_fitted=True,
        encoder=le,
        classes=le.classes_,
    )
    final_XGB_visualizer.fit(X_train_final, y_train_final)
    final_XGB_visualizer.score(X_train_final, y_train_final)
    final_XGB_visualizer.show(
        outpath="roc_auc_final_XGBclassifer_golden_features_train_performance.pdf"
    )
    roc_data_train_final_xgb = save_roc_data(
        final_XGB_visualizer,
        save_name="roc_auc_final_XGBclassifer_golden_features_train_roc_data.csv",
    )
    plt.close()

    final_XGB_visualizer = ROCAUC(
        grid_final_model.best_estimator_,
        is_fitted=True,
        encoder=le,
        classes=le.classes_,
    )
    final_XGB_visualizer.fit(X_train_final, y_train_final)
    final_XGB_visualizer.score(X_test_final, y_test_final)
    final_XGB_visualizer.show(
        outpath="roc_auc_final_XGBclassifer_golden_features_test_performance.pdf"
    )
    roc_data_test_final_xgb = save_roc_data(
        final_XGB_visualizer,
        save_name="roc_auc_final_XGBclassifer_golden_features_test_roc_data.csv",
    )
    plt.close()

    final_XGB_visualizer.estimator.save_model("final_XGB_model.json")
    # create a TreeExplainer object
    XGB_explainer = shap.TreeExplainer(final_XGB_visualizer.estimator)

    # calculate shap values
    shap_values_XGB = XGB_explainer.shap_values(X_test_final)
    # plot
    shap.summary_plot(
        shap_values_XGB, X_test_final, max_display=min(50, X_test_final.shape[1])
    )
    plt.savefig("xgb_final_features_shap_test_performance.pdf", format="pdf")
    plt.close()

    if print_expression_heatmap is True:
        label_encoding_df_copy = label_encoding_df.copy()
        label_encoding_df_copy.set_index('label', inplace=True)
        label_encoding_dic = label_encoding_df_copy['class'].to_dict()
        label_encoding_dic_swapped = OrderedDict((value, key) for key, value in label_encoding_dic.items())
        print("### Plotting feature expression heatmap ###\n")
        expression_heatmap(
            X[np.unique(runs_proteins)], y, label_encoding_dic_swapped,
            filename="relevant_feature_expression_heatmap_final.pdf",
        )

    if experiment_simulations > 1:
        print(
            f"### Compute and plot statistics of {experiment_simulations} simulations ###\n"
        )
        acc_vec = []
        wavg_f1_vec = []
        avg_f1_vec = []
        cross_validation_scores = []
        mean_cross_validation_score = []
        for run, grid_clf in zip(runs_classification_reports, runs_model):
            acc_vec.append(run["accuracy"])
            wavg_f1_vec.append(run["weighted avg"]["f1-score"])
            avg_f1_vec.append(run["macro avg"]["f1-score"])
            try:
                cross_validation_scores.append(
                    grid_clf.best_score_
                )  # is not always available
                mean_cross_validation_score += grid_clf.best_score_
            except AttributeError:
                pass

        mean_auc = np.mean(runs_auc_values)
        mean_acc = np.mean(acc_vec)
        mean_wavg_f1 = np.mean(wavg_f1_vec)
        mean_avg_f1 = np.mean(avg_f1_vec)
        mean_avg_cv_score = np.mean(cross_validation_scores)

        auc_std = np.std(runs_auc_values)
        std_acc = np.std(acc_vec)
        wavg_f1_std = np.std(wavg_f1_vec)
        avg_f1_std = np.std(avg_f1_vec)
        cv_std = np.std(cross_validation_scores)

        print(
            "mean accuracy:",
            mean_acc,
            "mean_wavg_f1:",
            mean_wavg_f1,
            "mean_avg_f1:",
            mean_avg_f1,
            "mean_auc:",
            mean_auc,
            f"mean cv {scoring}:",
            mean_avg_cv_score,
            "\n",
        )
        print(
            "accuracy std:",
            std_acc,
            "wavg_f1_std:",
            wavg_f1_std,
            "avg_f1_std:",
            avg_f1_std,
            "auc_std:",
            auc_std,
            f"CV {scoring} std:",
            cv_std,
        )

        df = pd.DataFrame(
            {
                "ACC": acc_vec,  # these are computed on default decision threshold and thereby are not informative
                "WAVG_F1": wavg_f1_vec,  # these are computed on default decision threshold and thereby are not informative
                "MAVG_F1": avg_f1_vec,  # these are computed on default decision threshold and thereby are not informative
                "ROC AUC": runs_auc_values,
                f"CV_{scoring}": cross_validation_scores,
            }
        )
        df_sns = dfMat22ColumnRep(df, col_1_name="score_vals", col_2_name="metrics")
        y_handle = df_sns.columns[0]
        x_handle = df_sns.columns[1]
        plt.figure(0)
        sns.violinplot(x=x_handle, y=y_handle, data=df_sns, scale="width")
        plt.savefig("violin_plot.pdf", format="pdf")
        plt.figure(1)
        plt.close()

    return ccc
