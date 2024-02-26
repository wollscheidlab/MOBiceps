"""
File: foldsConstruction.py
Author: Jens Settelmeier
Created: 6/21/23
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold


def construct_folds(
    X,
    y,
    patient_file_annotations,
    random_state=42,
    test_size=0.15,
    n_splits=5,
    n_repeats=1,
):
    unique_classes = np.unique(y)
    print("### unique classes: ", unique_classes, " ###\n")
    data = []
    labels = []
    label_vec = []

    if patient_file_annotations is None:
        print("### no patient_file_annotations provided, construct fake one ###\n")
        # construct patient_file_annotation fake data frame
        patient_file_annotations = pd.DataFrame(
            columns=["files", "PatientID"]
        )  # was "file"
        patient_file_annotations["files"] = X.index.tolist()  # was "file"
        patient_file_annotations["PatientID"] = np.arange(len(X.index.tolist()))
        patient_file_annotations.set_index("files", inplace=True)  # was "file"
    else:
        print("### (Patient) sample annotation file available... ###\n")

    print("### check available data and listed files... ###\n")

    patienten_ID_vec = np.unique(patient_file_annotations["PatientID"])

    for patient_idx in patienten_ID_vec:
        files4PatientIdx = patient_file_annotations[
            patient_file_annotations["PatientID"] == patient_idx
        ].index.tolist()
        data.append(files4PatientIdx)
        label4files4PatientIdx = y.loc[
            patient_file_annotations[
                patient_file_annotations["PatientID"] == patient_idx
            ].index.tolist()
        ].tolist()
        labels.append(label4files4PatientIdx)
        label_vec.append(
            label4files4PatientIdx[0]
        )  # all labels in the label tupple are the same

    (
        X_train_FileNames,
        X_test_FileNames,
        y_train_Labels,
        y_test_Labels,
    ) = train_test_split(
        data,
        labels,
        random_state=random_state,
        stratify=label_vec,
        shuffle=True,
        test_size=test_size,
    )
    # hold out test set
    X_test = X.loc[[item for sublist in X_test_FileNames for item in sublist]].copy()
    y_test = y.loc[[item for sublist in X_test_FileNames for item in sublist]].copy()

    folds = int(min(max(3, np.round(len(X_train_FileNames) * 0.1)), n_splits))
    skf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=n_repeats)

    index_vec = np.arange(len(X_train_FileNames))
    encoded_label_vec_for_training = []
    for i in y_train_Labels:
        encoded_label_vec_for_training.append(np.nonzero(unique_classes == i[0])[0][0])
    folds = []
    for t_idx, v_idx in skf.split(index_vec, encoded_label_vec_for_training):
        folds.append([t_idx, v_idx])

    # retrieve file names belonging to folds
    folds_filenames = []
    for fold in folds:
        train_filenames = np.array(X_train_FileNames, dtype=object)[[fold[0]]]
        val_filenames = np.array(X_train_FileNames, dtype=object)[[fold[1]]]
        folds_filenames.append([train_filenames, val_filenames])

    # construct folds
    X_train = X.loc[[item for sublist in X_train_FileNames for item in sublist]].copy()
    y_train = y.loc[[item for sublist in X_train_FileNames for item in sublist]].copy()

    X_train_with_numerical_indexes = X_train.reset_index().reset_index().copy()
    # now translate filename indexes to numerical indexes to make it work for rfe

    numerical_folds = []
    for fold in folds_filenames:

        i_th_numerical_train_idx_for_fold = []
        for idx_t in fold[0]:
            tmp = []
            for filename in idx_t:
                filename_corresponding_index = int(
                    X_train_with_numerical_indexes[
                        X_train_with_numerical_indexes["files"] == filename
                    ]["index"]
                )
                tmp.append(filename_corresponding_index)
            i_th_numerical_train_idx_for_fold.extend(tmp)

        i_th_numerical_val_idx_for_fold = []
        for idx_v in fold[1]:
            tmp = []
            for filename in idx_v:
                filename_corresponding_index = int(
                    X_train_with_numerical_indexes[
                        X_train_with_numerical_indexes["files"] == filename
                    ]["index"]
                )
                tmp.append(filename_corresponding_index)
            i_th_numerical_val_idx_for_fold.extend(tmp)

        numerical_folds.append(
            [i_th_numerical_train_idx_for_fold, i_th_numerical_val_idx_for_fold]
        )
    folds = numerical_folds
    return (
        folds_filenames,
        folds,
        X_train,
        y_train,
        X_test,
        y_test,
        patient_file_annotations,
    )
