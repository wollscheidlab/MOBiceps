# Author: Jens Settelmeier 
# Created on 21.08.24 15:33
# File Name: Comparisons.py
# Contact: jenssettelmeier@gmail.com
# License: Apache License 2.0
# You can't climb the ladder of success
# with your hands in your pockets (A. Schwarzenegger)


import os
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt


number_of_samples = 1000
number_of_noise_features = 198
# Generate a non-linear dataset
X_orig, y = make_moons(n_samples=number_of_samples, noise=0.3, random_state=42)

# Manually add noisy / redundant features
np.random.seed(42)
noisy_features = np.random.randn(number_of_samples, number_of_noise_features)  # Add 5 noisy features
X = np.hstack((X_orig, noisy_features))


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Setup RFECV with Logistic Regression
log_reg = LogisticRegression(solver='lbfgs', max_iter=1000)
rfecv = RFECV(estimator=log_reg, step=1, cv=StratifiedKFold(5), scoring='accuracy')
rfecv.fit(X_train, y_train)
# Transform the test set based on the selected features
X_test_transformed = rfecv.transform(X_test)


# Make predictions with the fitted model
log_reg_pred = rfecv.estimator_.predict(X_test_transformed)
log_reg_prob = rfecv.estimator_.predict_proba(X_test_transformed)[:, 1]  # Probability estimates

# Logistic Regression on full dataset
log_reg_full = LogisticRegression(solver='lbfgs', max_iter=1000)
log_reg_full.fit(X_train, y_train)
full_pred = log_reg_full.predict(X_test)
full_prob = log_reg_full.predict_proba(X_test)[:, 1]  # Probability estimates

# Calculate and print accuracy and ROC AUC Score
print("Accuracy of RFECV LR :", accuracy_score(y_test, log_reg_pred))
print("ROC AUC of RFECV LR :", roc_auc_score(y_test, log_reg_prob))
print("Accuracy with full dataset (including noise) of the LR:", accuracy_score(y_test, full_pred))
print("ROC AUC with full dataset (including noise) of the LR:", roc_auc_score(y_test, full_prob))

# Compute ROC curve for Logistic Regression with RFECV
fpr_log_rfecv, tpr_log_rfecv, _ = roc_curve(y_test, log_reg_prob)

# Compute ROC curve for Logistic Regression on full dataset
fpr_log_full, tpr_log_full, _ = roc_curve(y_test, full_prob)

plt.figure(figsize=(10, 8))
plt.plot(fpr_log_rfecv, tpr_log_rfecv, label='RFECV with Logistic Regression (area = {:.2f})'.format(roc_auc_score(y_test, log_reg_prob)))
plt.plot(fpr_log_full, tpr_log_full, label='Logistic Regression (area = {:.2f})'.format(roc_auc_score(y_test, full_prob)))


from sklearn.tree import DecisionTreeClassifier
# Setup RFECV with Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)
rfecv = RFECV(estimator=decision_tree, step=1, cv=StratifiedKFold(5), scoring='accuracy')
rfecv.fit(X_train, y_train)
# Transform the test set based on the selected features
X_test_transformed = rfecv.transform(X_test)

# Make predictions with the fitted model
dt_pred = rfecv.estimator_.predict(X_test_transformed)
dt_prob = rfecv.estimator_.predict_proba(X_test_transformed)[:,1]

# Decision Tree on full dataset
decision_tree_full = DecisionTreeClassifier(random_state=42)
decision_tree_full.fit(X_train, y_train)
full_pred = decision_tree_full.predict(X_test)
full_prob = decision_tree_full.predict_proba(X_test)[:, 1]  # Probability of the positive class
full_accuracy = accuracy_score(y_test, full_pred)

# Check accuracy
print("Accuracy of RFECV Decision Tree:", accuracy_score(y_test, dt_pred))
print("AUC of RFECV Decision Tree:", roc_auc_score(y_test, dt_prob))

print("Accuracy with full dataset (including noise) of the DT:", full_accuracy)
print("AUC with full dataset (including noise) of the DT:", roc_auc_score(y_test, full_prob))

# Compute ROC curve for Decision Tree with RFECV
fpr_dt_rfecv, tpr_dt_rfecv, _ = roc_curve(y_test, dt_prob)

# Compute ROC curve for Decision Tree on full dataset
fpr_dt_full, tpr_dt_full, _ = roc_curve(y_test, full_prob)

plt.plot(fpr_dt_rfecv, tpr_dt_rfecv, label='RFECV with Decision Tree (area = {:.2f})'.format(roc_auc_score(y_test, dt_prob)))
plt.plot(fpr_dt_full, tpr_dt_full, label='Decision Tree (area = {:.2f})'.format(roc_auc_score(y_test, full_prob)))


from xgboost import XGBClassifier

# Setup RFECV with XGBoost Classifier
xgboost = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
rfecv = RFECV(estimator=xgboost, step=1, cv=StratifiedKFold(5), scoring='accuracy')
rfecv.fit(X_train, y_train)

# Transform the test set based on the selected features
X_test_transformed = rfecv.transform(X_test)

# Make predictions with the fitted model
xgb_pred = rfecv.estimator_.predict(X_test_transformed)
xgb_prob = rfecv.estimator_.predict_proba(X_test_transformed)[:, 1]

# XGBoost on full dataset
xgboost_full = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgboost_full.fit(X_train, y_train)
full_pred = xgboost_full.predict(X_test)
full_prob = xgboost_full.predict_proba(X_test)[:, 1]  # Probability of the positive class
full_accuracy = accuracy_score(y_test, full_pred)

# Check accuracy
print("Accuracy of RFECV XGBoost:", accuracy_score(y_test, xgb_pred))
print("AUC of RFECV XGBoost:", roc_auc_score(y_test, xgb_prob))

print("Accuracy with full dataset (including noise) of XGBoost:", full_accuracy)
print("AUC with full dataset (including noise) of XGBoost:", roc_auc_score(y_test, full_prob))

# Compute ROC curve for XGBoost with RFECV
fpr_xgb_rfecv, tpr_xgb_rfecv, _ = roc_curve(y_test, xgb_prob)

# Compute ROC curve for XGBoost on full dataset
fpr_xgb_full, tpr_xgb_full, _ = roc_curve(y_test, full_prob)

plt.plot(fpr_xgb_rfecv, tpr_xgb_rfecv, label='RFECV with XGBoost (area = {:.2f})'.format(roc_auc_score(y_test, xgb_prob)))
plt.plot(fpr_xgb_full, tpr_xgb_full, label='XGBoost (area = {:.2f})'.format(roc_auc_score(y_test, full_prob)))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves of different models')
plt.legend(loc='lower right')
plt.savefig(f'Model_comparison.pdf')
plt.show()

'''
#%%
import pandas as pd
from MOBiceps.rfePlusPlusWF import execute_rfePP

col_names = [f'feature_{i}' for i in range(number_of_noise_features+2)]
X_df = pd.DataFrame(data=X, columns= col_names)
classes = ['control' if x == 1 else 'non_control' for x in y]
X_df['class'] = classes
files_col = [f'file_{i}' for i in range(number_of_samples)]
X_df.insert(0,'files', files_col)

current_path = '/media/dalco/FireCuda1/projects_21082024/MOAgent_revision/MOAgent_output'
path_to_search_output = os.path.join(current_path,'expression_table.csv')
X_df.to_csv(path_to_search_output, index=False)

y_df = pd.DataFrame(data=classes, columns=['class'])
y_df.insert(0,'files',files_col)
path_to_class_annotation = os.path.join(current_path,'class_annotations.csv')
y_df.to_csv(path_to_class_annotation, index=False)
path_to_output = current_path

most_contributing_features = execute_rfePP(path_to_search_output, path_to_class_annotation, path_to_output)


'''