# Author: Jens Settelmeier 
# Created on 21.08.24 17:49
# File Name: scale_investigation_with_features2.py
# Contact: jenssettelmeier@gmail.com
# License: Apache License 2.0
# You can't climb the ladder of success
# with your hands in your pockets (A. Schwarzenegger)


import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from MOBiceps.rfePlusPlusWF import execute_rfePP
import time
from memory_profiler import memory_usage
import io

def profile_function(func, *args, **kwargs):
    start_time = time.time()
    mem_usage = memory_usage((func, args, kwargs), max_usage=True, retval=True)
    elapsed_time = time.time() - start_time
    result = mem_usage[1]
    max_memory  = mem_usage[0]
    return result, elapsed_time, max_memory

current_path = '/media/dalco/FireCuda1/projects_21082024/MOAgent_revision/MOAgent_sample_scalability'
log_file_path = os.path.join(current_path, 'performance_log_increasing_feature_size.txt')
feature_sizes = [50-2, 100-2, 500-2, 1000-2, 5000-2, 10_000-2, 50_000-2, 100_000-2]
number_of_samples = 200

# Ensure the directory exists and prepare the log file
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Open the log file once with unbuffered I/O
with io.open(log_file_path, 'w', buffering=1) as file:
    file.write("Feature_Size,Elapsed_Time_sec,Memory_Usage_MiB\n")
    for number_of_noise_features in feature_sizes:
        path_to_output = os.path.join(current_path, f'{number_of_noise_features}_features')
        os.makedirs(path_to_output, exist_ok=True)
        X_orig, y = make_moons(n_samples=number_of_samples, noise=0.3, random_state=42)
        np.random.seed(42)
        noisy_features = np.random.randn(number_of_samples, number_of_noise_features)
        X = np.hstack((X_orig, noisy_features))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        col_names = [f'feature_{i}' for i in range(number_of_noise_features + 2)]
        X_df = pd.DataFrame(X, columns=col_names)
        classes = ['control' if x == 1 else 'non_control' for x in y]
        X_df['class'] = classes
        files_col = [f'file_{i}' for i in range(number_of_samples)]
        X_df.insert(0, 'files', files_col)

        path_to_search_output = os.path.join(path_to_output, 'expression_table.csv')
        path_to_class_annotation = os.path.join(path_to_output, 'class_annotations.csv')
        X_df.to_csv(path_to_search_output, index=False)
        y_df = pd.DataFrame({'class': classes, 'files': files_col})
        y_df.to_csv(path_to_class_annotation, index=False)

        # Run analysis and measure performance
        result, elapsed_time, max_memory = profile_function(execute_rfePP, path_to_search_output,
                                                            path_to_class_annotation, path_to_output)
        # Log results to file and flush after every write
        feature_eintrag = number_of_noise_features + 2
        file.write(f"{feature_eintrag},{elapsed_time},{max_memory}\n")
        file.flush()  # Ensure data is written to disk immediately
        print(f"Completed {feature_eintrag} features: Time {elapsed_time:.2f} sec, Memory {max_memory:.2f} MiB")


