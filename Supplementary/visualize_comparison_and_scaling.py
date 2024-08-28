# Author: Jens Settelmeier 
# Created on 27.08.24 14:25
# File Name: visualize_comparison_and_scaling.py
# Contact: jenssettelmeier@gmail.com
# License: Apache License 2.0
# You can't climb the ladder of success
# with your hands in your pockets (A. Schwarzenegger)


import matplotlib.pyplot as plt
import pandas as pd

# comparison
path_to_sample_scaling = '/media/dalco/FireCuda1/projects_21082024/MOAgent_revision/performance_log_increasing_sample_size.txt'
path_to_feature_scaling = '/media/dalco/FireCuda1/projects_21082024/MOAgent_revision/performance_log_increasing_feature_size.txt'

df_samples = pd.read_table(path_to_sample_scaling, sep=',')
df_samples['Elapsed_Time_sec'] = df_samples['Elapsed_Time_sec']/60
df_features = pd.read_table(path_to_feature_scaling,sep=',')
df_features['Elapsed_Time_sec'] = df_features['Elapsed_Time_sec']/60


def plot_data_and_save(df, zusatz, log_scale):
    # First plot: Elapsed Time vs. Sample Size
    plt.figure(figsize=(10, 6))
    plt.plot(df[f'{zusatz}_Size'], df['Elapsed_Time_sec'], marker='o', linestyle='-')
    plt.xlabel(f'{zusatz}')
    plt.ylabel('Elapsed Time (min)')
    plt.title(f'Elapsed Time vs. {zusatz}')
    plt.grid(True)
    if log_scale is True:
        plt.xscale('log')  # Applying logarithmic scale to the x-axis
       # plt.yscale('log')  # Applying logarithmic scale to the y-axis
    plt.tight_layout()
    plt.savefig(f'Elapsed_Time_vs_{zusatz}_Size.pdf')
    #plt.close()

    # Second plot: Memory Usage vs. Sample Size
    plt.figure(figsize=(10, 6))
    plt.plot(df[f'{zusatz}_Size'], df['Memory_Usage_MiB'], marker='o', linestyle='-', color='red')
    plt.xlabel(f'{zusatz}')
    plt.ylabel('Memory Usage (MiB)')
    plt.title(f'Memory Usage vs. {zusatz} Size')
    plt.grid(True)
    if log_scale is True:
        plt.xscale('log')  # Applying logarithmic scale to the x-axis
       # plt.yscale('log')  # Applying logarithmic scale to the y-axis
    plt.tight_layout()
    plt.savefig(f'Memory_Usage_vs_{zusatz}_Size.pdf')
    #plt.close()


# Call the function to plot data
plot_data_and_save(df_samples, zusatz = 'Sample', log_scale=True)
plot_data_and_save(df_features, zusatz = 'Feature', log_scale=True)

