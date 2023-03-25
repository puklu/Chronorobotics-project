from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tzlocal
from constants import CURRENT_SYSTEM_TIME, PLOTS_PATH

SHOW_PLOT = False


def pyplot_time_cost_line_plot(x_axis_data, y_axis_data, current_time_local, show_plot=SHOW_PLOT):
    """
    To plot cost vs timestamp using matplot
    Args:
        x_axis_data: Time stamps of the maps
        y_axis_data: Calculated score/cost of each map based on the current time
        current_time_local: Current system time

    """
    plt.figure()
    plt.plot(x_axis_data, y_axis_data)
    plt.title(current_time_local)
    plt.xticks(rotation=30)
    plt.grid()
    if SHOW_PLOT:
        plt.show()


def seaborn_time_cost_line_plot(x_axis_data, y_axis_data, xticks, current_time_local, show_plot=SHOW_PLOT):
    """
    To plot cost vs timestamp using seaborn
    Args:
        x_axis_data: Time stamps of the maps
        y_axis_data: Calculated score/cost of each map based on the current time
        current_time_local: Current system time

    """

    # new_y = np.interp(x_ticks, x_axis_data, y_axis_data)

    # x_cos = list(range(int(min(x_axis_data)), int(max(x_axis_data)), 3600))
    # y_cos = np.cos(x_cos)
    # ax = sns.lineplot(x=x_cos, y=y_cos, marker='o')
    plt.figure(figsize=(23, 12))
    ax = sns.lineplot(x=x_axis_data, y=y_axis_data, marker='o')
    ax.set_facecolor('#F0F0F0')
    # ax = plt.gca()
    ax.set_xticklabels(xticks, rotation=30, ha='right')
    # ax.set_xlim(x_ticks[0], x_ticks[-1])
    # plt.xticks(x_ticks, x_axis_data)
    ax.set_title(f"Score for each map. Current time: {current_time_local}", fontsize=16, fontweight='bold')
    ax.grid(True, linewidth=1.0, color='white')
    ax.set_xlabel("Map timestamp", fontsize=12)
    ax.set_ylabel("Similarity", fontsize=12)
    # ax.text(3, 7, current_time_local, fontsize=12, color='red')
    plt.savefig(f"{PLOTS_PATH}/time_cost.svg")

    if SHOW_PLOT:
        plt.show()


def visualise_heatmap(data, xlabels, ylabels, title, show_plot=SHOW_PLOT):
    """
    Plots a heatmap for a given data
    Args:
        data: The similarity matrix/ fft data etc
        xlabels: Labels for x-axis.
        ylabels: Labels for y-axis.
    """
    plt.figure(figsize=(23, 18))
    ax = sns.heatmap(data,xticklabels=xlabels, yticklabels=ylabels ) #, annot=True, fmt=".3f")
    ax.set_title(title, fontsize=16, fontweight='bold')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    plt.savefig(f"{PLOTS_PATH}/similarity_matrix.svg")

    if SHOW_PLOT:
        plt.show()


def visualise_similarity_matrix(env_name):
    from fetch_utils import fetch_environment
    env_obj = fetch_environment(env_name)  # fetching the env object
    similarity_matrix = env_obj.similarity_matrix
    map_names = env_obj.map_metadata['maps_names']

    plot_title = "Similarity matrix"
    visualise_heatmap(similarity_matrix, map_names, map_names, plot_title)


def visualise_fft(spectrum, show_plot=SHOW_PLOT):
    """
    Plots magnitude spectrum provided the calculated magnitude spectrum
    Args:
        spectrum:

    Returns:

    """
    # plot the magnitude spectrum
    plt.imshow(spectrum, cmap='gray')

    if SHOW_PLOT:
        plt.show()


def visualise_fft_for_env(env_name):
    """
    Plots spectrum for an environment
    Args:
        env_name: Name of the environment

    Returns:

    """
    from fetch_utils import fetch_environment
    from cost_calculation import calculate_fft
    env_obj = fetch_environment(env_name)  # fetching the env object
    maps_names = env_obj.map_metadata['maps_names']
    similarity_matrix = env_obj.similarity_matrix
    spectrum = calculate_fft(similarity_matrix)

    # pyplot_time_cost_line_plot(maps_names, spectrum[0], current_time_local= "lol")
    visualise_heatmap(spectrum, maps_names, maps_names, 'fft')
    # visualise_fft(spectrum)

