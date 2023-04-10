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
        show_plot: Saves plot to results/plots/ directory if set to True. True by default

    """
    plt.figure()
    plt.plot(x_axis_data, y_axis_data)
    plt.title(current_time_local)
    plt.xticks(rotation=30)
    plt.grid()
    if show_plot:
        plt.show()


def seaborn_time_cost_line_plot(x_axis_data, y_axis_data, xticks, current_time_local, env_name, show_plot=SHOW_PLOT):
    """
    To plot cost vs timestamp using seaborn
    Args:
        x_axis_data: Time stamps of the maps
        y_axis_data: Calculated score/cost of each map based on the current time
        current_time_local: Current system time
        show_plot: Saves plot to results/plots/ directory if set to True. True by default

    """

    # new_y = np.interp(x_ticks, x_axis_data, y_axis_data)

    # x_cos = list(range(int(min(x_axis_data)), int(max(x_axis_data)), 3600))
    # y_cos = np.cos(x_cos)
    # ax = sns.lineplot(x=x_cos, y=y_cos, marker='o')
    plt.figure(figsize=(23, 12))
    ax = sns.lineplot(x=x_axis_data, y=y_axis_data, marker='o')
    ax.set_facecolor('#F0F0F0')
    ax.set_xticklabels(xticks, rotation=30, ha='right')
    ax.set_title(f"Score for each map. Current time: {current_time_local}", fontsize=16, fontweight='bold')
    ax.grid(True, linewidth=1.0, color='white')
    ax.set_xlabel("Map timestamp", fontsize=12)
    ax.set_ylabel("Cost [-]", fontsize=12)
    # ax.set_ylim(0, 1)
    plt.savefig(f"{PLOTS_PATH}/{env_name}_time_cost.eps", format='eps')

    if show_plot:
        plt.show()


def visualise_heatmap(data, xlabels, ylabels, title, env_name, show_plot=SHOW_PLOT):
    """
    Plots a heatmap for a given data
    Args:
        data: The similarity matrix/ fft data etc
        xlabels: Labels for x-axis.
        ylabels: Labels for y-axis.
        show_plot: Saves plot to results/plots/ directory if set to True. True by default

    """
    plt.figure(figsize=(23, 18))
    ax = sns.heatmap(data,xticklabels=xlabels, yticklabels=ylabels, linewidths=0.005, annot=True, fmt=".3f")
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    plt.savefig(f"{PLOTS_PATH}/{env_name}_similarity_matrix.eps" , format='eps')

    if show_plot:
        plt.show()


def visualise_similarity_matrix(env_name):

    from fetch_utils import fetch_environment
    env_obj = fetch_environment(env_name)  # fetching the env object

    # similarity_matrix = env_obj.similarity_matrix
    similarity_matrix = env_obj.softmax_similarity_matrix

    map_names = env_obj.map_metadata['maps_names']

    plot_title = "Similarity matrix"
    visualise_heatmap(similarity_matrix, map_names, map_names, plot_title)


def plot_time_series(times, values, show_plot=SHOW_PLOT):
    """

    Args:
        times:
        values:
        show_plot:

    Returns:

    """
    plt.figure(figsize=(23, 12))
    ax = sns.scatterplot(x=times, y=values, marker='o', s=50)
    ax.set_facecolor('#F0F0F0')
    # ax.set_xticklabels(times, rotation=30, ha='right')
    ax.set_title(f"Time series to calculate the periodicities", fontsize=16, fontweight='bold')
    ax.grid(True, linewidth=0.5, color='white')
    ax.set_xlabel("Time difference[sec]", fontsize=12)
    ax.set_ylabel("Value [-]", fontsize=12)

    # plt.scatter(times,values)
    plt.savefig(f"{PLOTS_PATH}/time_series.eps", format='eps')

    if show_plot:
        plt.show()


def plot_predicted_timeseries(FreMEn_class, times, values, show_plot=SHOW_PLOT):
    plt.figure(figsize=(23, 12))
    ax = sns.scatterplot(x=times, y=values, marker='o', s=50, label="Actual")
    predicted_values = FreMEn_class.predict(times)
    ax = sns.scatterplot(x=times, y=predicted_values, marker='o', s=50, label='Predicted')
    ax.set_facecolor('#F0F0F0')

    # ax.set_xticklabels(times, rotation=30, ha='right')
    ax.set_title(f"Actual vs predicted time series", fontsize=16, fontweight='bold')
    ax.grid(True, linewidth=0.5, color='white')
    ax.set_xlabel("Time difference[sec]", fontsize=12)
    ax.set_ylabel("Value [-]", fontsize=12)

    # plt.scatter(times,values)
    plt.savefig(f"{PLOTS_PATH}/predicted_time_series.eps", format='eps')

    if show_plot:
        plt.show()
