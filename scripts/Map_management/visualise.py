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
    xtick_labels_visible = [label if i % 1 == 0 else "" for i, label in enumerate(xticks)]
    ax.set_xticklabels(xtick_labels_visible, rotation=30, ha='right', fontsize=18)
    plt.yticks(fontsize=18)
    ax.set_title(f"Current time: {current_time_local}", fontsize=20, fontweight='bold', pad=20)
    ax.grid(True, linewidth=1.0, color='white')
    ax.set_xlabel("Map timestamp [-]", fontsize=18, labelpad=20)
    ax.set_ylabel("Cost [-]", fontsize=18, labelpad=20)
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.20)
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
    FONT_SIZE = 30
    plt.figure(figsize=(48, 41))

    ax = sns.heatmap(data, xticklabels=xlabels, yticklabels=ylabels, linewidths=0.005)#, annot=True, fmt=".3f")

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=FONT_SIZE)

    ax.set_title(title, fontsize=35, fontweight='bold', pad=30)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=FONT_SIZE)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=FONT_SIZE)
    ax.set_xlabel("Map timestamp [-]", fontweight='bold', fontsize=FONT_SIZE, labelpad=50)
    ax.set_ylabel("Map timestamp [-]", fontweight='bold', fontsize=FONT_SIZE, labelpad=50)

    plt.savefig(f"{PLOTS_PATH}/{env_name}_similarity_matrix.eps", format='eps', bbox_inches="tight")

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


def plot_time_series(times, values, env_name, show_plot=SHOW_PLOT):
    """

    Args:
        times:
        values:
        show_plot:

    Returns:

    """
    plt.figure(figsize=(23, 12))
    # ax = sns.scatterplot(x=times, y=values, marker='o', s=50)
    ax = sns.lineplot(x=times, y=values, marker='o')
    ax.set_facecolor('#F0F0F0')
    ax.set_facecolor('#F0F0F0')
    # ax.set_xticklabels(times, rotation=30, ha='right')
    ax.set_title(f"Time series", fontsize=20, fontweight='bold')
    ax.grid(True, linewidth=0.5, color='white')
    ax.set_xlabel("Time difference[hours]", fontsize=18, fontweight='bold', labelpad=50)
    ax.set_ylabel("Value [-]", fontsize=18, fontweight='bold', labelpad=50)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.scatter(times,values)
    plt.savefig(f"{PLOTS_PATH}/{env_name}_time_series.eps", format='eps')

    if show_plot:
        plt.show()


def plot_predicted_timeseries(FreMEn_class, times, values, env_name, show_plot=SHOW_PLOT):
    """
    Plots a timeseries and its prediction using fremen
    Args:
        FreMEn_class:
        times:
        values:
        env_name:
        show_plot:

    Returns:

    """
    plt.figure(figsize=(23, 12))

    new_times = np.arange(times[0], times[-1], 3600)

    times = times/3600
    # ax = sns.scatterplot(x=times, y=values, marker='o', s=50, label="Actual")
    ax = sns.lineplot(x=times, y=values, marker='o', label="Actual")

    predicted_values = FreMEn_class.predict(new_times)
    ax = sns.lineplot(x=new_times/3600, y=predicted_values, marker='o', label='Predicted')

    ax.set_facecolor('#F0F0F0')
    ax.set_title(f"Actual vs Predicted Time Series", fontsize=20, fontweight='bold')
    ax.grid(True, linewidth=0.5, color='white')
    ax.set_xlabel("Time difference [hours]", fontsize=18, fontweight='bold', labelpad=50)
    ax.set_ylabel("Value [-]", fontsize=18, fontweight='bold', labelpad=50)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    leg = plt.legend()
    # Set legend font size
    for text in leg.get_texts():
        plt.setp(text, fontsize='18')

    # plt.scatter(times,values)
    plt.savefig(f"{PLOTS_PATH}/{env_name}_predicted_time_series.eps", format='eps')

    if show_plot:
        plt.show()


def plot_predicted_timeseries2(FreMEn_class, times, values, env_name, show_plot=SHOW_PLOT):
    """
    Plots all the timeseries and each timeseries's prediction using fremen.
    Args:
        FreMEn_class:
        times:
        values:
        env_name:
        show_plot:

    Returns:

    """
    plt.figure(figsize=(23, 12))

    for i in range(len(times)):
        new_times = np.arange(times[i][0], times[i][-1], 3600)

        # ax = sns.scatterplot(x=times, y=values, marker='o', s=50, label="Actual")
        ax = sns.lineplot(x=times[i], y=values[i], marker='o', label="Actual")

        predicted_values = FreMEn_class[i].predict(new_times)
        # ax = sns.scatterplot(x=times, y=predicted_values, marker='o', s=50, label='Predicted')
        ax = sns.lineplot(x=new_times, y=predicted_values, marker='o', label='Predicted')

    ax.set_facecolor('#F0F0F0')

    # ax.set_xticklabels(times, rotation=30, ha='right')
    ax.set_title(f"Actual vs predicted time series", fontsize=16, fontweight='bold')
    ax.grid(True, linewidth=0.5, color='white')
    ax.set_xlabel("Time difference[hours]", fontsize=12)
    ax.set_ylabel("Value [-]", fontsize=12)

    # plt.scatter(times,values)
    plt.savefig(f"{PLOTS_PATH}/{env_name}_predicted_time_series2.eps", format='eps')

    if show_plot:
        plt.show()
