import sys
import csv
import time

import numpy as np
from numpy import sin, pi, cos
from scipy.special import softmax
import pandas as pd
from PIL import Image
import tzlocal
from datetime import datetime

from visualise import pyplot_time_cost_line_plot, seaborn_time_cost_line_plot, visualise_heatmap, plot_time_series, plot_predicted_timeseries
from constants import CURRENT_SYSTEM_TIME, IMAGES_PATH, SIAMESE_NETWORK_PATH, PATH_TO_SIAMESE_MODEL, RESULTS_PATH
from FreMEn import FreMEn

sys.path.append(str(SIAMESE_NETWORK_PATH))
sys.path.append('../')

from Siamese_network_image_alignment.demo import run_demo

# beta for softmax ---
BETA = 1 / 50
SAVE_PLOTS = True


def final_cost_calc(env_name, periodicities, amplitudes, current_time=CURRENT_SYSTEM_TIME):
    """

    Args:
        env_name:
        periodicities:
        amplitudes
        current_time:

    Returns:

    """
    # TODO: Remove the following line after testing
    current_time = 1628412743

    time_cost = time_cost_calc(env_name, periodicities, amplitudes, current_time)  # final_cost = {map_name: [map_timestamp, cost]}

    if time_cost is None:
        return

    final_cost = time_cost.copy()

    for map_ in time_cost:
        time_cost_for_map = time_cost[map_][0]
        distance = time_cost[map_][0]
        final_calculated_cost = 0 * distance + 1 * time_cost_for_map  # TODO: Needs to be decided
        final_cost[map_].append(final_calculated_cost)

    # print(final_cost)

    return final_cost


def time_cost_calc(env_name, periodicities, amplitudes, current_time=CURRENT_SYSTEM_TIME, save_plot=SAVE_PLOTS):
    """
    Calculates the cost based on the timestamp (unix time) for a map.
    Args:
        env_name: name of the environment for which costs are to be calculated
        periodicities: the significant frequencies present in the maps for an environment
        amplitudes: Magnitude of each frequency element
        current_time (optional): Current time,to be provided in case the cost is to be calculated w.r.t to some
        other time instead of current time.
        save_plot: Set to True to plot the cost
    Returns:
        time_cost
    """

    local_timezone = tzlocal.get_localzone()  # get pytz timezone

    from fetch_utils import fetch_environment, fetch_map_metadata
    env_obj = fetch_environment(env_name)  # fetching the env object

    if env_obj is None:
        print(f"{env_name} doesn't exist!")
        return

    env_map_metadata = fetch_map_metadata(env_obj)
    map_timestamps = env_map_metadata['timestamp']
    maps_names = env_map_metadata['maps_names']
    distance = env_map_metadata['distance']
    maps_timestamps = [map_timestamp[0] for map_timestamp in map_timestamps]

    # periodicities.sort()
    # maps_timestamps.sort()

    N = len(periodicities)
    M = len(maps_timestamps)

    time_costs = {}
    for map_name in maps_names:
        time_costs[map_name] = []

    maps_timestamps = np.asarray(maps_timestamps, dtype=int)  # TODO: MAYBE dtype SHOULD BE FLOAT??
    maps_timestamps = np.expand_dims(maps_timestamps, axis=-1)
    maps_timestamps = maps_timestamps.reshape(M, 1)

    periodicities = np.asarray(periodicities)
    periodicities = np.expand_dims(periodicities, axis=-1)
    periodicities = periodicities.reshape(1, N)

    amplitudes = np.asarray(amplitudes)
    amplitudes = np.expand_dims(amplitudes, axis=-1)
    amplitudes = amplitudes.reshape(1, N)

    time_difference = maps_timestamps - current_time

    omega = 2 * pi / periodicities

    # print(amplitudes)

    # TODO: softmaxing amplitudes not the best strategy. SHOULD BE CHANGED!
    # amplitudes = softmax(amplitudes)  # bringing amplitudes in [0,1] range

    # print(amplitudes)

    cosines = -amplitudes*cos(time_difference * omega) + 1  # add 1 to make the values positive

    # cosines = -amplitudes*cos(time_difference * omega)

    cost = (1 / (2 * N)) * cosines.sum(axis=1)  # diving by 2N to normalize

    # min_cost = min(cost)
    # max_cost = max(cost)
    # cost = (cost-min_cost)/(max_cost-min_cost)

    for i in range(M):
        map_timestamp_local = datetime.fromtimestamp(maps_timestamps[i][0], local_timezone).strftime(
            '%Y-%m-%d %H:%M:%S')
        time_costs[maps_names[i]] = [cost[i], distance[i], maps_timestamps[i][0], map_timestamp_local]

    # PLOTTING THE COST
    if save_plot:
        current_time_local = datetime.fromtimestamp(current_time, local_timezone).strftime('%Y-%m-%d %H:%M:%S')
        maps_timestamps = []
        maps_timestamps_local = []
        costs = []

        sorted_by_timestamp = sorted(time_costs.items(), key=lambda x: x[1][2])

        for item_ in sorted_by_timestamp:
            maps_timestamps.append(item_[1][2])
            maps_timestamps_local.append((f"{item_[1][3]}\n({item_[0]})"))
            costs.append(item_[1][0])


        # data = pd.DataFrame({'x': maps_timestamps_local, 'y': cost})
        # seaborn_time_cost_line_plot(maps_timestamps_ticks_local, cost_for_xticks, current_time_local)

        # pyplot_time_cost_line_plot(maps_timestamps_local, cost, current_time_local)
        seaborn_time_cost_line_plot(maps_timestamps_local, costs, maps_timestamps_local, current_time_local, env_name=env_name)
        # seaborn_time_cost_line_plot(maps_timestamps[:,0], costs, maps_timestamps_local, current_time_local)

    # saving the dictionary as a CSV file
    # Open a file in write mode
    with open(f"{RESULTS_PATH}/{env_name}_time_costs.csv", "w", newline='') as f:
        writer = csv.writer(f)

        # Write the header row
        # writer.writerow(f"Current time: {current_time_local}")
        writer.writerow(["Current time",  current_time_local])
        writer.writerow(["map name", "time_cost | distance | timestamp | timestamp_local"])

        # Write the data rows
        for key, value in time_costs.items():
            writer.writerow([key, value])

    return time_costs


def image_similarity_matrix_update(similarity_matrix, images_names, maps_timestamps_local, save_plot=SAVE_PLOTS):
    """
    When a new map is uploaded for an environment, the similarity matrix needs to be calculated again for the environment.
    This method takes in the current similarity matrix of the environment, and then updates it taking in consideration the
    map being uploaded. The matrix is NOT sorted by timestamps here.
    Args:
        similarity_matrix: The current similarity matrix for the environment.
        images_names: Names of the (first) images of all the maps which are stored in <IMAGES_PATH>.
        The last name is the latest map that is being uploaded.
    Returns:

    """
    names_of_images = images_names

    # calculate the similarity matrix
    num_of_images = len(names_of_images)

    # width and height of the images (assuming all the images will have the same dimensions)
    with Image.open(IMAGES_PATH / names_of_images[0]) as img:
        width, height = img.size

    # path of the image of the current map being uploaded
    img2_path = str(IMAGES_PATH / names_of_images[-1])

    print("Calculating similarity matrix......")
    similarity_for_the_new_image = []
    for i in range(num_of_images):
        # paths of the previous images
        img1_path = str(IMAGES_PATH / names_of_images[i])

        similarity_score_for_the_new_image = run_demo(img1_path=img1_path,
                                                      img2_path=img2_path,
                                                      img_width=width,
                                                      img_height=height,
                                                      path_to_model=PATH_TO_SIAMESE_MODEL)

        similarity_score_for_the_new_image = similarity_score_for_the_new_image.item()
        similarity_for_the_new_image.append(similarity_score_for_the_new_image)

    similarity_for_the_new_image_vertical = similarity_for_the_new_image[0:-1]

    similarity_for_the_new_image = np.asarray(similarity_for_the_new_image)
    similarity_for_the_new_image = np.expand_dims(similarity_for_the_new_image, axis=-1)

    similarity_for_the_new_image_vertical = np.asarray(similarity_for_the_new_image_vertical)
    similarity_for_the_new_image_vertical = np.expand_dims(similarity_for_the_new_image_vertical, axis=-1)

    if len(similarity_matrix) > 0:
        similarity_matrix = np.append(similarity_matrix, similarity_for_the_new_image_vertical, axis=1)
        similarity_matrix = np.append(similarity_matrix, similarity_for_the_new_image.T, axis=0)

    else:
        similarity_matrix = similarity_for_the_new_image

    # print(f"similarity_matrix= {similarity_matrix}")

    # Softmaxing the matrix values
    softmax_similarity_matrix = np.zeros((num_of_images, num_of_images))
    for j in range(num_of_images):
        softmax_similarity_matrix[j] = softmax(BETA*similarity_matrix[j])

    # print(f"softmax_similarity_matrix= {softmax_similarity_matrix}")

    if save_plot:
        map_names = []
        for image in names_of_images:
            env_name, map_name, _ = image.split(".")
            map_names.append(map_name)

        # Plotting
        # visualise_heatmap(softmax_similarity_matrix, map_names, map_names, title='Similarity matrix')
        visualise_heatmap(softmax_similarity_matrix, maps_timestamps_local, maps_timestamps_local, title='Similarity matrix', env_name=env_name)
        # visualise_fft(magnitude_spectrum)

    return similarity_matrix, softmax_similarity_matrix


def calculate_similarity_matrix_and_periodicities(env_name, save_plot=SAVE_PLOTS):
    """
    Calculates the similarity matrix of images present in <IMAGES_PATH>. Not used by any other module.
    Doesn't use the matrix stored in the db. Recalculates again for the environment by downloading the first images
    of each map. Plots and saves the matrix as well. Doesn't update the matrix stored in the database.
    So useful in case matrix needs to be visualised again for any reason. Calculates the times series followed by the
    calculation of periodicities. THE MATRIX IS SORTED BY TIME HERE.
    Args:
        env_name: Name of the environment for which the matrix needs to be calculated
        save_plot: to plot heatmap of the matrix
    Returns:
        similarity_matrix: Similarity matrix for the environment
        softmax_similarity_matrix: Similarity matrix with softmax
        amplitudes: Magnitude of each frequency element
        omegas: Angular frequencies ( 2*pi/T )
        time_periods: The time periods corresponding to omegas. Easier to use than omegas.
        phis: Phase shift for each spectral element.
    """
    from fetch_utils import fetch_first_images, fetch_environment
    env_obj = fetch_environment(env_name)  # fetching the env details
    # time1 = time.time()
    fetch_first_images(env_obj)
    # time2 = time.time()

    map_andTimestamp_andLocal = env_obj.map_metadata['timestamp']
    map_andTimestamp_andLocal = dict(sorted(map_andTimestamp_andLocal.items(), key=lambda x: x[1][0])) # sorting the dict by timestamps

    maps_names = []
    maps_timestamps_local = []
    maps_timestamps = []

    for item_ in map_andTimestamp_andLocal.items():
        maps_names.append(item_[0])
        maps_timestamps.append(item_[1][0])
        maps_timestamps_local.append(item_[1][1])

    # time3 = time.time()

    images_names = []
    for map_name in maps_names:
        images_names.append(f"{env_name}.{map_name}.jpg")

    names_of_images = images_names
    num_of_images = len(names_of_images)

    similarity_matrix = np.zeros((num_of_images, num_of_images))

    # width and height of the images (assuming all the images will have the same dimensions)
    with Image.open(IMAGES_PATH / names_of_images[0]) as img:
        width, height = img.size

    print("Calculating similarity matrix......")
    for i in range(num_of_images):
        img1_path = str(IMAGES_PATH / names_of_images[i])
        for j in range(num_of_images):
            if j >= i:  # since it will be a symmetrix matrix, calculating only the halp of the matrix to save on computation
                img2_path = str(IMAGES_PATH / names_of_images[j])
                similarity_matrix[i][j] = run_demo(img1_path=img1_path,
                                                   img2_path=img2_path,
                                                   img_width=width,
                                                   img_height=height,
                                                   path_to_model=PATH_TO_SIAMESE_MODEL)

                similarity_matrix[j][i] = similarity_matrix[i][j]  # assigning the values on the lower triangle

    # print(f"similarity_matrix= {similarity_matrix}")
    # time4 = time.time()

    softmax_similarity_matrix = np.zeros((num_of_images, num_of_images))
    for j in range(num_of_images):
        softmax_similarity_matrix[j] = softmax(BETA * similarity_matrix[j])

    # time5 = time.time()

    times, values = calculate_timeseries(similarity_matrix=softmax_similarity_matrix, timestamps=maps_timestamps)

    # time6 = time.time()

    amplitudes, omegas, time_periods, phis = calculate_periodicities(times=times, values=values)

    # time7 = time.time()

    if save_plot:
        map_names = []
        for image in names_of_images:
            env_name, map_name, _ = image.split(".")
            map_names.append(map_name)

        # Plotting
        # visualise_heatmap(softmax_similarity_matrix, map_names, map_names, title='Similarity matrix')

        visualise_heatmap(softmax_similarity_matrix, maps_timestamps_local, maps_timestamps_local, title='Similarity matrix', env_name=env_name)
        # visualise_fft(softmax_similarity_matrix)

    # time8 = time.time()

    # print(f"fetching images time: {time2-time1}")
    # print(f"preparing data time{time3-time2}")
    # print(f"calculating the matrix time: {time4-time3}")
    # print(f"softmaxing time:{time5-time4}")
    # print(f"calculating timeseries time: {time6-time5}")
    # print(f"calculating periodicities: {time7-time6}")
    # print(f"visualising time: {time8-time7}")

    return similarity_matrix, softmax_similarity_matrix, amplitudes, omegas, time_periods, phis


def calculate_timeseries(similarity_matrix, timestamps, save_plot=SAVE_PLOTS):
    """
    Creates a time series for the similarity matrix
    Args:
        similarity_matrix: 2 dimensional similarity matrix
        timestamps: A list of timestamps of the maps corresponding to the similarity matrix

    Returns:
        times: a list containing the difference of the timestamps of the similarity matrix
        values: the value of the similarity matrix corresponding to the timestamps whose difference is in times list
    """
    print("Calculating time series....")
    num_of_maps = len(similarity_matrix[0])
    times = []
    values = []

    # zeros_temp = []
    for i in range(num_of_maps):
        for j in range(num_of_maps):
            if j >= i:
                times.append(abs(timestamps[i] - timestamps[j]))
                values.append(similarity_matrix[i][j])
            # if j == i:
            #     zeros_temp.append(similarity_matrix[i][j])

    # zeroes_value = sum(zeros_temp)/ len(zeros_temp)
    # times.append(0)
    # values.append(zeroes_value)

    # print(f"Sorted Times: {sorted(times)}")
    # print(f"Values: {values}")

    times = np.asarray(times)
    values = np.asarray(values)

    if save_plot:
        plot_time_series(times, values)

    return times, values


def calculate_periodicities(times, values,  save_plot=SAVE_PLOTS):
    """
    Calculates non uniform fft for a timeseries. FreMEn class is used. Being used for finding the frequencies present in
    the data for the environment.
    Args:
        times: A list containing times for the datapoints.
        values: A list containing the values corresponding to the times list.

    Returns:
        amplitudes: Magnitude of each frequency element
        omegas: Angular frequencies ( 2*pi/T )
        time_periods: The time periods corresponding to omegas. Easier to use than omegas.
        phis: Phase shift for each spectral element.
    """
    print("Calculating periodicities....")

    fremen = FreMEn()
    fremen.fit(times, values)

    amplitudes = fremen.alphas
    omegas = fremen.omegas
    phis = fremen.phis
    time_periods = (2*np.pi)/omegas

    print(f"amplitudes: {amplitudes}")
    print(f"omegas: {omegas}")
    print(f"phis: {phis}")
    print(f"time periods: {time_periods}")
    print(f" time periods in hours: {time_periods/3600}")

    # If the magnitudes of the periodicities are too low, setting periodicities to 24 hours and 1 week --------------
    new_amplitudes = []
    new_omegas = []
    new_time_periods = []
    new_phis = []
    for idx, amplitude in enumerate(amplitudes):
        if amplitude > 0.3:
            new_amplitudes.append(amplitude)
            new_omegas.append(omegas[idx])
            new_phis.append(phis[idx])

    if not new_amplitudes:
        new_amplitudes = np.array([1, 0.7])
        new_time_periods = np.array([86400, 604800])  # 24 hours and 1 week
        new_omegas = 2*pi/new_time_periods

    new_amplitudes = np.array(new_amplitudes)
    new_omegas = np.array(new_omegas)
    new_time_periods = np.array(new_time_periods)
    new_phis = np.array(new_phis)
    # --------------------------------------------------------------------------------------------------------------

    if save_plot:
        plot_predicted_timeseries(FreMEn_class=fremen, times=times, values=values)

    # return amplitudes, omegas, time_periods, phis
    return amplitudes, omegas, time_periods, phis
