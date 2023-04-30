import sys
import csv
import time
import os
from warnings import warn

import numpy as np
from numpy import sin, pi, cos
from scipy.special import softmax
from PIL import Image
import tzlocal
from datetime import datetime

from visualise import pyplot_time_cost_line_plot, seaborn_time_cost_line_plot, visualise_heatmap, plot_time_series, \
    plot_predicted_timeseries, plot_predicted_timeseries2
from constants import CURRENT_SYSTEM_TIME, IMAGES_PATH, SIAMESE_NETWORK_PATH, PATH_TO_SIAMESE_MODEL, RESULTS_PATH
from FreMEn import FreMEn

sys.path.append(str(SIAMESE_NETWORK_PATH))
sys.path.append('../')

from Siamese_network_image_alignment.demo import run_demo

# beta for softmax ---
BETA = 1 / 50
SAVE_PLOTS = True

TEST_DATA = {"path0_map0": [1628441543.1538866, "2021-08-08 18:52:23"],
             "path0_map1": [1628446944.9681673, "2021-08-08 20:22:24"],
             "path0_map2": [1628457620, "2021-08-08 23:20:20"],
             "path0_map3": [1628443562.5771434, "2021-08-08 19:26:02"],
             "path0_map4": [1628445730.1549253, "2021-08-08 20:02:10"],
             "path0_map5": [1628532555, "2021-08-09 20:09:15"],
             "path0_map6": [1628447864.6729546, "2021-08-08 20:37:44"],
             "path0_map7": [1628448852.1848671, "2021-08-08 20:54:12"],
             "path0_map8": [1628450613.950567, "2021-08-08 21:23:33"],
             "path0_map9": [1628537656, "2021-08-09 21:34:16"],
             "path0_map10": [1628543174, "2021-08-09 23:06:14"],
             "path0_map11": [1628629965, "2021-08-10 23:12:45"],
             "path0_map12": [1628625016, "2021-08-10 21:50:16"],
             "path0_map13": [1628715812, "2021-08-11 23:03:32"],
             "path0_map15": [1628698644, "2021-08-11 18:17:24"],
             "path0_map16": [1628612717, "2021-08-10 18:25:17"],
             "path0_map17": [1628527576, "2021-08-09 18:46:16"],
             "path0_map18": [1628533349, "2021-08-09 20:22:29"],
             "path0_map19": [1628534238, "2021-08-09 20:37:18"],
             "path0_map20": [1628429197, "2021-08-08 15:26:37"],
             "path0_map21": [1628516065, "2021-08-09 15:34:25"],
             "path0_map22": [1628603830, "2021-08-10 15:57:10"],
             "path0_map23": [1628532992, "2021-08-09 20:16:32"],
             "path0_map24": [1628620012, "2021-08-10 20:26:52"]
             }

TEST_DATA_test = {"path0_map0": [1628427600, "2021-08-08 15:00:00"],
                  "path0_map1": [1628431200, "2021-08-08 16:00:00"],
                  "path0_map2": [1628434800, "2021-08-08 17:00:00"],
                  "path0_map3": [1628438400, "2021-08-08 18:00:00"],
                  "path0_map4": [1628442000, "2021-08-08 19:00:00"],
                  "path0_map5": [1628445600, "2021-08-08 20:00:00"],
                  "path0_map6": [1628449200, "2021-08-08 21:00:00"],
                  "path0_map7": [1628452800, "2021-08-08 22:00:00"],
                  "path0_map8": [1628456400, "2021-08-08 23:00:00"],
                  "path0_map9": [1628460000, "2021-08-09 00:00:00"],
                  "path0_map10": [1628463600, "2021-08-09 01:00:00"],
                  "path0_map11": [1628467200, "2021-08-09 02:00:00"],
                  "path0_map12": [1628470800, "2021-08-09 03:00:00"],
                  "path0_map13": [1628474400, "2021-08-09 04:00:00"],
                  "path0_map15": [1628478000, "2021-08-09 05:00:00"],
                  "path0_map16": [1628481600, "2021-08-09 06:00:00"],
                  "path0_map17": [1628485200, "2021-08-09 07:00:00"],
                  "path0_map18": [1628488800, "2021-08-09 08:00:00"],
                  "path0_map19": [1628492400, "2021-08-09 09:00:00"],
                  "path0_map20": [1628496000, "2021-08-09 10:00:00"],
                  "path0_map21": [1628499600, "2021-08-09 11:00:00"],
                  "path0_map22": [1628503200, "2021-08-09 12:00:00"],
                  "path0_map23": [1628506800, "2021-08-09 13:00:00"],
                  "path0_map24": [1628510400, "2021-08-09 14:00:00"]
                  }

size = 81
rate = 3 * 3600
time_period = 24 * 3600
softmax_similarity_matrix = np.zeros((size, size))
MAPS_TIMESTAMPS_TEST = np.arange(1628427600, 1628427600 + size * rate, rate)


def convert_timestamp_to_local(timestamps):
    """
    Converts unix time to local time.
    Args:
        timestamps: List of unix times

    Returns:
        list of local times
    """
    local_timezone = tzlocal.get_localzone()  # get pytz timezone
    local_timestamps = []
    for i in range(len(timestamps)):
        local_time = datetime.fromtimestamp(timestamps[i][0], local_timezone).strftime('%Y-%m-%d %H:%M:%S')
        local_timestamps.append(local_time)

    return local_timestamps


def final_cost_calc(env_name, periodicities, amplitudes, phis, fremen=None, current_time=CURRENT_SYSTEM_TIME):
    """
    Calculates the final cost incorporating all the optimisation criterias (map similarity, distance etc.)
    Args:
        env_name: String, name of the environment
        periodicities: list, containing all the periodicities present in the data.
        amplitudes: list, containing magnitudes corresponding to all the periodicities present in that data.
        phis: list, phase shifts corresponding to the periodicities.
        fremen: FreMEn class object.
        current_time: unix time, the time for which the cost needs to be calculated. Default: current system time.

    Returns:
        dictionary, containing the final cost of all the maps ( also contains similarity cost, distance).
    """
    time_cost = time_cost_calc(env_name, periodicities, amplitudes, phis, fremen,
                               current_time)  # final_cost = {map_name: [map_timestamp, cost]}

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


def time_cost_calc(env_name, periodicities, amplitudes, phis, fremen=None, current_time=CURRENT_SYSTEM_TIME,
                   save_plot=SAVE_PLOTS):
    """
    Calculates the cost based on the timestamp (unix time) for a map.
    Args:
        env_name: name of the environment for which costs are to be calculated
        periodicities: the significant frequencies present in the maps for an environment
        amplitudes: Magnitude of each frequency element
        phis: list, phase shifts corresponding to the periodicities.
        fremen: FreMEn class object.
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
        print(f"{env_name} doesn't exist in the database!")
        return

    env_map_metadata = fetch_map_metadata(env_obj)
    maps_names = env_map_metadata['maps_names']
    maps_timestamps = []
    distance = []

    # TODO: The following dictionary is just testing data, MUST BE DELETED/COMMENTED AFTER TESTING
    # warn("Working with hardcoded testing data in final_cost_calc")
    # env_map_metadata['timestamp'] = TEST_DATA

    for map_ in maps_names:
        maps_timestamps.append(env_map_metadata['timestamp'][map_][0])
        distance.append(env_map_metadata['distance'][map_])

    N = len(periodicities)
    M = len(maps_timestamps)

    time_costs = {}
    for map_name in maps_names:
        time_costs[map_name] = []

    # TODO: TESTING DATA AGAIN...DELETE AFTER USE
    # warn("Using test data for maps_timestamps in cost calculation")
    # maps_timestamps = MAPS_TIMESTAMPS_TEST
    # M = len(maps_timestamps)

    maps_timestamps = np.asarray(maps_timestamps, dtype=int)  # TODO: MAYBE dtype SHOULD BE FLOAT??
    maps_timestamps = np.expand_dims(maps_timestamps, axis=-1)
    maps_timestamps = maps_timestamps.reshape(M, 1)

    periodicities = np.asarray(periodicities)
    periodicities = np.expand_dims(periodicities, axis=-1)
    periodicities = periodicities.reshape(1, N)

    amplitudes = np.asarray(amplitudes)
    amplitudes = np.expand_dims(amplitudes, axis=-1)
    amplitudes = amplitudes.reshape(1, N)

    phis = np.asarray(phis)
    phis = np.expand_dims(phis, axis=-1)
    phis = phis.reshape(1, N)

    time_difference = current_time - maps_timestamps

    # prediction = fremen.predict(time_difference)
    # cost = - prediction

    omega = 2 * pi / periodicities

    # print(f"timestamps:{maps_timestamps}")
    # print(f"amplitudes: {amplitudes}")
    # print(f"periodicities: {periodicities}")
    # print(f"phis: {phis}")
    # print(f"time difference: {time_difference}")

    cosines = -amplitudes * cos((time_difference - phis) * omega) + 1  # add 1 to make the values positive

    cost = (1 / (2 * N)) * cosines.sum(axis=1)  # diving by 2N to normalize

    # --------------------------------------------------------------------------------------------------
    # # TODO: PLOT FOR TESTING. REMOVE THE FOLLOWING  LINES
    # current_time_local = datetime.fromtimestamp(current_time, local_timezone).strftime('%Y-%m-%d %H:%M:%S')
    # maps_timestamps_local = convert_timestamp_to_local(maps_timestamps)
    # seaborn_time_cost_line_plot(maps_timestamps_local, cost, maps_timestamps_local, current_time_local,
    #                                 env_name="plot_for_thesis")

    # ---------------------------------------------------------------------------------------------------

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
            maps_timestamps_local.append((f"{item_[1][3]}"))#\n({item_[0]})"))
            costs.append(item_[1][0])

        seaborn_time_cost_line_plot(maps_timestamps_local, costs, maps_timestamps_local, current_time_local,
                                    env_name=env_name)
        # seaborn_time_cost_line_plot(maps_timestamps_local, prediction, maps_timestamps_local, current_time_local, env_name=env_name)
        # seaborn_time_cost_line_plot(maps_timestamps, costs, maps_timestamps, current_time_local, env_name=env_name)

    # saving the dictionary as a CSV file
    # Open a file in write mode
    with open(f"{RESULTS_PATH}/{env_name}_time_costs.csv", "w", newline='') as f:
        writer = csv.writer(f)

        # Write the header row
        # writer.writerow(f"Current time: {current_time_local}")
        writer.writerow(["Current time", current_time_local])
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
        softmax_similarity_matrix[j] = softmax(BETA * similarity_matrix[j])

    # print(f"softmax_similarity_matrix= {softmax_similarity_matrix}")

    if save_plot:
        map_names = []
        for image in names_of_images:
            env_name, map_name, _ = image.split(".")
            map_names.append(map_name)

        # Plotting
        # visualise_heatmap(softmax_similarity_matrix, map_names, map_names, title='Similarity matrix')
        visualise_heatmap(softmax_similarity_matrix, maps_timestamps_local, maps_timestamps_local,
                          title='Similarity matrix', env_name=env_name)
        # visualise_fft(magnitude_spectrum)

    return similarity_matrix, softmax_similarity_matrix


def calculate_similarity_matrix_and_periodicities(env_name, save_plot=SAVE_PLOTS):
    """
    Calculates the similarity matrix of images present in <IMAGES_PATH>. Not used by any other module.
    Doesn't use the matrix stored in the db. Recalculates again for the environment by downloading the first images
    of each map. Plots and saves the matrix as well. Calculates the times series followed by the
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
    from upload_utils import env_upload
    from fetch_utils import save_env_details
    env_obj = fetch_environment(env_name)  # fetching the env details
    fetch_first_images(env_obj)

    # ---------------------------------------------------------------------------------------------------------------
    map_andTimestamp_andLocal = env_obj.map_metadata['timestamp']

    # TODO: The following dictionary is just testing data, MUST BE DELETED/COMMENTED AFTER TESTING..works ONLY FOR env0
    # warn("Working with hardcoded testing data in calculate_similarity_matrix")
    # map_andTimestamp_andLocal = TEST_DATA

    map_andTimestamp_andLocal = dict(
        sorted(map_andTimestamp_andLocal.items(), key=lambda x: x[1][0]))  # sorting the dict by timestamps

    maps_names = []
    maps_timestamps_local = []
    maps_timestamps = []

    for item_ in map_andTimestamp_andLocal.items():
        maps_names.append(item_[0])
        maps_timestamps.append(item_[1][0])
        maps_timestamps_local.append(item_[1][1])

    # Neural Network and similarity calculations ---------------------------------------------------------------
    images_names = []
    for map_name in maps_names:
        images_names.append(f"{env_name}.{map_name}.jpg")

    names_of_images = images_names
    num_of_images = len(maps_names)

    similarity_matrix = np.zeros((num_of_images, num_of_images))

    # width and height of the images (assuming all the images will have the same dimensions)
    with Image.open(IMAGES_PATH / names_of_images[0]) as img:
        width, height = img.size

    print("Calculating similarity matrix......")
    for i in range(num_of_images):
        img1_path = str(IMAGES_PATH / names_of_images[i])
        for j in range(num_of_images):
            if j >= i:  # since it will be a symmetrix matrix, calculating only half of the matrix to save on computation
                img2_path = str(IMAGES_PATH / names_of_images[j])
                similarity_matrix[i][j] = run_demo(img1_path=img1_path,
                                                   img2_path=img2_path,
                                                   img_width=width,
                                                   img_height=height,
                                                   path_to_model=PATH_TO_SIAMESE_MODEL)

                similarity_matrix[j][i] = similarity_matrix[i][j]  # assigning the values for the lower triangle

    softmax_similarity_matrix = np.zeros((num_of_images, num_of_images))
    for j in range(num_of_images):
        softmax_similarity_matrix[j] = softmax(BETA * similarity_matrix[j])

    # -----------------------------------------------------------------------------------------------------------------

    # times, values = calculate_timeseries(similarity_matrix=softmax_similarity_matrix, timestamps=maps_timestamps, env_name=env_name)

    times, values = calculate_timeseries2(similarity_matrix=softmax_similarity_matrix[0],
                                          timestamps=maps_timestamps, env_name=env_name)
    amplitudes, omegas, time_periods, phis, fremen = calculate_periodicities(times=times, values=values,
                                                                             env_name=env_name)

    # times, values = calculate_timeseries_test2(similarity_matrix=softmax_similarity_matrix, timestamps=maps_timestamps, env_name=env_name)
    # fremens, results = calculate_periodicities2(times=times, values=values, env_name=env_name)

    # Plotting ------------------------------------------------------------------------------------------------------
    if save_plot:
        map_names = []
        for image in names_of_images:
            env_name, map_name, _ = image.split(".")
            map_names.append(map_name)

        # visualise_heatmap(softmax_similarity_matrix, map_names, map_names, title='Similarity matrix')

        visualise_heatmap(softmax_similarity_matrix, maps_timestamps_local, maps_timestamps_local,
                          title='Similarity matrix', env_name=env_name)

    # -------------------------------------------------------------------------------------------------------------
    # deleting all the downloaded images to save space
    for path, directories, files in os.walk(IMAGES_PATH):
        for file in files:
            os.remove(f"{IMAGES_PATH}/{file}")

    # ------------------------------------------------------------------------------------------------------------
    # updating the env obj to upload the new values
    env_obj.similarity_matrix = similarity_matrix
    env_obj.softmax_similarity_matrix = softmax_similarity_matrix
    env_obj.time_series['times'] = times
    env_obj.time_series['values'] = values
    env_obj.fremen_output['amplitudes'] = amplitudes
    env_obj.fremen_output['omegas'] = omegas
    env_obj.fremen_output['time_periods'] = time_periods
    env_obj.fremen_output['phis'] = phis
    env_upload(env_data=env_obj)  # uploading the env obj
    save_env_details(env_name=env_name)
    # -----------------------------------------------------------------------------------------------------------

    return similarity_matrix, softmax_similarity_matrix, amplitudes, omegas, time_periods, phis, fremen


def calculate_timeseries(similarity_matrix, timestamps, env_name, save_plot=SAVE_PLOTS):
    """
    Creates a time series using the ENTIRE similarity matrix
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

    times = np.asarray(times)
    values = np.asarray(values)

    if save_plot:
        plot_time_series(times / 3600, values, env_name=env_name)

    return times, values


def calculate_timeseries2(similarity_matrix, timestamps, env_name, save_plot=SAVE_PLOTS):
    """
    Creates a time series WHEN ONLY A SINGLE ROW of the similarity matrix is used.
    Args:
        similarity_matrix: 2 dimensional similarity matrix
        timestamps: A list of timestamps of the maps corresponding to the similarity matrix

    Returns:
        times: a list containing the difference of the timestamps of the similarity matrix
        values: the value of the similarity matrix corresponding to the timestamps whose difference is in times list
    """
    print("Calculating time series....")
    values = similarity_matrix
    times = []

    for i in range(len(similarity_matrix)):
        times.append(timestamps[i] - timestamps[0])

    times = np.asarray(times)
    values = np.asarray(values)

    if save_plot:
        plot_time_series(times / 3600, values, env_name=env_name)

    return times, values


def calculate_periodicities(times, values, env_name, save_plot=SAVE_PLOTS):
    """
    Calculates non uniform fft for a SINGLE timeseries. FreMEn class is used. Being used for finding the frequencies present in
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

    times = np.asarray(times)
    amplitudes = fremen.alphas
    omegas = fremen.omegas
    phis = fremen.phis
    time_periods = (2 * np.pi) / omegas

    print(f"amplitudes: {amplitudes}")
    print(f"omegas: {omegas}")
    print(f"phis: {phis}")
    print(f"time periods: {time_periods}")
    print(f" time periods in hours: {time_periods / 3600}")

    if save_plot:
        plot_predicted_timeseries(FreMEn_class=fremen,
                                  times=times,
                                  values=values,
                                  env_name=env_name)

    return amplitudes, omegas, time_periods, phis, fremen


def calculate_timeseries3(similarity_matrix, timestamps, env_name, save_plot=SAVE_PLOTS):
    """
    Creates a time series for the similarity matrix. Each row of the matrix is used to create a matrix of timeseries'
    Args:
        similarity_matrix: 2 dimensional similarity matrix
        timestamps: A list of timestamps of the maps corresponding to the similarity matrix

    Returns:
        times: a matrix containing the difference of the timestamps of the similarity matrix
        values: a matrix of the values of the similarity matrix corresponding to the timestamps whose difference is in times list
    """
    print("Calculating time series....")
    values = similarity_matrix
    times = []

    for i in range(similarity_matrix.shape[0]):
        row = []
        for j in range(similarity_matrix.shape[0]):
            row.append(timestamps[j] - timestamps[i])
        times.append(row)

    times = np.asarray(times)
    values = np.asarray(values)

    # if save_plot:
    #     plot_time_series(times/3600, values, env_name=env_name)

    return times, values


def calculate_periodicities2(times, values, env_name, save_plot=SAVE_PLOTS):
    """
    Calculates non uniform fft for a matrix containing multiple timeseries'. FreMEn class is used.
    Being used for finding the frequencies present in the data for the environment.
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

    results = []
    fremens = []

    for i in range(len(times[0])):
        fremen = FreMEn()
        fremens.append(fremen)
        fremen.fit(times[i], values[i])
        results.append([fremen.alphas, fremen.omegas, fremen.phis, (2 * np.pi) / fremen.omegas,
                        ((2 * np.pi) / fremen.omegas) / 3600])

    with open(f"{RESULTS_PATH}/{env_name}_avg_time_costs.csv", "w", newline='') as f:
        writer = csv.writer(f)

        writer.writerow(["alphas", "omegas", "phis", "time periods", "time periods in hrs"])
        # Write the data rows
        for result in results:
            writer.writerow(result)

    if save_plot:
        plot_till = 2
        plot_predicted_timeseries2(FreMEn_class=fremens[:plot_till],
                                   times=times[:plot_till],
                                   values=values[:plot_till],
                                   env_name=env_name)

    return fremens, results


def strands_test():
    """
    A test function to test finding frequencies on another dataset
    Returns:

    """
    from pathlib import Path
    ROOT = Path(__file__).absolute().parent.parent.parent
    images = ROOT / "temp_test_data/place_0/"

    times = []
    values = []
    images_names = []

    for path, directories, files in os.walk(images):
        for file in files:
            if file.endswith(".png"):
                images_names.append(file)

    images_names = sorted(images_names)

    with Image.open(images / images_names[0]) as img:
        width, height = img.size

    img1_path = str(images / images_names[0])

    for j in range(len(images_names)):
        times.append(int(images_names[j].split(".")[0]) * 600)
        img2_path = str(images / images_names[j])
        value = run_demo(img1_path=img1_path,
                         img2_path=img2_path,
                         img_width=width,
                         img_height=height,
                         path_to_model=PATH_TO_SIAMESE_MODEL)

        values.append(value)

    times = np.asarray(times)
    values = np.asarray(values)
    fremen, amplitudes, omegas, time_periods, phis = calculate_periodicities(times=times, values=values,
                                                                             env_name="Strands dataset")
