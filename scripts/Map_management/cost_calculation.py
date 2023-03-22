import os
import sys

import numpy as np
from numpy import sin, pi, cos
from scipy.special import softmax
from PIL import Image
import tzlocal
from datetime import datetime

from visualise import pyplot_time_cost_line_plot, seaborn_time_cost_line_plot, visualise_heatmap, visualise_fft
from constants import CURRENT_SYSTEM_TIME, IMAGES_PATH, SIAMESE_NETWORK_PATH, ROOT, PATH_TO_SIAMESE_MODEL

sys.path.append(str(SIAMESE_NETWORK_PATH))
sys.path.append('../')

from Siamese_network_image_alignment.demo import run_demo

# beta for softmax ---
BETA = 1/50
VISUALIZE_DATA = True


def time_cost_calc(maps_timestamps, periodicities, current_time=CURRENT_SYSTEM_TIME, is_plot=VISUALIZE_DATA):
    """
    Calculates the cost based on the timestamp (unix time) for a map.
    Args:
        periodicities: the significant frequencies present in the maps for an environment
        maps_timestamps: the timestamps of all the maps of the environment
        current_time (optional): Current time,to be provided in case the cost is to be calculated w.r.t to some
        other time instead of current time.
        is_plot: Set to True to plot the cost
    Returns:
        time_cost
    """
    periodicities.sort()
    maps_timestamps.sort()

    N = len(periodicities)
    M = len(maps_timestamps)

    maps_timestamps = np.asarray(maps_timestamps)
    maps_timestamps = np.expand_dims(maps_timestamps, axis=-1)
    maps_timestamps = maps_timestamps.reshape(M, 1)

    periodicities = np.asarray(periodicities)
    periodicities = np.expand_dims(periodicities, axis=-1)
    periodicities = periodicities.reshape(1, N)

    periodicities_sum = sum(periodicities)
    time_difference = maps_timestamps - current_time

    omega = 2 * pi / periodicities

    # cosines = cos(maps_timestamps * omega - current_time)
    cosines = cos(time_difference * omega)
    # cosines = 0.5*cosines
    # cosines = abs(0.5*cos(CURRENT_SYSTEM_TIME) - cosines)

    cost = cosines.sum(axis=1)

    # plotting
    if is_plot:
        local_timezone = tzlocal.get_localzone()  # get pytz timezone
        current_time_local = datetime.fromtimestamp(current_time, local_timezone).strftime('%Y-%m-%d %H:%M:%S')

        maps_timestamps_local = []
        for i in range(len(maps_timestamps)):
            maps_timestamps_local.append(
                datetime.fromtimestamp(maps_timestamps[i][0], local_timezone).strftime('%Y-%m-%d %H:%M:%S'))

        # pyplot_time_cost_line_plot(maps_timestamps_local, cost, current_time_local)
        seaborn_time_cost_line_plot(maps_timestamps_local, cost, current_time_local)

    return cost


def image_similarity_matrix_update(similarity_matrix, images_names, is_plot=VISUALIZE_DATA):
    """
    When a new map is uploaded for an environment, the similarity matrix needs to be calculated again for the environment.
    This method takes in the current similarity matrix of the environment, and then updates it taking in consideration the
    map being uploaded.
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
    img = Image.open(IMAGES_PATH / names_of_images[0])
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

    # SHOULD BE DONE ROW WISE
    # softmax_similarity_matrix = softmax(BETA*similarity_matrix)
    # print(f"softmax_similarity_matrix= {softmax_similarity_matrix}")

    magnitude_spectrum = calculate_fft(similarity_matrix)

    if is_plot:
        map_names = []
        for image in names_of_images:
            env_name, map_name, _ = image.split(".")
            map_names.append(map_name)

        # Plotting
        visualise_heatmap(similarity_matrix, map_names, map_names, 'Similarity matrix')
        visualise_fft(magnitude_spectrum)


    return similarity_matrix


def image_similarity_matrix_calc(images_names, is_plot=VISUALIZE_DATA):
    """
    Calculates the similarity matrix of images present in <IMAGES_PATH>
    Args:
        images_names: Images for which the matrix has to be calculated
        is_plot: to plot heatmap of the matrix
    """

    names_of_images = images_names
    num_of_images = len(names_of_images)

    similarity_matrix = np.zeros((num_of_images, num_of_images))

    # width and height of the images (assuming all the images will have the same dimensions)
    img = Image.open(IMAGES_PATH / names_of_images[0])
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

    # softmax_similarity_matrix = softmax(BETA*similarity_matrix)
    # print(f"softmax_similarity_matrix= {softmax_similarity_matrix}")

    magnitude_spectrum = calculate_fft(similarity_matrix)

    if is_plot:
        map_names = []
        for image in names_of_images:
            env_name, map_name, _ = image.split(".")
            map_names.append(map_name)

        # Plotting
        visualise_heatmap(similarity_matrix, map_names, map_names)
        visualise_fft(magnitude_spectrum)

    return similarity_matrix


def calculate_fft(data):
    """
    Calculates 2d FFT
    Args:
        data:

    Returns: The magnitude spectrum

    """
    fft_data = np.fft.fft2(data)

    # shift the zero-frequency component to the center of the spectrum
    fft_data = np.fft.fftshift(fft_data)

    # calculate the magnitude of the spectrum
    magnitude_spectrum = 20 * np.log(np.abs(fft_data))

    return magnitude_spectrum




