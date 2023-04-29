import os
from datetime import datetime
import tzlocal

from load_map import load_map
from graph_creation import create_graph
from cost_calculation import image_similarity_matrix_update, calculate_timeseries, calculate_periodicities
from fetch_utils import fetch_first_images
from constants import IMAGES_PATH


def extract_map_metadata(env_obj, map_name, start_node_name, end_node_name, path):
    """
    Extracts the map meta_data for the environment when a map is uploaded for an environment.
    Also APPENDS to the corresponding env variables for the map.
    Args:
        env_obj: Instance of Environment class
        map_name: Name of the map being uploaded
        start_node_name: Name of the starting node of the map
        end_node_name: Name of the ending node of the map

    Returns: An UPDATED env object

    """
    images, distances, trans, times = load_map(mappaths=f"{path}")

    distance = distances[0][-1]

    env_name = env_obj.name

    env_obj.map_metadata['maps_names'].append(map_name)
    env_obj.map_metadata['images'][map_name] = images
    env_obj.map_metadata['trans'][map_name] = trans
    env_obj.map_metadata['times'][map_name] = times
    env_obj.map_metadata['distances'][map_name] = distances
    env_obj.map_metadata['distance'][map_name] = distance

    # -------------------------------------------------------------------------------------------------
    '''
        Calculating timestamp for the map
        The starting point of times is assumed to be the timestamp for the map
    '''
    starting_timestamp = times[0][0].to_time()  # TODO: this may need some change
    local_timezone = tzlocal.get_localzone()  # get pytz timezone
    local_time = datetime.fromtimestamp(starting_timestamp, local_timezone).strftime('%Y-%m-%d %H:%M:%S')

    env_obj.map_metadata['timestamp'][map_name] = [starting_timestamp, local_time]

    # -------------------------------------------------------------------------------------------------

    # # Updating the similarity matrix for the environment --------------------------------------------

    # first downloading all the first images for all the maps of the environment from the db
    fetch_first_images(env_obj)

    maps_names = []
    maps_timestamps_local = []
    maps_timestamps = []

    for item_ in env_obj.map_metadata['timestamp'].items():
        maps_names.append(item_[0])
        maps_timestamps.append(item_[1][0])
        maps_timestamps_local.append(item_[1][1])

    similarity_matrix_for_env = env_obj.similarity_matrix

    images_names = []
    for map_name in maps_names:
        images_names.append(f"{env_name}.{map_name}.jpg")

    # the updated similarity matrix
    updated_similarity_matrix, updated_softmax_similarity_matrix = image_similarity_matrix_update(similarity_matrix_for_env, images_names, maps_timestamps_local)

    env_obj.similarity_matrix = updated_similarity_matrix
    env_obj.softmax_similarity_matrix = updated_softmax_similarity_matrix

    # deleting all the downloaded images to save space
    for path, directories, files in os.walk(IMAGES_PATH):
        for file in files:
            os.remove(f"{IMAGES_PATH}/{file}")

    # -----------------------------------------------------------------------------------------------

    # calculate time series -------------------------------------------------------------------------
    # TODO: NEED TO CHANGE THE METHOD THAT IS CALLED TO CALCULATE TIMESERIES
    times, values = calculate_timeseries(similarity_matrix=updated_softmax_similarity_matrix, timestamps=maps_timestamps, env_name=env_name)
    env_obj.time_series['times'] = times
    env_obj.time_series['values'] = values

    # -----------------------------------------------------------------------------------------------

    # Calculate periodicities -----------------------------------------------------------------------
    amplitudes, omegas, time_periods, phis, fremen = calculate_periodicities(times=times, values=values, env_name=env_name)

    env_obj.fremen_output['amplitudes'] = amplitudes
    env_obj.fremen_output['omegas'] = omegas
    env_obj.fremen_output['time_periods'] = time_periods
    env_obj.fremen_output['phis'] = phis

    # -----------------------------------------------------------------------------------------------

    # # creating Nodes ------------------------------------------------------------------------------
    start_node, end_node = create_graph(env_obj, start_node_name, end_node_name, map_name, distance)

    env_obj.map_metadata['start_node'].append(start_node)
    env_obj.map_metadata['end_node'].append(end_node)

    # adding the nodes to environment object
    if start_node_name not in env_obj.nodes_names:
        env_obj.nodes_names.append(start_node_name)
        env_obj.nodes.append(start_node)
    if end_node_name not in env_obj.nodes_names:
        env_obj.nodes_names.append(end_node_name)
        env_obj.nodes.append(end_node)

    # ----------------------------------------------------------------------

    return env_obj
