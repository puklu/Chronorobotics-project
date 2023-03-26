import os
from datetime import datetime
import tzlocal

from load_map import load_map
from graph_creation import create_graph
from cost_calculation import image_similarity_matrix_update
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
    # if map_name not in env_obj.map_metadata['maps_names']:
    env_obj.map_metadata['maps_names'].append(map_name)
    # env_obj.map_metadata['images'].append(images)
    # env_obj.map_metadata['trans'].append(trans)
    env_obj.map_metadata['times'].append(times)
    # env_obj.map_metadata['distances'].append(distances)
    env_obj.map_metadata['distance'].append(distance)  # the length of the path

    # ---------------------------------------------------------------------------
    '''
        Calculating timestamp for the map
        The starting point of times is assumed to be the timestamp for the map
    '''
    length_of_times = len(env_obj.map_metadata['times'][0][0])
    starting_timestamp = times[0][0].to_time()  # TODO: this may need some change
    # average_timestamp = int(env_obj.map_metadata['times'][0][0][int(length_of_times / 2)].to_time())

    local_timezone = tzlocal.get_localzone()  # get pytz timezone
    local_time = datetime.fromtimestamp(starting_timestamp, local_timezone).strftime('%Y-%m-%d %H:%M:%S')
    # local_time = datetime.utcfromtimestamp(starting_timestamp).strftime('%Y-%m-%d %H:%M:%S')

    env_obj.map_metadata['timestamp'].append((starting_timestamp, local_time))

    # ---------------------------------------------------------------------------------

    # # Updating the similarity matrix for the environment ------------------------------

    # first downloading all the first images for all the maps of the environment from the db
    fetch_first_images(env_obj)

    maps_names = env_obj.map_metadata['maps_names']
    similarity_matrix_for_env = env_obj.similarity_matrix

    images_names = []
    for map_name in maps_names:
        images_names.append(f"{env_name}.{map_name}.jpg")

    # the updated similarity matrix
    updated_similarity_matrix, updated_softmax_similarity_matrix = image_similarity_matrix_update(similarity_matrix_for_env, images_names)

    env_obj.similarity_matrix = updated_similarity_matrix
    env_obj.softmax_similarity_matrix = updated_softmax_similarity_matrix

    # deleting all the downloaded images to save space
    for path, directories, files in os.walk(IMAGES_PATH):
        for file in files:
            os.remove(f"{IMAGES_PATH}/{file}")

    # ---------------------------------------------------------------------

    # # creating Nodes ------------------------------------------------------
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
