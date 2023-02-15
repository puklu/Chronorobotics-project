from datetime import datetime
from zipfile import ZipFile

from constants import ROOT, OBJECTS_PATH
from load_map import load_map
from graph_creation import create_graph


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

    # if map_name not in env_obj.map_metadata['maps_names']:
    env_obj.map_metadata['maps_names'].append(map_name)
    # env_obj.map_metadata['images'].append(images)
    # env_obj.map_metadata['trans'].append(trans)
    env_obj.map_metadata['times'].append(times)
    # env_obj.map_metadata['distances'].append(distances)
    env_obj.map_metadata['distance'].append(distance)  # the length of the path

    '''
        Calculating timestamp for the map
        The middle point of times is assumed to be the timestamp for the map
    '''
    length_of_times = len(env_obj.map_metadata['times'][0][0])
    average_timestamp = int(env_obj.map_metadata['times'][0][0][int(length_of_times / 2)].to_time())
    timestamp = datetime.utcfromtimestamp(average_timestamp).strftime('%Y-%m-%d %H:%M:%S')
    env_obj.map_metadata['timestamp'].append(timestamp)


    # creating Nodes
    start_node, end_node = create_graph(env_obj, start_node_name, end_node_name, map_name, distance)  # , cost)

    env_obj.map_metadata['start_node'].append(start_node)
    env_obj.map_metadata['end_node'].append(end_node)

    # adding the nodes to environment object
    if start_node_name not in env_obj.nodes_names:
        env_obj.nodes_names.append(start_node_name)
        env_obj.nodes.append(start_node)
    if end_node_name not in env_obj.nodes_names:
        env_obj.nodes_names.append(end_node_name)
        env_obj.nodes.append(end_node)

    return env_obj

#
# def update_map_metadata(env_obj):
#     """
#     UPDATES the existing meta-data for the environment
#     Args:
#         env_obj:
#
#     Returns:
#
#     """
#     idx = env_obj.map_metadata['maps_names'].index(map_name)
#     # env_obj.map_metadata['images'][idx] = images
#     # env_obj.map_metadata['trans'][idx] = trans
#     env_obj.map_metadata['times'][idx] = times
#     # env_obj.map_metadata['distances'][idx] = distances
#     env_obj.map_metadata['distance'][idx] = distance  # the length of the path
#
#     '''
#     Calculating timestamp for the map
#     The middle point of times is assumed to be the timestamp for the map
#     '''
#     length_of_times = len(env_obj.map_metadata['times'][0][0])
#     average_timestamp = int(env_obj.map_metadata['times'][0][0][int(length_of_times / 2)].to_time())
#     timestamp = datetime.utcfromtimestamp(average_timestamp).strftime('%Y-%m-%d %H:%M:%S')
#     env_obj.map_metadata['timestamp'][idx] = timestamp


def OBSOLETE_extract_map_metadata(env_obj, map_name):
    """
    Adds the map_metadata for a given environment for the given map. Works only when all the maps are uploaded at once.
    PROBABLY OBSOLETE NOW.
    """
    maps_path = OBJECTS_PATH / "maps"
    env_name = env_obj.name

    with ZipFile(f"{ROOT}/objects/maps/{env_name}/{map_name}.zip", 'r') as zObject:
        zObject.extractall(path=f"{ROOT}/objects/maps/{env_name}/{map_name}")

    map_path = maps_path / env_name / map_name / map_name

    # loading all the maps of the current environment
    images, distances, trans, times = load_map(mappaths=str(map_path))

    if map_name not in env_obj.map_metadata['maps_names']:
        env_obj.map_metadata['maps_names'].append(map_name)
        # env_obj.map_metadata['images'].append(images)
        # env_obj.map_metadata['trans'].append(trans)
        env_obj.map_metadata['times'].append(times)
        env_obj.map_metadata['distances'].append(distances)
