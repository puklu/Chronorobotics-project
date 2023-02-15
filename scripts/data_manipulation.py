"""
This is to manually manipulate the data in db.
Can come in handy for testing
"""
import os
from datetime import datetime

from constants import DOT_ROS_PATH, CLIENT, MAP_BUCKET
from load_map import load_map
from graph_creation import create_graph
from fetch_utils import fetch_environment
from classes_ import Environment


def extract_map_metadata_manipulated(env_obj, map_name, start_node_name, end_node_name, path, DISTANCE=1, COST=0, TIMESTAMP=None):
    """
    Extracts the meta_data of an environment when a map is uploaded for an environment.
    Also updates the necessary env variables.
    Args:
        env_obj: Instance of Environment class
        map_name: Name of the map being uploaded
        start_node_name: Name of the starting node of the map
        end_node_name: Name of the ending node of the map
        DISTANCE: MANIPULATED DISTANCE

    Returns: An updated env object

    """
    images, distances, trans, times = load_map(mappaths=f"{path}")

    # if map_name not in env_obj.map_metadata['maps_names']:
    env_obj.map_metadata['maps_names'].append(map_name)
    # env_obj.map_metadata['images'].append(images)
    # env_obj.map_metadata['trans'].append(trans)
    env_obj.map_metadata['times'].append(times)
    # env_obj.map_metadata['distances'].append(distances)
    env_obj.map_metadata['distance'].append(DISTANCE)  # the length of the path

    '''
    Calculating timestamp for the map
    The middle point of times is assumed to be the timestamp for the map
    '''
    if TIMESTAMP is None:
        length_of_times = len(env_obj.map_metadata['times'][0][0])
        average_timestamp = int(env_obj.map_metadata['times'][0][0][int(length_of_times / 2)].to_time())
        timestamp = datetime.utcfromtimestamp(average_timestamp).strftime('%Y-%m-%d %H:%M:%S')
        env_obj.map_metadata['timestamp'].append(timestamp)
    else:
        env_obj.map_metadata['timestamp'].append(TIMESTAMP)

    # creating Nodes
    start_node, end_node = create_graph(env_obj, start_node_name, end_node_name, map_name, DISTANCE, COST)

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


def manipulated_map_upload(env_name, map_name, start_node, end_node, manipulated_distance, manipulated_cost=0, path_to_directory_containing_map_directory=None):
    """
    Uploads a zipped map to db, BUT the distance and cost can be manually manipulated for the map
    Args:
        env_name: Name of the environment
        map_name: Name of the map
        start_node: Name of the first node
        end_node: Name of the second node
        manipulated_distance: The distance that the user wants the two nodes to be apart by
        manipulated_cost: The cost that the user want to assign to the map
        path_to_directory_containing_map_directory: The location of the map on the disk
    """
    from upload_utils import zip_the_map, env_upload

    map_obj_name = f"{env_name}.{map_name}.zip"  # name of map object in db

    # zipping the map
    location_of_map = zip_the_map(env_name=env_name,
                                  map_name=map_name,
                                  path_to_directory_containing_map_directory=path_to_directory_containing_map_directory)

    if location_of_map is None:  # map doesn't exist in local
        return

    # Fetch env obj from db, append to it, then replace the one in db
    env_obj = fetch_environment(env_name)
    if env_obj:  # if the env exists in db then update it
        if map_name not in env_obj.map_metadata['maps_names']:
            env_obj = extract_map_metadata_manipulated(env_obj=env_obj, map_name=map_name,
                                                       start_node_name=start_node,
                                                       end_node_name=end_node, path=location_of_map,
                                                       DISTANCE=manipulated_distance,
                                                       COST=manipulated_cost)

    else:  # else create the env obj
        env_obj = Environment(name=env_name, gps_position=None)  # env object for the current environment
        env_obj = extract_map_metadata_manipulated(env_obj=env_obj, map_name=map_name, start_node_name=start_node,
                                                   end_node_name=end_node, path=location_of_map,
                                                   DISTANCE=manipulated_distance)

    if path_to_directory_containing_map_directory is None:
        map_path = f"{str(DOT_ROS_PATH)}/{env_name}.{map_name}.zip"  # path of the zipped map that will be uploaded
    else:
        map_path = f"{path_to_directory_containing_map_directory}/{env_name}.{map_name}.zip"  # path of the zipped map that will be uploaded

    # Uploading the map
    try:
        statobj = CLIENT.stat_object(MAP_BUCKET, map_obj_name, ssec=None, version_id=None, extra_query_params=None)
        print(f"{map_obj_name} already exists in {MAP_BUCKET} bucket")

    except:
        CLIENT.fput_object(bucket_name=MAP_BUCKET, object_name=map_obj_name, file_path=map_path)
        env_upload(env_data=env_obj)  # uploading the env obj
        print(f"Map {map_obj_name} uploaded to {MAP_BUCKET} bucket")

    # Delete the zipped map from local
    if path_to_directory_containing_map_directory is None:
        os.remove(f"{str(DOT_ROS_PATH)}/{env_name}.{map_name}.zip")
    else:
        os.remove(f"{path_to_directory_containing_map_directory}/{env_name}.{map_name}.zip")

