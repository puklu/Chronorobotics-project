import os
from zipfile import ZipFile
from copyreg import pickle
import pickle
import json
import graphviz
import logging

from constants import FETCHED_MAPS_PATH, FETCHED_MAP_OBJ_PATH, FETCHED_ENV_OBJ_PATH, ENV_BUCKET, MAP_BUCKET, CLIENT, \
    FIRST_IMAGE_BUCKET, IMAGES_PATH, RESULTS_PATH, LOGS_PATH
from cost_calculation import time_cost_calc

logging.basicConfig(filename=f"{LOGS_PATH}/fetch_utils.log", level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def save_env_details(env_name):
    """
    Saves  the contents of env class as JSON file in results directory, a graphviz representation of the graph
    of the environment.
    Args:
        env_name: name of the environment for which details need to be saved
     """
    g = graphviz.Digraph()  # for graph visualisation
    env_obj = fetch_environment(env_name)  # fetching the env object
    meta_data_dict = fetch_map_metadata(env_obj)  # getting map metadata from it

    if meta_data_dict:
        # a dictionary to save all data as JSON
        env_details = {"neighbours": [],
                       "start_nodes": [],
                       "end_nodes": [],
                       "Similarity_matrix": env_obj.similarity_matrix.tolist(),
                       "name": env_obj.name,
                       "nodes_names": env_obj.nodes_names,
                       "maps_names": meta_data_dict['maps_names'],
                       "distance": meta_data_dict['distance'],
                       "timestamp": meta_data_dict['timestamp']}

        # print(f"env name: {env_obj.name}")
        # print(f"nodes in the env: {env_obj.nodes_names}")
        # print(f"maps in the env: {meta_data_dict['maps_names']}")
        # print(f"distance: {meta_data_dict['distance']}")
        # print(f"timestamps: {meta_data_dict['timestamp']}")

        # print(f"Nodes and their neighbours are")
        for node in env_obj.nodes:
            neighbours = {}

            # print(f"Weight and cost of the node are: {node.g_cost} | {node.h_cost}")
            # print(f"Neighbours of {node.key} with distance are:", end=' ')
            for neighbour in node.neighbours:
                if node.key in neighbours:
                    neighbours[f"{node.key}"].append([neighbour[0].key, neighbour[2], neighbour[1]])

                else:
                    neighbours[f"{node.key}"] = [[neighbour[0].key, neighbour[2], neighbour[1]]]

                # print(f"{neighbour[0].key}: {neighbour[2]}", end=' | ')
            # print()

            env_details["neighbours"].append(neighbours)

        with open(f"{RESULTS_PATH}/env_details.json", "w") as f:
            json.dump(env_details, f)

        # visualising the environment graph ------------------------------------------------
        maps_names = meta_data_dict['maps_names']
        start_nodes = meta_data_dict['start_node']
        end_nodes = meta_data_dict['end_node']

        for gidx in range(len(maps_names)):
            g.attr('edge', fontsize='5')
            g.node(start_nodes[gidx].key, style='filled', fillcolor='darkturquoise', fontcolor='black')
            g.node(end_nodes[gidx].key, style='filled', fillcolor='darkturquoise', fontcolor='black')
            g.edge(start_nodes[gidx].key, end_nodes[gidx].key, label=maps_names[gidx], color='black', fontcolor='black')

        g.attr(rankdir='LR')
        g.graph_attr['label'] = f"Graph representation of {env_obj.name}"
        g.graph_attr['labelloc'] = 'b'
        g.render(f"{RESULTS_PATH}/env_graph", format='eps')
        # --------------------------------------------------------------------------------

        print(f"Details for {env_name} downloaded to {RESULTS_PATH}")

    else:
        print(f"metadata doesn't exist")


def fetch_map_metadata(env_obj):
    """
    Fetches metadata about the maps in the environment
    Args:
        env_obj: env object

    Returns: A dictionary containing the map metadata

    """
    if env_obj:
        env_map_metadata = env_obj.map_metadata
        return env_map_metadata
    else:
        return None


def fetch_maps_by_time_cost(env_name, periodicities):
    """
    TODO: USELESS NOW PROBABLY

    """

    env_obj = fetch_environment(env_name)  # fetching the env object

    if env_obj is None:
        print(f"{env_name} doesn't exist!")
        return

    env_map_metadata = fetch_map_metadata(env_obj)
    suitable_maps = []
    # suitable_times = suitable_timestamps(periodicities)
    time_costs = []
    map_timestamps = env_map_metadata['timestamp']

    maps_timestamps = [map_timestamp[0] for map_timestamp in map_timestamps]

    # TESTING DATA , SHOULD BE COMMENTED OUT TO WORK ON ACTUAL DATA
    # maps_timestamps = [1678863600, 1678864500, 1678865400, 1678866300, 1678867200, 1678868100, 1678869000, 1678869900,
    #                    1678870800, 1678871700, 1678872600, 1678892400, 1678914000, 1678950000, 1647327600]
    #                     Mar 15 2023 08:00:00,
    #                     Mar 15 2023 08:15:00,
    #                     Mar 15 2023 08:30:00
    #                     Mar 15 2023 08:45:00
    #                     Mar 15 2023 09:00:00,
    #                     Mar 15 2023 09:15:00
    #                     Mar 15 2023 09:30:00,
    #                     Mar 15 2023 09:45:00
    #                     Mar 15 2023 10:00:00
    #                     Mar 15 2023 10:15:00
    #                     Mar 15 2023 10:30:00,
    #                     Mar 15 2023 16:00:00
    #                     Mar 15 2023 22:00:00
    #                     Mar 16 2023 08:00:00
    #                     Mar 15 2022 08:00:00

    cost_ = time_cost_calc(maps_timestamps, periodicities, 1678863600)

    return cost_


def fetch_maps(env, map_to_fetch=None):
    """
    Fetches a map into ~.ros/  for an environment
    Args:
        map_to_fetch: name of the map to be fetched
        env: Environment for which maps are to be fetched
    Returns:
        downloads the map to ~.ros/
    """
    # check if the map exists on local
    local_path = f"{FETCHED_MAPS_PATH}/{map_to_fetch}"
    if os.path.isdir(local_path):
        print(f"{map_to_fetch} for {env} exists on local, fetching from db skipped....")
        return

    # deleting all existing fetched items from the fetched_objects directory first
    try:
        if not FETCHED_MAP_OBJ_PATH.is_dir():  # Creating the directory if it doesnt exist
            FETCHED_MAP_OBJ_PATH.mkdir(parents=True, exist_ok=True)

        all_files = os.listdir(f"{FETCHED_MAP_OBJ_PATH}/{env}/")
        for f in all_files:
            os.remove(f"{FETCHED_MAP_OBJ_PATH}/{f}")
    except:
        pass

    # downloading all info for the environment as a pkl file
    if not FETCHED_ENV_OBJ_PATH.is_dir():  # Creating the directory if it doesnt exist
        FETCHED_ENV_OBJ_PATH.mkdir(parents=True, exist_ok=True)

    try:
        CLIENT.fget_object(ENV_BUCKET, env, file_path=f"{FETCHED_ENV_OBJ_PATH}/{env}.pkl")
    except:
        print("Environment doesnt exist in db")
        return

    # reading the downloaded env pkl file
    with open(f"{FETCHED_ENV_OBJ_PATH}/{env}.pkl", 'rb') as f:
        map_data = pickle.load(f)

    # extracting map_metadata for the env i.e names of all the maps for the env
    maps_metadata_for_env = map_data.map_metadata

    # all the maps for the env fetched if map name not provided
    if not map_to_fetch:
        # getting all the maps from maps bucket for the env
        for map_ in maps_metadata_for_env['maps_names']:
            try:
                CLIENT.fget_object(MAP_BUCKET, f"{env}.{map_}.zip",
                                   file_path=f"{FETCHED_MAP_OBJ_PATH}/{env}/{env}.{map_}.zip")
                # unzipping into ~.ros
                with ZipFile(f"{FETCHED_MAP_OBJ_PATH}/{env}/{env}.{map_}.zip", 'r') as zObject:
                    zObject.extractall(path=f"{FETCHED_MAPS_PATH}")
                    print(f"{map_} fetched to {FETCHED_MAPS_PATH}")
            except:
                print(f"{map_} doesn't exist in the db for {env}")

    # only the map with the map name provided fetched for the env
    else:
        try:
            CLIENT.fget_object(MAP_BUCKET, f"{env}.{map_to_fetch}.zip",
                               file_path=f"{FETCHED_MAP_OBJ_PATH}/{env}/{env}.{map_to_fetch}.zip")
            # unzipping into ~.ros
            with ZipFile(f"{FETCHED_MAP_OBJ_PATH}/{env}/{env}.{map_to_fetch}.zip", 'r') as zObject:
                zObject.extractall(path=f"{FETCHED_MAPS_PATH}")
                print(f"{map_to_fetch} fetched to {FETCHED_MAPS_PATH}")
        except:
            print(f"{map_to_fetch} doesn't exist in the db for {env}")


def fetch_environment(env):
    """
    Fetches environment details
    Args:
        env: Environment which is to be fetched
    Returns:

    """
    if not FETCHED_ENV_OBJ_PATH.is_dir():  # Creating the directory if it doesn't exist
        FETCHED_ENV_OBJ_PATH.mkdir(parents=True, exist_ok=True)

    # deleting all existing fetched items from the directory first
    all_files = os.listdir(f"{FETCHED_ENV_OBJ_PATH}/")
    for f in all_files:
        os.remove(f"{FETCHED_ENV_OBJ_PATH}/{f}")

    # downloading all info for the environment as a pkl file
    try:
        CLIENT.fget_object(ENV_BUCKET, env, file_path=f"{FETCHED_ENV_OBJ_PATH}/{env}.pkl")

        # reading the downloaded env pkl file
        with open(f"{FETCHED_ENV_OBJ_PATH}/{env}.pkl", 'rb') as f:
            env_data = pickle.load(f)

        return env_data

    except:
        return


def fetch_first_images(env_obj):
    """
    Downloads the first image of every map of the environment to images directory
    Args:
        env_obj: Instance of Environment class for the environment
    """
    if not IMAGES_PATH.is_dir():  # Creating the directory if it doesn't exist
        IMAGES_PATH.mkdir(parents=True, exist_ok=True)

    images_to_fetch = []
    env_name = env_obj.name

    maps_names = env_obj.map_metadata['maps_names']

    for map_name in maps_names:
        images_to_fetch.append(f"{env_name}.{map_name}.jpg")

    for image in images_to_fetch:
        try:
            CLIENT.fget_object(FIRST_IMAGE_BUCKET, image, file_path=f"{IMAGES_PATH}/{image}")
            logging.info(f"Image: {image} downloaded to {IMAGES_PATH} ")
            # print(f"Image: {image} downloaded to {IMAGES_PATH} ")
        except:
            logging.critical(f"Image: {image} not found in {FIRST_IMAGE_BUCKET} bucket")
            # print(f"Image: {image} not found in {FIRST_IMAGE_BUCKET} bucket")
