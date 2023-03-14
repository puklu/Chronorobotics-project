import os
from zipfile import ZipFile
from copyreg import pickle
import pickle

from constants import FETCHED_MAPS_PATH, FETCHED_MAP_OBJ_PATH, FETCHED_ENV_OBJ_PATH, ENV_BUCKET, MAP_BUCKET, CLIENT
from cost_calculation import suitable_timestamps, time_cost_calc


def print_env_details(env_name):
    """
    Prints the contents of env class (for easy testing)
    Args:
        env_name: name of the environment for which details need to be printed
     """
    env_obj = fetch_environment(env_name)  # fetching the env object
    meta_data_dict = fetch_map_metadata(env_obj)  # getting map metadata from it

    if meta_data_dict:
        print(f"env name: {env_obj.name}")
        print(f"nodes in the env: {env_obj.nodes_names}")
        print(f"maps in the env: {meta_data_dict['maps_names']}")
        print(f"distance: {meta_data_dict['distance']}")
        print(f"timestamps: {meta_data_dict['timestamp']}")
        print(f"times: {meta_data_dict['times'][0][0][0]}")
        print(f"times: {meta_data_dict['times'][0][0][0].to_time()}")

        # print(f"START NODES")
        # for snode in meta_data_dict['start_node']:
        #     print(snode.key)
        #
        # print()
        # print(f"END NODES")
        # for enode in meta_data_dict['end_node']:
        #     print(enode.key)

        print(f"Nodes and their neighbours are")
        for node in env_obj.nodes:
            # print(f"Weight and cost of the node are: {node.g_cost} | {node.h_cost}")
            print(f"Neighbours of {node.key} with distance are:", end=' ')
            for neighbour in node.neighbours:
                print(f"{neighbour[0].key}: {neighbour[2]}", end=' | ')
            print()

        # print(env_obj.map_metadata)
        # for node in env_obj.nodes:
        #     print()

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
    env_obj = fetch_environment(env_name)  # fetching the env object
    env_map_metadata = fetch_map_metadata(env_obj)
    suitable_maps = []
    # suitable_times = suitable_timestamps(periodicities)
    time_costs = []
    map_timestamps = env_map_metadata['timestamp']

    for map_timestamp in map_timestamps:
        cost_ = time_cost_calc(map_timestamp[0], periodicities)
        time_costs.append(cost_)

    ten_least_costs = time_costs.copy()
    ten_least_costs.sort()
    if len(ten_least_costs) >= 10:
        ten_least_costs = ten_least_costs[0:10]

    # maps to fetch
    for i in range(len(ten_least_costs)):
        # TODO: ACTUAL FETCHING OF THE MAP AFTER COMPARING FEATURES
        map_idx = time_costs.index(ten_least_costs[i])
        map_name = env_map_metadata['maps_names'][map_idx]
        print(map_name)


def fetch_maps_according_to_time(env_name, periodicities):
    # TODO: MIGHT HAVE TO BE REPLACED BY THE METHOD ABOVE
    env_obj = fetch_environment(env_name)  # fetching the env object
    env_map_metadata = fetch_map_metadata(env_obj)
    suitable_maps = []
    suitable_times = suitable_timestamps(periodicities)

    # print(suitable_times)

    for suitable_time in suitable_times:
        for timestamp in env_map_metadata['timestamp']:

            if suitable_time[0] < timestamp[0] < suitable_time[1]:
                map_idx = env_map_metadata['timestamp'].index(timestamp)
                suitable_maps.append(env_map_metadata['maps_names'][map_idx])
                # print(env_map_metadata['maps_names'][map_idx])

    # fetching maps form db
    if len(suitable_maps) == 0:
        print("No suitable maps according to time!")
        return
    else:
        for map_ in suitable_maps:
            fetch_maps(env_name, map_)


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
