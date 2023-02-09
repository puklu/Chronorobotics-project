import os
from zipfile import ZipFile
from copyreg import pickle
import pickle
import numpy as np
from constants import ENV_BUCKET, MAP_BUCKET, CLIENT, TO_UPLOAD_PATH, OBJECTS_PATH, ROOT, OBJECTS_PATH, ENV_BUCKET, \
    MAP_BUCKET, FETCHED_MAP_OBJ_PATH, FETCHED_ENV_OBJ_PATH, \
    FETCHED_MAPS_PATH, DOT_ROS_PATH
from classes_ import Features, Map, Environment


def load_map(mappaths):
    images = []
    distances = []
    trans = []
    times = []

    if "," in mappaths:
        mappaths = mappaths.split(",")
    else:
        mappaths = [mappaths]

    for map_idx, mappath in enumerate(mappaths):
        tmp = []
        for file in list(os.listdir(mappath)):
            if file.endswith(".npy"):
                tmp.append(file[:-4])
        print(str(len(tmp)) + " images found in the map")

        # rospy.logwarn(str(len(tmp)) + " images found in the map")
        tmp.sort(key=lambda x: float(x))

        # print(map_idx, tmp)

        tmp_images = []
        tmp_distances = []
        tmp_trans = []
        tmp_times = []

        for idx, dist in enumerate(tmp):

            tmp_distances.append(float(dist))
            feature = Features()
            with open(os.path.join(mappath, dist + ".npy"), 'rb') as fp:
                map_point = np.load(fp, allow_pickle=True, fix_imports=False).item(0)
                r = map_point["representation"]
                ts = map_point["timestamp"]
                diff_hist = map_point["diff_hist"]

                if map_idx > 0 and map_point["source_map_align"] != mappaths[0]:
                    print("Multimap with invalid target!" + str(mappath))
                    # rospy.logwarn("Multimap with invalid target!" + str(mappath))
                    raise Exception("Invalid map combination")

                feature.shape = r.shape
                feature.values = r.flatten()
                tmp_images.append(feature)
                tmp_times.append(ts)

                if diff_hist is not None:
                    tmp_trans.append(diff_hist)
                # rospy.loginfo("Loaded feature: " + dist + str(".npy"))
                # print("Loaded feature: " + dist + str(".npy"))

        tmp_times[-1] = tmp_times[-2]  # to avoid very long period before map end
        images.append(tmp_images)
        distances.append(tmp_distances)
        trans.append(tmp_trans)
        times.append(tmp_times)
        # rospy.logwarn("Whole map " + str(mappath) + " sucessfully loaded")
        print("Whole map " + str(mappath) + " successfully loaded")

        return images, distances, trans, times


def create_buckets():
    """
    Creates empty 'environment' and 'maps' buckets in the db
    """
    global CLIENT
    # create the buckets
    found = CLIENT.bucket_exists(MAP_BUCKET)
    if not found:
        CLIENT.make_bucket(MAP_BUCKET)
        print(f"Bucket {MAP_BUCKET} created")
    else:
        print(f"Bucket {MAP_BUCKET} already exists")

    found = CLIENT.bucket_exists(ENV_BUCKET)
    if not found:
        CLIENT.make_bucket(ENV_BUCKET)
        print(f"Bucket {ENV_BUCKET} created")
    else:
        print(f"Bucket {ENV_BUCKET} already exists")


def env_upload(env_data):
    """
    Uploads an environment object  to db
    Args:
        A object instance of class Environment   
    """
    global CLIENT

    if not TO_UPLOAD_PATH.is_dir():  # Creating the directory if it doesnt exist
        TO_UPLOAD_PATH.mkdir(parents=True, exist_ok=True)

    obj_name = env_data.name

    env_data.pickle_env()

    # env object is always updated in db even if it already exists
    CLIENT.fput_object(bucket_name=ENV_BUCKET, object_name=obj_name,
                       file_path=str(TO_UPLOAD_PATH) + "/" + 'pickled_env.pkl')
    print(f"Environment {obj_name} uploaded to {ENV_BUCKET} bucket")


def map_upload(map_data):
    """
    Uploads a map object  to db
    Args:
        A object instance of class Map   
    """
    global CLIENT
    obj_name = map_data.name

    # check if the map exists in db
    try:
        statobj = CLIENT.stat_object(MAP_BUCKET, obj_name, ssec=None, version_id=None, extra_query_params=None)
        print(f"{obj_name} already exists in {MAP_BUCKET} bucket")

    except:
        map_data.pickle_map()
        CLIENT.fput_object(bucket_name=MAP_BUCKET, object_name=obj_name,
                           file_path=str(TO_UPLOAD_PATH) + "/" + 'pickled_map.pkl',
                           metadata={"env_id": map_data.env_id})
        print(f"Map {obj_name} uploaded to {MAP_BUCKET} bucket")


def map_upload2(env_name, obj_name, map_name, path):
    """
    Uploads a zipped map to db
    Args:
        path: Path from where file needs to be uploaded
        env_name: name of the environment
        obj_name: name of the map object to be set in the bucket
        map_name: map name on local
    """

    global CLIENT

    # check if the map exists in db
    try:
        statobj = CLIENT.stat_object(MAP_BUCKET, obj_name, ssec=None, version_id=None, extra_query_params=None)
        print(f"{obj_name} already exists in {MAP_BUCKET} bucket")

    except:
        CLIENT.fput_object(bucket_name=MAP_BUCKET, object_name=obj_name, file_path=path)
        print(f"Map {obj_name} uploaded to {MAP_BUCKET} bucket")


def extract_map_metadata2(env_obj, map_name, start_node, end_node):
    """
    Extracts the meta_data of an environment when a map is uploaded for an environment.
    Also updates the necessary env variables.
    Args:
        env_obj: Instance of Environment class
        map_name: Name of the map being uploaded
        start_node: Name of the starting node of the map
        end_node: Name of the ending node of the map

    Returns: An updated env object

    """
    images, distances, trans, times = load_map(mappaths=f"{str(DOT_ROS_PATH)}/{map_name}")

    distance = distances[0][-1]

    env_obj.map_metadata['maps_names'].append(map_name)
    # env_obj.map_metadata['images'].append(images)
    # env_obj.map_metadata['trans'].append(trans)
    env_obj.map_metadata['times'].append(times)
    env_obj.map_metadata['distances'].append(distances)
    env_obj.map_metadata['distance'].append(distance)  # the length of the path
    env_obj.map_metadata['start_node'].append(start_node)
    env_obj.map_metadata['end_node'].append(end_node)

    # adding the nodes to environment object
    if start_node not in env_obj.nodes:
        env_obj.nodes.append(start_node)
    if end_node not in env_obj.nodes:
        env_obj.nodes.append(end_node)

    return env_obj


def extract_map_metadata(env_obj, map_name):
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


def upload_objects2():
    """
    Uploads all the maps in maps_path directory and creates the corresponding env objects and uploads them.
    """
    maps_path = OBJECTS_PATH / "maps"

    if not maps_path.is_dir():
        maps_path.mkdir(parents=True, exist_ok=True)

    number_of_environments = len(list(os.listdir(maps_path)))  # count of all the environments

    if number_of_environments == 0:
        print(
            f"There are no maps in {maps_path} directory in the project. NOT UPLOADING ANYTHING! Please place the zipped maps here.")
        return

    environments = list(os.listdir(maps_path))  # list of all the environments

    for i in range(number_of_environments):  # iterating over each environment

        # first check if the env already exists in the db
        if fetch_environment(environments[i]):
            env_obj = fetch_environment(environments[i])  # if the env exists in db then use it
        else:
            env_obj = Environment(name=environments[i],
                                  gps_position=None)  # else env object for the current environment

        dirs_in_env_dir = list(os.listdir((maps_path / environments[i])))
        maps = []
        for dir_ in dirs_in_env_dir:  # considering only the .zip files
            if dir_.endswith('.zip'):
                maps.append(dir_)

        number_of_maps = len(maps)

        for j in range(number_of_maps):
            map_name = maps[j]

            map_obj_name = f"{environments[i]}.{maps[j]}"  # name to be used for the map object

            # adding all the map_metadata for the env
            extract_map_metadata(env_obj, map_name[:-4])

            # upload the map file
            map_path = f"{str(OBJECTS_PATH)}/maps/{environments[i]}/{map_name}"
            map_upload2(env_name=environments[i], obj_name=map_obj_name, map_name=map_name, path=map_path)

        env_upload(env_data=env_obj)


def fetch_maps2(env, map_to_fetch):
    """
    Fetches maps into ~.ros/  for an environment
    Args:
        map_to_fetch: name of the map to be fetched
        env: Environment for which maps are to be fetched
    Returns:
        map(s) to ~.ros/
    """
    global CLIENT

    # check if the map exists on local
    local_path = f"{FETCHED_MAPS_PATH}/{env}/{map_to_fetch}"
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
        A dict with key = "env name" and value = Environment class objects
    """
    global CLIENT

    if not FETCHED_ENV_OBJ_PATH.is_dir():  # Creating the directory if it doesnt exist
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


def delete_a_map(env_name, map_name):
    """
    Deletes a map for an environment and updated the corresponding env object
    Args:
        env_name: The name of the environment to which the map belongs
        map_name: Name of the map to be deleted
    """
    map_obj_name = f"{env_name}.{map_name}.zip"

    CLIENT.remove_object(MAP_BUCKET, map_obj_name)  # map deleted

    env_obj = fetch_environment(env_name)  # fetching the env details

    if env_obj:
        maps_in_env = env_obj.map_metadata['maps_names']  # maps list in env before updating (map deleted at this point)
        if map_name in maps_in_env:
            idx = maps_in_env.index(map_name)  # index of the deleted map in map_metadata

            # deleting all the data at the index corresponding to the deleted map
            del env_obj.map_metadata['maps_names'][idx]
            del env_obj.map_metadata['distance'][idx]
            del env_obj.map_metadata['start_node'][idx]
            del env_obj.map_metadata['end_node'][idx]

            # updating the nodes of the environment
            env_obj.nodes = []
            for snode in env_obj.map_metadata['start_node']:
                if snode not in env_obj.nodes:
                    env_obj.nodes.append(snode)
            for enode in env_obj.map_metadata['end_node']:
                if enode not in env_obj.nodes:
                    env_obj.nodes.append(enode)

            if len(env_obj.nodes) > 0:  # if there are still nodes in the environment
                env_upload(env_data=env_obj)  # uploading the updated env object
            else:  # if there are no nodes left after deleting the map, then delete the env_obj
                CLIENT.remove_object(ENV_BUCKET, env_name)  # env object deleted

            print(f"Map: {map_name} deleted and environment: {env_name} updated")

        else:
            print(f"map: {map_name} doesn't exist in db for the env: {env_name}")

    else:
        print(f"Environment: {env_name} doesn't exist in db")


def delete_all_maps_of_an_environment(env_name):
    """
    Deletes all the maps from an environment followed by the deletion of the env object
    Args:
        env_name: Name of the environment for which all maps need to be deleted

    """
    env_obj = fetch_environment(env_name)  # fetching the env details

    if env_obj:
        list_of_maps_in_the_env = env_obj.map_metadata['maps_names']

        # deleting all the maps
        for map_ in list_of_maps_in_the_env:
            map_obj_name = f"{env_name}.{map_}.zip"
            try:
                CLIENT.remove_object(MAP_BUCKET, map_obj_name)  # map deleted
                print(f"map: {map_} deleted from env: {env_name}")
            except:
                print(f"{map_} doesn't exist in the db, so nothing deleted ")

        # deleting the env_obj
        CLIENT.remove_object(ENV_BUCKET, env_name)  # env object deleted

    else:
        print(f"Environment: {env_name} doesn't exist in db")
