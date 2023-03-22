import os
from pathlib import Path
from zipfile import ZipFile

from constants import TO_UPLOAD_PATH, OBJECTS_PATH, ENV_BUCKET, MAP_BUCKET, FIRST_IMAGE_BUCKET, DOT_ROS_PATH, CLIENT
from classes_ import Environment
from fetch_utils import fetch_environment
from meta_data_extraction import extract_map_metadata
from data_manipulation import extract_map_metadata_manipulated, manipulated_map_upload


def create_buckets():
    """
    Creates empty 'environment' and 'maps' buckets in the db
    """
    found = CLIENT.bucket_exists(MAP_BUCKET)
    if not found:
        CLIENT.make_bucket(MAP_BUCKET)
        print(f"Bucket {MAP_BUCKET} created")
    # else:
    #     print(f"Bucket {MAP_BUCKET} already exists")

    found = CLIENT.bucket_exists(ENV_BUCKET)
    if not found:
        CLIENT.make_bucket(ENV_BUCKET)
        print(f"Bucket {ENV_BUCKET} created")
    # else:
    #     print(f"Bucket {ENV_BUCKET} already exists")

    found = CLIENT.bucket_exists(FIRST_IMAGE_BUCKET)
    if not found:
        CLIENT.make_bucket(FIRST_IMAGE_BUCKET)
        print(f"Bucket {FIRST_IMAGE_BUCKET} created")
    # else:
    #     print(f"Bucket {ENV_BUCKET} already exists")

def env_upload(env_data):
    """
    Uploads an environment object  to db
    Args:
        A object instance of class Environment   
    """
    if not TO_UPLOAD_PATH.is_dir():  # Creating the directory if it doesn't exist
        TO_UPLOAD_PATH.mkdir(parents=True, exist_ok=True)
    obj_name = env_data.name

    env_data.pickle_env()

    # env object is always updated in db even if it already exists
    CLIENT.fput_object(bucket_name=ENV_BUCKET, object_name=obj_name,
                       file_path=str(TO_UPLOAD_PATH) + "/" + 'pickled_env.pkl')


    print(f"Environment {obj_name} uploaded to {ENV_BUCKET} bucket")


def OBSOLETE_map_upload(map_data):
    """
    OBSOLETE !!! DO NOT USE !!!
    Uploads a map object  to db
    Args:
        A object instance of class Map   
    """

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


def zip_the_map(env_name, map_name, path_to_directory_containing_map_directory):
    """
    Zips a directory containing a map
    Args:
        env_name: Name of the environment to which the map belongs
        map_name: Name of the map
        path_to_directory_containing_map_directory: Location of directory that has the directory that contains the map on disk

    Returns: Location of the specific map on disk
    """
    if path_to_directory_containing_map_directory is None:
        path = Path(f"{DOT_ROS_PATH}/{map_name}")  # location of map in local ( ~/.ros/ )
        zip_file_name = f"{DOT_ROS_PATH}/{env_name}.{map_name}.zip"
        location_of_map = path
    else:
        path = Path(f"{path_to_directory_containing_map_directory}/{map_name}")
        zip_file_name = f"{path_to_directory_containing_map_directory}/{env_name}.{map_name}.zip"
        location_of_map = path  # NOT the location of zipped file

    # if the map doesn't exist on local
    if not path.is_dir():
        print(f"The map {map_name} doesn't exist in local")
        return None

    # zipping the map
    with ZipFile(zip_file_name, 'w') as zip:
        for path, directories, files in os.walk(path):
            for file in files:
                out_file_name = os.path.join(path, file)
                zip.write(out_file_name, arcname=f"{map_name}/{os.path.basename(out_file_name)}")  # zipping the file

    return location_of_map


def map_upload(env_name, map_name, start_node, end_node, path_to_directory_containing_map_directory=None):
    """
    Uploads a zipped map to db
    Args:
        path_to_directory_containing_map_directory: path provided if maps are not to be uploaded from the default ~/.ros/ directory
        env_name: name of the environment
        map_name: map name on local
        start_node: name of the first node of the map
        end_node: name of the second node of the map
    """

    create_buckets()  # create the buckets in case they do not exist in the db

    map_obj_name = f"{env_name}.{map_name}.zip"  # name of map object in db

    # zip the map
    location_of_map = zip_the_map(env_name=env_name, map_name=map_name,
                                  path_to_directory_containing_map_directory=path_to_directory_containing_map_directory)

    if location_of_map is None:  # map doesn't exist in local
        return

    # uploading the first image of the map
    first_image_upload(env_name, map_name)

    # Fetch env obj from db, append to it, then replace the one in db
    env_obj = fetch_environment(env_name)
    if env_obj:  # if the env exists in db then update it
        if map_name not in env_obj.map_metadata['maps_names']:  # adding the name of the map to metadata if doesn't exist
            env_obj = extract_map_metadata(env_obj=env_obj, map_name=map_name, start_node_name=start_node,
                                           end_node_name=end_node, path=location_of_map)

    else:  # else create the env obj
        env_obj = Environment(name=env_name, gps_position=None)  # env object for the current environment
        env_obj = extract_map_metadata(env_obj=env_obj, map_name=map_name, start_node_name=start_node,
                                       end_node_name=end_node, path=location_of_map)

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


def first_image_upload(env_name, map_name, path_to_directory_containing_map_directory=None):
    """
    Uploads the first image of a map to first-image bucket in the db
    Args:
        env_name: name of the environment to which the map belongs
        map_name: name of the map
        path_to_directory_containing_map_directory: path to the map on local
    """
    image_obj_name = f"{env_name}.{map_name}.jpg"
    map_path = Path(f"{DOT_ROS_PATH}/{map_name}")

    # finding the first image for the map when uploading
    if not map_path.is_dir():
        print(f"The map {map_name} doesn't exist in local")
        return None

    images = []
    for path, directories, files in os.walk(map_path):
        for file in files:
            if file.endswith(".jpg"):
                images.append(file)
    images.sort()
    first_image_name = images[0]
    first_image_path = f"{DOT_ROS_PATH}/{map_name}/{first_image_name}"

    # Uploading the first image
    try:
        statobj = CLIENT.stat_object(FIRST_IMAGE_BUCKET, image_obj_name, ssec=None, version_id=None, extra_query_params=None)
        # print(f"{image_obj_name} already exists in {FIRST_IMAGE_BUCKET} bucket")
    except:
        CLIENT.fput_object(bucket_name=FIRST_IMAGE_BUCKET, object_name=image_obj_name, file_path=first_image_path)
        # print(f"Image {image_obj_name} uploaded to {FIRST_IMAGE_BUCKET} bucket")


def batch_upload():
    """
    Uploads all the maps in maps_path directory and creates the corresponding env objects and uploads them.
    NOTE: Start nodes, end nodes, and any other that need to be MANIPULATED should be manually entered in the method.
    NOTE: To manipulate data, MANIPULATE flag should be set to True manually
    """

    # ******************************************************************************
    # to be set by the user manually to give names to nodes
    START_NODE_NAMES = ['a', 'b', 'c', 'd', 'e', 'f', 'b', 'd', 'c', 'b']
    END_NODE_NAMES = ['b', 'c', 'd', 'e', 'f', 'a', 'e', 'g', 'g', 'e']

    # *******************************************************************************
    MANIPULATE = True  # ATTENTION !! Data is probably being manipulated !!
    # *******************************************************************************
    DISTANCE = [2, 4, 1, 1, 2, 8, 3, 1, 6, 1]
    COST =     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    TIMESTAMPS = [None, None, None, None, None, None, None, None, None, None]
    # *******************************************************************************

    maps_path = OBJECTS_PATH / "maps"  # path of maps on local

    if not maps_path.is_dir():
        maps_path.mkdir(parents=True, exist_ok=True)

    number_of_environments = len(list(os.listdir(maps_path)))  # count of all the environments

    if number_of_environments == 0:
        print(
            f"There are no maps in {maps_path} directory.")
        return

    environments = list(os.listdir(maps_path))  # list of all the environments

    # iterating over each environment
    for i in range(number_of_environments):

        dirs_in_env_dir = list(os.listdir((maps_path / environments[i])))

        dirs_in_env_dir.sort()  # ATTENTION! Sorting maps names. Hence naming of maps important when adding data (distance etc) manually.

        maps = []

        for dir_ in dirs_in_env_dir:
            if not dir_.endswith(".zip"):
                maps.append(dir_)

        number_of_maps = len(maps)

        # iterating over each map in the current environment
        for j in range(number_of_maps):

            map_name = maps[j]

            start_node_name = START_NODE_NAMES[j]
            end_node_name = END_NODE_NAMES[j]

            if not MANIPULATE:
                map_upload(env_name=environments[i],
                           map_name=map_name,
                           start_node=start_node_name,
                           end_node=end_node_name,
                           path_to_directory_containing_map_directory=f"{maps_path}/{environments[i]}")

            elif MANIPULATE:
                manipulated_map_upload(env_name=environments[i],
                                       map_name=map_name,
                                       start_node=start_node_name,
                                       end_node=end_node_name,
                                       manipulated_distance= DISTANCE[j],
                                       manipulated_cost=COST[j],
                                       path_to_directory_containing_map_directory=f"{maps_path}/{environments[i]}", )


