from copyreg import pickle
import pickle
import os
from pathlib import Path
import argparse
from zipfile import ZipFile

from utils import load_map, create_buckets, env_upload, map_upload, map_upload2
from constants import ROOT, OBJECTS_PATH, CLIENT, ENV_BUCKET, MAP_BUCKET, FETCHED_MAP_OBJ_PATH, FETCHED_ENV_OBJ_PATH, \
    FETCHED_MAPS_PATH, DOT_ROS_PATH
from classes_ import Map, Environment
from shutil import make_archive


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-e', help="Name of the environment for which map needs to be fetched")
    parser.add_argument('-m', help="Name of the map that needs to be fetched")
    parser.add_argument('-oe', help="Name of the environment for which only env needs to be fetched")
    parser.add_argument('-u', help="Name of the map to upload")

    args = parser.parse_args()

    if args.oe and not args.e and not args.m:
        env_obj = fetch_environment(args.oe)
        if env_obj:
            env_map_metadata = env_obj.map_metadata
            print(env_map_metadata)
        else:
            print(f"{args.oe} doesn't exist in db")

    elif not args.e and not args.m and not args.u:
        create_buckets()  # create the buckets
        upload_objects2()  # upload to db

    elif args.u and args.e:

        # zipping the map
        path = f"{DOT_ROS_PATH}/{args.u}"
        make_archive(f"{DOT_ROS_PATH}/{args.e}.{args.u}", "zip", path)
        # with ZipFile(f"{DOT_ROS_PATH}/{args.e}.{args.u}.zip", 'w') as zip:
        #     for path, directories, files in os.walk(path):
        #         for file in files:
        #             file_name = os.path.join(path, file)
        #             zip.write(file_name) # zipping the file


        map_obj_name = f"{args.e}.{args.u}.zip"  # name to be used for the map object
        map_name = f"{args.e}.{args.u}"

        # Fetch env obj from db, append to it, then replace the one in db
        env_obj = fetch_environment(args.e)
        if env_obj:  # if the env exists in db then fetch it
            env_map_metadata = env_obj.map_metadata
            images, distances, trans, times = load_map(mappaths=f"{str(DOT_ROS_PATH)}/{args.u}")
            env_obj.map_metadata['maps_names'].append(map_name)
            # env_obj.map_metadata['images'].append(images)
            # env_obj.map_metadata['trans'].append(trans)
            env_obj.map_metadata['times'].append(times)
            env_obj.map_metadata['distances'].append(distances)

        else:  # else create the env obj
            env_obj = Environment(name=args.e, gps_position=None, nodes=None,
                                  edges=None)  # env object for the current environment
            images, distances, trans, times = load_map(mappaths=f"{str(DOT_ROS_PATH)}/{args.u}")

            env_obj.map_metadata['maps_names'].append(map_name)
            # env_obj.map_metadata['images'].append(images)
            # env_obj.map_metadata['trans'].append(trans)
            env_obj.map_metadata['times'].append(times)
            env_obj.map_metadata['distances'].append(distances)

        map_path = f"{str(DOT_ROS_PATH)}/{args.e}.{args.u}.zip"
        map_upload2(env_name=args.e, obj_name=map_obj_name, map_name=map_name, path=map_path)
        env_upload(env_data=env_obj)

        # Delete the zip file
        os.remove(f"{str(DOT_ROS_PATH)}/{args.e}.{args.u}.zip")

    else:
        fetch_maps2(args.e, args.m)
        # fetch_environment(args.e)


def extract_map_metadata(env_obj, map_name):
    """
    Adds the map_metadata for a given environment for the given map
    """
    maps_path = OBJECTS_PATH / "maps"
    env_name = env_obj.name

    with ZipFile(f"{ROOT}/objects/maps/{env_name}/{map_name}.zip", 'r') as zObject:
        zObject.extractall(path=f"{ROOT}/objects/maps/{env_name}/{map_name}")

    map_path = maps_path / env_name / map_name / map_name

    # loading all the maps of the current environment
    images, distances, trans, times = load_map(mappaths=str(map_path))

    env_obj.map_metadata['maps_names'].append(map_name)
    # env_obj.map_metadata['images'].append(images)
    # env_obj.map_metadata['trans'].append(trans)
    env_obj.map_metadata['times'].append(times)
    env_obj.map_metadata['distances'].append(distances)


def upload_objects2():
    maps_path = OBJECTS_PATH / "maps"

    number_of_environments = len(list(os.listdir(maps_path)))  # count of all the environments

    environments = list(os.listdir(maps_path))  # list of all the environments

    for i in range(number_of_environments):  # iterating over each envrionment
        env_obj = Environment(name=environments[i], gps_position=None, nodes=None,
                              edges=None)  # env object for the current environment

        dirs_in_env_dir = list(os.listdir((maps_path / environments[i])))
        maps = []
        for dir in dirs_in_env_dir:  # considering only the .zip files
            if dir.endswith('.zip'):
                maps.append(dir)

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
    Fetches maps as a zipped file for an environment
    Args:
        env: Environment for which maps are to be fetched
    Returns:
        Zipped maps
    """
    global CLIENT

    # check if the map exists on local
    local_path = f"{FETCHED_MAPS_PATH}/{env}/{map_to_fetch}"
    if os.path.isdir(local_path):
        print(f"{map_to_fetch} for {env} exists on local, fetching from db skipped....")
        return

    # deleting all existing fetched items from the fetched_objects directory first
    try:
        all_files = os.listdir(f"{FETCHED_MAP_OBJ_PATH}/{env}/")
        for f in all_files:
            os.remove(f"{FETCHED_MAP_OBJ_PATH}/{f}")
    except:
        pass

        # downloading all info for the environment as a pkl file
    CLIENT.fget_object(ENV_BUCKET, env, file_path=f"{FETCHED_ENV_OBJ_PATH}/{env}.pkl")

    # reading the downloaded env pkl file
    with open(f"{FETCHED_ENV_OBJ_PATH}/{env}.pkl", 'rb') as f:
        map_data = pickle.load(f)

        # extracting map_metadata for the env i.e names of all the maps for the env
    maps_metadata_for_env = map_data.map_metadata

    # all the maps for the env fetched if map name not provided
    if not map_to_fetch:
        # getting all the maps from maps bucket for the env
        for map_ in maps_metadata_for_env['maps_names']:
            CLIENT.fget_object(MAP_BUCKET, f"{env}.{map_}.zip",
                               file_path=f"{FETCHED_MAP_OBJ_PATH}/{env}/{env}.{map_}.zip")
            # unzipping into ~.ros
            with ZipFile(f"{FETCHED_MAP_OBJ_PATH}/{env}/{env}.{map_}.zip", 'r') as zObject:
                zObject.extractall(path=f"{FETCHED_MAPS_PATH}/{env}")
                print(f"{map_} fetched to {FETCHED_MAPS_PATH}/{env}")

    # only the map with the map name provided fetched for the env
    else:
        try:
            CLIENT.fget_object(MAP_BUCKET, f"{env}.{map_to_fetch}.zip",
                               file_path=f"{FETCHED_MAP_OBJ_PATH}/{env}/{env}.{map_to_fetch}.zip")
            # unzipping into ~.ros
            with ZipFile(f"{FETCHED_MAP_OBJ_PATH}/{env}/{env}.{map_to_fetch}.zip", 'r') as zObject:
                zObject.extractall(path=f"{FETCHED_MAPS_PATH}/{env}")
                print(f"{map_to_fetch} fetched to {FETCHED_MAPS_PATH}/{env}")
        except:
            raise Exception(f"{map_to_fetch} doesn't exist in the db for {env}")


def fetch_environment(env):
    """
    Fetches environment details
    Args:
        env: Environment which is to be fetched
    Returns:
        A dict with key = "env name" and value = Environment class objects
    """
    global CLIENT
    # try:
    #     response = client.get_object(ENV_BUCKET, env)
    # finally:
    #     response.close()
    #     response.release_conn()  

    # return response.getheaders()  # TODO: HOW TO READ THE DATA RETURNED BY get_object ???

    # TODO: The block needs to be replaced by the block above
    # Workaround until get_object starts working

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


main()
