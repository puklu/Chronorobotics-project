from copyreg import pickle
import pickle
import os
from pathlib import Path
import uuid
import urllib3
import argparse

from utils import load_map, create_buckets, env_upload, map_upload
from constants import ROOT, OBJECTS_PATH, CLIENT, ENV_BUCKET, MAP_BUCKET
from classes_ import Map, Environment

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', help="Name of the environment for which maps need to be fetched")
    parser.add_argument('-e', help="Name of the environment for which needs to be fetched")
    args = parser.parse_args()

    create_buckets()    # create the buckets

    upload_objects()    # upload all the maps for all the environments

    # if only the maps need to be fetched
    if args.m and not args.e:  
        env_reached = args.m
        maps = fetch_maps(env_reached)  # a dictionary containing all the maps for the environment
        # environment = fetch_environment(env_reached)  # dictionary containing the environment

        print(maps)
        
    # if only environment needs to be fetched
    elif args.e and not args.m:
        env_reached = args.e
        environment = fetch_environment(env_reached)   # dictionary containing the environment

        print(environment[env_reached].name)

    # if both environment and maps need to be fetched
    elif args.e and args.m:
        map_reached = args.m
        env_reached = args.e
        maps = fetch_maps(map_reached)  # a dictionary containing all the maps for the environment
        environment = fetch_environment(env_reached)  # dictionary containing the environment

        print(environment, maps)

    # if no argument is provided
    else:
        raise Exception("No environment provided")
     

      
def upload_objects():

    maps_path = OBJECTS_PATH / "maps"

    number_of_environments = len(list(os.listdir(maps_path)))  # count of all the environments
    environments = list(os.listdir(maps_path))  # list of all the environments

    for i in range(number_of_environments):   # iterating over each envrionment
        env_obj =  Environment(name=environments[i], gps_position=None, nodes=None, edges=None) # env object for the cirrent environment

        number_of_maps = len(list(os.listdir((maps_path / environments[i])))) # count of all the maps for the current environment

        map_paths_for_env = []

        for j in range(number_of_maps):
            # for loading the maps for the environment
            images = []
            distances = []
            trans = []
            times= []

            maps = list(os.listdir((maps_path / environments[i])))  

            map_paths_for_env.append(str(maps_path) + '/'+ environments[i]+'/'+maps[j]) # paths of all the maps for the current environment
            mappaths = ','.join(map_paths_for_env)

            # laoding all the maps of the current environment
            images, distances, trans, times = load_map(mappaths=mappaths)

            # Map object for the map
            name = f"{environments[i]}.{maps[j]}"  # name to be used for the map object
            map_obj = Map(name=name, start_node=None, end_node=None , images=images, distances=distances, trans=trans, times=times)

            # updating the map_metadata for the environment
            env_obj.map_metadata.append(name)

            # upload the map object
            map_upload(map_data=map_obj)

        # upload the environment object
        env_upload(env_data=env_obj)


def fetch_maps(env):
    """
    Fetches maps for an environment
    Args:
        env: Environment for which maps are to be fetched
    Returns:
        A dict with keys of format "env name"."map name" and values = Map class objects
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
    maps = {}

    # deleting all existing fetched items from the directory first
    all_files = os.listdir(f"{ROOT}/fetched_objects/maps/")
    for f in all_files:
        os.remove(f"{ROOT}/fetched_objects/maps/{f}")

    # downloading all info for the environment as a pkl file
    CLIENT.fget_object(ENV_BUCKET, env, file_path = f"{ROOT}/fetched_objects/environment/{env}.pkl")
    
    # reading the downloaded env pkl file
    with open(f"{ROOT}/fetched_objects/environment/{env}.pkl", 'rb') as f:
        map_data = pickle.load(f)   
    
    # extracting map_metadata for the env i.e names of all the maps for the env
    maps_metadata_for_env = map_data.map_metadata

    # getting all the maps from maps bucket for the env
    for map_ in maps_metadata_for_env:
        CLIENT.fget_object(MAP_BUCKET, map_, file_path = f"fetched_objects/maps/maps.{env}.pkl")
        with open(f"fetched_objects/maps/maps.{env}.pkl", 'rb') as f:
            maps[map_] = pickle.load(f)  # storing the MAP objects in a dictionary
    
    return maps  # returning the maps as a dictionary where the key is the map name
    

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
    environment = {}

    # deleting all existing fetched items from the directory first
    all_files = os.listdir(f"{ROOT}/fetched_objects/environment/")
    for f in all_files:
        os.remove(f"{ROOT}/fetched_objects/environment/{f}")

    # downloading all info for the environment as a pkl file
    CLIENT.fget_object(ENV_BUCKET, env, file_path = f"{ROOT}/fetched_objects/environment/{env}.pkl")
    
    # reading the downloaded env pkl file
    with open(f"fetched_objects/environment/{env}.pkl", 'rb') as f:
        env_data = pickle.load(f)   
    
    environment[env] = env_data
      
    return environment  # returning the env as a dictionary where the key is the env name
    

main()

    

    
    



    

   

     





    

