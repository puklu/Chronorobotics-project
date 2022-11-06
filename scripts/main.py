from copyreg import pickle
import pickle
import os
from pathlib import Path
import uuid
import urllib3
import argparse

from utils import load_map, create_buckets
from constants import ROOT, OBJECTS_PATH, DOWNLOADED_ENVS_PATH, DOWNLOADED_MAPS_PATH, TO_UPLOAD_PATH, CLIENT, ENV_BUCKET, MAP_BUCKET


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-e', help="Name of the environment for which maps need to be fetched")
    args = parser.parse_args()

    create_buckets()    # TODO: to uncomment

    upload_objects()    # TODO: to uncomment

    if args.e:
        env_reached = args.e
        maps = fetch_maps(env_reached)  # gets all the maps for the env
    else:
        print("No environment provided")  
        env_reached = 'env0'   # TODO: just for testing, remove later
        maps = fetch_maps(env_reached)  

    for key in maps:  # printing out the maps
        print(maps[key])


class Features():
    def __init__(self) -> None:
        self.shape = None
        self.values = None


class Environment:
    def __init__(self, name, gps_position, nodes, edges) -> None:
        self.name = name
        self.gps_position = gps_position
        self.nodes = nodes
        self.edges = edges
        self.map_metadata = []

    # def __repr__(self):
    #     return {"name": self.name,
    #             "gps_position": self.gps_position,
    #             "nodes": self.nodes,
    #             "edges":self.edges,
    #             "map_metadata": self.map_metadata}    

    def pickle_env(self):
        with open(str(TO_UPLOAD_PATH)+ "/" + "pickled_env.pkl", 'wb') as f:
            pickle.dump(self, f)
        
    def read_pickled_env(self,file_path):
        with open(file_path, 'rb') as f:
            env_data = pickle.load(f)   
        return env_data  


class Map:
    def __init__(self, name, start_node, end_node, images, distances, trans, times):
        self.env_id = None
        self.name = name
        self.start_node = start_node
        self.end_node = end_node
        self.images = images
        self.distances = distances
        self.trans = trans
        self.times = times
        
        # self.env_id = self.name+"."+self.start_node+"."+self.end_node
        
    # def __repr__(self):
    #     return {"env_id": self.env_id,
    #             "name" : self.name,
    #             "start_node": self.start_node,
    #             "end_node": self.end_node,
    #             "image": self.images,
    #             "distance": self.distances,
    #             "trans": self.trans,
    #             "time":self.times}

    def pickle_map(self):
        with open(str(TO_UPLOAD_PATH) + "/" + "pickled_map.pkl", 'wb') as f:
            pickle.dump(self, f)
        

    def read_pickled_map(self,file_path):
        with open(file_path, 'rb') as f:
            map_data = pickle.load(f)   
        return map_data

    def read(self):   
        pickled_object = pickle.dumps(self)
        return pickled_object



def map_upload(map_data):
    """
    Uploads a map object  to db
    Args:
        A object instance of class Map
    Returns:
        env_id of the object    
    """
    global CLIENT
    obj_name = map_data.name 

    # check if the map exists in db
    try:
        statobj = CLIENT.stat_object(MAP_BUCKET, obj_name, ssec=None, version_id=None, extra_query_params=None)
        print(f"{obj_name} already exists in {MAP_BUCKET}")
        
    except:
        # TODO: To change it to put_object
        # client.put_object(bucket_name=MAP_BUCKET, object_name=obj_name,data=(map_data), length=5000000)#, part_size = 50000000)
        map_data.pickle_map()
        CLIENT.fput_object(bucket_name=MAP_BUCKET, object_name=obj_name,file_path=str(TO_UPLOAD_PATH)+ "/" +'pickled_map.pkl', metadata={"env_id":map_data.env_id})
        print(f"Map {obj_name} uploaded to {MAP_BUCKET}")     


def env_upload(env_data):
    """
    Uploads an environment object  to db
    Args:
        A object instance of class Environment   
    """
    global CLIENT
    obj_name= env_data.name

    # check if the env variables exist for the map in db
    try:
        statobj = CLIENT.stat_object(ENV_BUCKET, obj_name, ssec=None, version_id=None, extra_query_params=None)
        print(f"{obj_name} already exists in {ENV_BUCKET}")
    except:
        # TODO: To change it to put_object
        # client.put_object(bucket_name=ENV_BUCKET, object_name=obj_name,data=(env_data), length=5000000)#, part_size = 50000000)   
        env_data.pickle_env()
        CLIENT.fput_object(bucket_name=ENV_BUCKET, object_name=obj_name,file_path=str(TO_UPLOAD_PATH)+ "/" + 'pickled_env.pkl')
        print(f"Environment {obj_name} uploaded to {ENV_BUCKET}")     

      
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

    # downloading all info for the environment as a pkl file
    CLIENT.fget_object(ENV_BUCKET, env, file_path = f"{ROOT}/temp/environment/map.{env}.pkl")
    
    # reading the downloaded env pkl file
    with open(f"temp/environment/map.{env}.pkl", 'rb') as f:
        map_data = pickle.load(f)   
    
    # extracting map_metadata for the env i.e names of all the maps for the env
    maps_metadata_for_env = map_data.map_metadata

    # getting all the maps from maps bucket for the env
    for map_ in maps_metadata_for_env:
        CLIENT.fget_object(MAP_BUCKET, map_, file_path = f"temp/maps/map.{env}.pkl")
        with open(f"temp/maps/map.{env}.pkl", 'rb') as f:
            maps[map_] = pickle.load(f)  # storing the MAP objects in a dictionary
    
    return maps  # returning the maps as a dictionary where the key is the map name
    
main() 

    

    
    



    

   

     





    

