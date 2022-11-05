import sys
from copyreg import pickle
import pickle
import os
from pathlib import Path
import numpy as np
import uuid
import urllib3
import argparse

from unicodedata import name
from minio import Minio
from minio.error import S3Error

import io

from dotenv import load_dotenv


load_dotenv()
END_POINT = os.getenv("END_POINT")
ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

OBJECTS_PATH = "objects/environment/"
ENV_BUCKET = "environment"
MAP_BUCKET = "maps"

DOWNLOADED_MAPS_PATH ='downloaded_objects/maps/'
DOWNLOADED_ENVS_PATH ='downloaded_objects/envs/'
TO_UPLOAD_PATH = 'pickled_objects/'

client = Minio( END_POINT,
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            secure=False)

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-e', help="Name of the environment for which maps need to be fetched")
    args = parser.parse_args()

    create_buckets()    # TODO: to uncomment

    upload_objects()    # TODO: to uncomment

    if args.e:
        env_reached = args.e
        print(fetch_maps(env_reached))
    else:
        print("No environment provided")  
        env_reached = 'env0'   # TODO: just for testing, remove later
        # print(fetch_maps(env_reached))  

    

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

    def __repr__(self):
        return {"name": self.name,
                "gps_position": self.gps_position,
                "nodes": self.nodes,
                "edges":self.edges,
                "map_metadata": self.map_metadata}    

    def pickle_env(self):
        with open(TO_UPLOAD_PATH + "pickled_env.pkl", 'wb') as f:
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
        
    def _repr__(self):
        return {"env_id": self.env_id,
                "name" : self.name,
                "start_node": self.start_node,
                "end_node": self.end_node,
                "image": self.images,
                "distance": self.distances,
                "trans": self.trans,
                "time":self.times}

    def pickle_map(self):
        with open(TO_UPLOAD_PATH+ "pickled_map.pkl", 'wb') as f:
            pickle.dump(self, f)
        

    def read_pickled_map(self,file_path):
        with open(file_path, 'rb') as f:
            map_data = pickle.load(f)   
        return map_data

    def read(self):   
        pickled_object = pickle.dumps(self)
        return pickled_object



def load_map(mappaths):

    images = []
    distances = []
    trans = []
    times= []

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
        print("Whole map " + str(mappath) + " sucessfully loaded")

        return images, distances, trans, times


def map_upload(map_data):
    """
    Uploads a map object  to db
    Args:
        A object instance of class Map
    Returns:
        env_id of the object    
    """
    global client
    obj_name = map_data.name 

    # check if the map exists in db
    statobj = client.stat_object(MAP_BUCKET, obj_name, ssec=None, version_id=None, extra_query_params=None)
    
    if statobj:
        print("Object already exists in db")
    
    else:
        map_data.pickle_map()
        # TODO: To change it to put_object
        # client.put_object(bucket_name=MAP_BUCKET, object_name=obj_name,data=(map_data), length=5000000)#, part_size = 50000000)
        client.fput_object(bucket_name=MAP_BUCKET, object_name=obj_name,file_path=TO_UPLOAD_PATH +'pickled_map.pkl', metadata={"env_id":map_data.env_id})
        print("Map uploaded to db")     

    return map_data.env_id


def env_upload(env_data):
    """
    Uploads an environment object  to db
    Args:
        A object instance of class Environment   
    """
    global client
    obj_name= env_data.name

    # check if the env variables exist for the map in db
    statobj = client.stat_object(ENV_BUCKET, obj_name, ssec=None, version_id=None, extra_query_params=None)
    
    if statobj:
        print("Object already exists in db")
    
    else:
        env_data.pickle_env()
           
        client.fput_object(bucket_name=ENV_BUCKET, object_name=obj_name,file_path=TO_UPLOAD_PATH + 'pickled_env.pkl')
        print("uploaded env")     


def create_buckets(): 
    global client
    # create the buckets
    found = client.bucket_exists(MAP_BUCKET)
    if not found:
        client.make_bucket(MAP_BUCKET)
    else:
        print(f"Bucket {MAP_BUCKET} already exists")

    found = client.bucket_exists(ENV_BUCKET)
    if not found:
        client.make_bucket(ENV_BUCKET)
    else:
        print(f"Bucket {ENV_BUCKET }already exists")    
       

def upload_objects():

    objects_path = Path("objects/")
    maps_path = objects_path / "maps"

    number_of_environments = len(list(os.listdir(maps_path)))  # count of all the environments
    environments = list(os.listdir(maps_path))  # list of all the environments

    for i in range(number_of_environments):   # iterating over each envrionment
        env_obj = Environment(name=environments[i], gps_position=None, nodes=None, edges=None) # env object for the cirrent environment

        number_of_maps = len(list(os.listdir((maps_path / environments[i])))) # count of all the maps for the current environment

        map_paths_for_env = []

        for j in range(number_of_maps):
            # for loading the maps for the environment
            images = []
            distances = []
            trans = []
            times= []
               
            maps = list(os.listdir((maps_path / environments[i])))  
            map_paths_for_env.append('objects/maps/'+environments[i]+'/'+maps[j]) # paths of all the maps for the current environment
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
    global client
    try:
        response = client.get_object(ENV_BUCKET, env)
    finally:
        response.close()
        response.release_conn()  

    return response.getheaders()  # TODO: HOW TO READ THE DATA RETURNED BY get_object ???

 
main() 

    

    
    



    

   

     





    

