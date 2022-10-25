from copyreg import pickle
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import uuid

from unicodedata import name
from minio import Minio
from minio.error import S3Error

import io

END_POINT = "localhost:9000"
ACCESS_KEY = "vivek"
SECRET_KEY = "vivek123"

OBJECTS_PATH = "objects/environment/"
ENV_BUCKET = "environment"
MAP_BUCKET = "maps"

DOWNLOADED_MAPS_PATH ='downloaded_objects/maps/'
DOWNLOADED_ENVS_PATH ='downloaded_objects/envs/'
TO_UPLOAD_PATH = 'to_upload/'

class Features():
    def __init__(self) -> None:
        self.shape = None
        self.values = None


class Environment:
    def __init__(self, name, gps_position, nodes, edges, map_metadata) -> None:
        self.name = name
        self.gps_position = gps_position
        self.nodes = nodes
        self.edges = edges
        self.map_metadata = map_metadata

    def pickle_env(self):
        with open("pickled_env.pkl", 'wb') as f:
            pickle.dump(self, f)
        

    def read_pickled_env(self,file_path):
        with open(file_path, 'rb') as f:
            env_data = pickle.load(f)   
        return env_data  

    # map metadata - a list containing object names of all maps for this env

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
        
        self.env_id = self.name+"."+self.start_node+"."+self.end_node
        
    def fetch_place(self, map_no, idx):
        return {"image": self.images[map_no][idx],
                "distance": self.distances[map_no][idx],
                "trans": self.trans[map_no][idx],
                "time":self.times[map_no][idx]}


    def pickle_map(self):
        with open("pickled_map.pkl", 'wb') as f:
            pickle.dump(self, f)
        

    def read_pickled_map(self,file_path):
        with open(file_path, 'rb') as f:
            map_data = pickle.load(f)   
        return map_data

    # def fetch_map(self):
    #     try:
    #         client.fget_object(bucket_name=MAP_BUCKET, object_name=obj_name, file_path='downloaded_objects/'+obj_name+'.pkl')
    #         print("Map already exists in db")


    def read(self):   
        pickled_object = pickle.dumps(self)
        return pickled_object



def load_map(mappaths, images, distances, trans, times):

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
                print("Loaded feature: " + dist + str(".npy"))
      
        tmp_times[-1] = tmp_times[-2]  # to avoid very long period before map end
        images.append(tmp_images)
        distances.append(tmp_distances)
        trans.append(tmp_trans)
        times.append(tmp_times)
        # rospy.logwarn("Whole map " + str(mappath) + " sucessfully loaded")
        print("Whole map " + str(mappath) + " sucessfully loaded")


# def map_upload(map_data):
#     # try:
#     #     client.stat_object(bucket_name=ENV_BUCKET, object_name='map')  # or stat_object can be used to check existence
#     # except:    
#     obj_name = map_data.env_id
#     client.put_object(bucket_name=MAP_BUCKET, object_name=obj_name,data=(map_data), length=5000000)#, part_size = 50000000)
#     print("uploaded")   


def map_upload(map_data):
    """
    Uploads a map object  to db
    Args:
        A object instance of class Map
    Returns:
        env_id of the object    
    """
    global client
    obj_name = map_data.env_id  # using the map object's env_id as object name in db

    # check if the map exists in db
    try:
        # client.stat_object(bucket_name=ENV_BUCKET, object_name=obj_name)
        client.fget_object(bucket_name=MAP_BUCKET, object_name=obj_name, file_path=DOWNLOADED_MAPS_PATH + obj_name+'.pkl')
        print("Map already exists in db. Downloading....")
    
    # if the map doesn't exist in db, then upload
    except:
        map_data.pickle_map()
        # pickle_map(map_data)
        # try:
        #     client.stat_object(bucket_name=MAP_BUCKET, object_name='map')  # or stat_object can be used to check existence
        # except:    
        
        client.fput_object(bucket_name=MAP_BUCKET, object_name=obj_name,file_path=TO_UPLOAD_PATH +'pickled_map.pkl', metadata={"env_id":map_data.env_id})
        print("Map uploaded to db")     

    return map_data.env_id


def env_upload(env_data):
    """
    Uploads an environment object  to db
    Args:
        A object instance of class Environment   
    """
    obj_name= env_data.map_metadata
    # check if the env variables exist for the map in db
    try:
        # client.stat_object(bucket_name=ENV_BUCKET, object_name=obj_name)
        client.fget_object(bucket_name=ENV_BUCKET, object_name=obj_name, file_path=DOWNLOADED_ENVS_PATH + obj_name+'.pkl')
        print("Env already exists in db. Downloading....")
    
    # if the map doesn't exist in db, then upload
    except:
        env_data.pickle_env()
        # pickle_map(map_data)
        # try:
        #     client.stat_object(bucket_name=ENV_BUCKET, object_name='map')  # or stat_object can be used to check existence
        # except:    
        client.fput_object(bucket_name=ENV_BUCKET, object_name=obj_name,file_path=TO_UPLOAD_PATH + 'pickled_env.pkl')
        print("uploaded env")     
    

def main():

    images = []
    distances = []
    trans = []
    times= []
    mappaths = 'objects/test_map/map0' #,objects/test_map/map1'
    
    GPS_POSITION = '142.43.1'

    # load the map
    load_map(mappaths=mappaths, images=images, distances=distances, trans=trans, times=times )
   
    # map object
    map0 = Map(name='kn', start_node='main_gate', end_node='library', images=images, distances=distances, trans=trans, times=times)

    # uploading the map object to db
    map_metadata = map_upload(map0) 

    # env object
    env0 = Environment(name=map0.name, gps_position=GPS_POSITION, nodes=None, edges=None, map_metadata=map_metadata)

    # uploading the environment obj to db
    env_upload(env0)






if __name__ == "__main__":

    client = Minio( END_POINT,
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            secure=False)

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
        

    main()
    
    



    

   

     





    

