import os
import numpy as np
from constants import ENV_BUCKET, MAP_BUCKET, CLIENT, TO_UPLOAD_PATH
from classes_ import Features


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

def create_buckets(): 
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
        print(f"Bucket {ENV_BUCKET } already exists")    

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
