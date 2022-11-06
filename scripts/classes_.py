from copyreg import pickle
import pickle
from pathlib import Path
from constants import ROOT, OBJECTS_PATH, DOWNLOADED_ENVS_PATH, DOWNLOADED_MAPS_PATH, TO_UPLOAD_PATH

# ROOT = Path(__file__).absolute().parent.parent
# OBJECTS_PATH = ROOT / "objects/"
# DOWNLOADED_MAPS_PATH = ROOT / 'downloaded_objects/maps/'
# DOWNLOADED_ENVS_PATH = ROOT / 'downloaded_objects/envs/'
# TO_UPLOAD_PATH = ROOT / 'pickled_objects/'


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
        with open(TO_UPLOAD_PATH+ "pickled_map.pkl", 'wb') as f:
            pickle.dump(self, f)
        

    def read_pickled_map(self,file_path):
        with open(file_path, 'rb') as f:
            map_data = pickle.load(f)   
        return map_data

    def read(self):   
        pickled_object = pickle.dumps(self)
        return pickled_object

