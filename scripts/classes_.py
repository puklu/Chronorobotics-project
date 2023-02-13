from copyreg import pickle
import pickle
from constants import TO_UPLOAD_PATH


class Environment:
    def __init__(self, name, gps_position) -> None:
        self.name = name
        self.gps_position = gps_position
        self.nodes = []
        self.nodes_names = []
        self.edges = []
        self.map_metadata = {'maps_names': [],
                             'images': [],
                             'trans': [],
                             'times': [],
                             'distances': [],
                             'distance': [],
                             'start_node': [],
                             'end_node': [],
                             'timestamp': []
                             }

    def pickle_env(self):
        with open(str(TO_UPLOAD_PATH) + "/" + "pickled_env.pkl", 'wb') as f:
            pickle.dump(self, f)

    def read_pickled_env(self, file_path):
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

    def pickle_map(self):
        with open(str(TO_UPLOAD_PATH) + "/" + "pickled_map.pkl", 'wb') as f:
            pickle.dump(self, f)

    def read_pickled_map(self, file_path):
        with open(file_path, 'rb') as f:
            map_data = pickle.load(f)
        return map_data

    def read(self):
        pickled_object = pickle.dumps(self)
        return pickled_object


class Node:
    """
    A class for creating the nodes of the graph
    """
    def __init__(self, key):
        self.key = key
        self.g_cost = None    # distance cost
        self.h_cost = 0       # heuristic cost
        self.f_cost = 0       # distance cost + heuristic cost
        self.via_node = None  # to be used while finding the shortest path (arrived to this node via which node)
        self.via_map = None   # to be used while finding the shortest path (arrived to this node via which path/map)
        self.isVisited = False
        self.neighbours = []


