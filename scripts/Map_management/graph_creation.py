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


def create_graph(env_obj, start_node_name, end_node_name, map_name, distance):
    """
    Creates instances of Node class and adds the neighbour and path weight information for the nodes
    Args:
        env_obj: the environment object
        start_node_name: Name of the starting node
        end_node_name: Name of the ending node
        map_name: name of the map connecting the two nodes
        distance: Distance between the two nodes
        cost: To keep track of cost until the node

    Returns: Node instances
    """
    if start_node_name not in env_obj.nodes_names:  # if the node already doesn't exist for the env
        start_node = Node(start_node_name)

    elif start_node_name in env_obj.nodes_names:    # if the node exists for the env, make that the start node
        s_idx = env_obj.nodes_names.index(start_node_name)
        start_node = env_obj.nodes[s_idx]

    if end_node_name not in env_obj.nodes_names:    # if the node already doesn't exist for the env
        end_node = Node(end_node_name)

    elif end_node_name in env_obj.nodes_names:      # if the node exists for the env, make that the end node
        e_idx = env_obj.nodes_names.index(end_node_name)
        end_node = env_obj.nodes[e_idx]

    # add the neighbour data
    start_node.neighbours.append([end_node, map_name, distance])
    end_node.neighbours.append([start_node, map_name, distance])

    # setting the weights to a big number for shortest path finding algorithm
    start_node.g_cost = 10000000
    end_node.g_cost = 10000000

    return start_node, end_node
