"""
This is to manually manipulate the data in db.
Can come in handy for testing
"""
from datetime import datetime

from classes_ import Node
from constants import DOT_ROS_PATH
from load_map import load_map


def extract_map_metadata_manipulated(env_obj, map_name, start_node_name, end_node_name, DISTANCE, path=None, TIMESTAMP=None):
    """
    Extracts the meta_data of an environment when a map is uploaded for an environment.
    Also updates the necessary env variables.
    Args:
        env_obj: Instance of Environment class
        map_name: Name of the map being uploaded
        start_node_name: Name of the starting node of the map
        end_node_name: Name of the ending node of the map
        DISTANCE: MANIPULATED DISTANCE

    Returns: An updated env object

    """
    if path is None:
        images, distances, trans, times = load_map(mappaths=f"{str(DOT_ROS_PATH)}/{map_name}")
    else:
        images, distances, trans, times = load_map(mappaths=f"{path}/{env_obj.name}/{map_name}")


    # distance = distances[0][-1]

    env_obj.map_metadata['maps_names'].append(map_name)
    # env_obj.map_metadata['images'].append(images)
    # env_obj.map_metadata['trans'].append(trans)
    env_obj.map_metadata['times'].append(times)
    # env_obj.map_metadata['distances'].append(distances)
    env_obj.map_metadata['distance'].append(DISTANCE)  # the length of the path

    '''
    Calculating timestamp for the map
    The middle point of times is assumed to be the timestamp for the map
    '''
    if TIMESTAMP is None:
        length_of_times = len(env_obj.map_metadata['times'][0][0])
        average_timestamp = int(env_obj.map_metadata['times'][0][0][int(length_of_times / 2)].to_time())
        timestamp = datetime.utcfromtimestamp(average_timestamp).strftime('%Y-%m-%d %H:%M:%S')
        env_obj.map_metadata['timestamp'].append(timestamp)
    else:
        env_obj.map_metadata['timestamp'].append(TIMESTAMP)

    # creating Nodes
    start_node, end_node = create_nodes(env_obj, start_node_name, end_node_name, DISTANCE)

    env_obj.map_metadata['start_node'].append(start_node)
    env_obj.map_metadata['end_node'].append(end_node)

    # adding the nodes to environment object
    if start_node_name not in env_obj.nodes_names:
        env_obj.nodes_names.append(start_node_name)
        env_obj.nodes.append(start_node)
    if end_node_name not in env_obj.nodes_names:
        env_obj.nodes_names.append(end_node_name)
        env_obj.nodes.append(end_node)

    return env_obj


def create_nodes(env_obj, start_node_name, end_node_name, distance):
    """
    Creates instances of Node class and adds the neighbour and path weight information for the nodes
    Args:
        env_obj: the environment object
        start_node_name: Name of the starting node
        end_node_name: Name of the ending node
        distance: Distance between the two nodes

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

    # setting the weights to a big number for shortest path finding algorithm
    start_node.weight = 10000
    end_node.weight = 10000

    start_node.neighbours.append((end_node, distance))
    end_node.neighbours.append((start_node, distance))

    return start_node, end_node

