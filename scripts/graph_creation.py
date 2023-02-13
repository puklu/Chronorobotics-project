from classes_ import Node


def create_graph(env_obj, start_node_name, end_node_name, map_name, distance, cost):
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

    start_node.neighbours.append((end_node, map_name, distance, cost))
    end_node.neighbours.append((start_node, map_name, distance, cost))

    # setting the weights to a big number for shortest path finding algorithm
    start_node.g_cost = 10000000
    end_node.g_cost = 10000000
    start_node.h_cost = 10000000
    end_node.h_cost = 10000000

    return start_node, end_node
