def find_shortest_path_nodes(node, starting_node, shortest_path):
    if node.key == starting_node.key:
        shortest_path.append(node)
        return

    find_shortest_path_nodes(node.via, starting_node, shortest_path)

    shortest_path.append(node)

    return shortest_path


def find_shortest_path_maps(env_obj, shortest_path_nodes):
    """
    Extracts the sequence of maps of shortest path between two nodes, given a sequence of nodes of shortest path
    between the two nodes is provided.
    Args:
        env_obj: env object
        shortest_path_nodes: sequence of nodes of shortest path between two nodes

    Returns: A sequence of maps of shortest path between two nodes
    """
    maps_in_env = env_obj.map_metadata['maps_names']
    start_nodes_in_env = env_obj.map_metadata['start_node']
    end_nodes_in_env = env_obj.map_metadata['end_node']
    shortest_path_maps = []

    for n_idx in range(len(shortest_path_nodes) - 1):
        for j in range(len(maps_in_env)):
            if start_nodes_in_env[j] == shortest_path_nodes[n_idx] and end_nodes_in_env[j] == shortest_path_nodes[n_idx + 1]:
                shortest_path_maps.append(maps_in_env[j])
                break
            elif end_nodes_in_env[j] == shortest_path_nodes[n_idx] and start_nodes_in_env[j] == shortest_path_nodes[
                n_idx + 1]:
                shortest_path_maps.append(maps_in_env[j])
                break

    return shortest_path_maps


def djikstra(nodes_list, starting_node_name, ending_node_name):
    """
    Djikistra algorithm to find the shortest path between two nodes in the graph
    Args:
        nodes_list: A list of all the nodes in the graph
        starting_node_name: Node from which search should start
        ending_node_name: Node at which search should end

    Returns: A list of nodes, which is the shortest path.

    """
    starting_node_name = starting_node_name
    ending_node_name = ending_node_name
    priority_q = nodes_list
    shortest_path_nodes = []

    # finding starting node and ending node
    for node in nodes_list:
        if node.key == starting_node_name:
            starting_node = node
            node.weight = 0
            node.via = starting_node

        if node.key == ending_node_name:
            ending_node = node

    priority_q.sort(key=lambda x: x.weight, reverse=False)

    for node in priority_q:
        if not node.isVisited:
            for neighbour in node.neighbours:
                cost = node.weight + neighbour[1]
                if neighbour[0].weight >= cost:
                    neighbour[0].via = node
                    neighbour[0].weight = cost
        node.isVisited = True
        priority_q.sort(key=lambda x: x.weight, reverse=False)

    find_shortest_path_nodes(ending_node, starting_node, shortest_path_nodes)

    return shortest_path_nodes


def get_shortest_path(env_obj, starting_node_name, end_node_name):
    """
    Calls djikistra algorithm
    Args:
        env_obj: env_ obj
        starting_node_name: Node from which search should start
        end_node_name: Node at which search should end
    Returns: A list of nodes, which is the shortest path.
    """
    shortest_path_nodes = djikstra(env_obj.nodes, starting_node_name, end_node_name)

    shortest_path_maps = find_shortest_path_maps(env_obj, shortest_path_nodes)

    return shortest_path_nodes, shortest_path_maps


def print_shortest_path(shortest_path_nodes_list, shortest_path_maps_list):
    """
    Prints the keys of the names of the shortest path and the sequence of maps for the shortest path
    Args:
        shortest_path_maps_list: the sequence of maps for the shortest path
        shortest_path_nodes_list: List of nodes of shortest path
    """
    print("The nodes of shortest path is: ")
    for node in shortest_path_nodes_list:
        print(node.key, end=' -> ')
    print()

    print("The sequence of maps of shortest path is: ")
    for map_name in shortest_path_maps_list:
        print(map_name, end=' -> ')
    print()
