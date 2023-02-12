def find_shortest_path(node, starting_node, shortest_path_nodes, shortest_path_maps):
    if node.key == starting_node.key:
        shortest_path_nodes.append(node)
        return

    find_shortest_path(node.via_node, starting_node, shortest_path_nodes, shortest_path_maps)

    shortest_path_nodes.append(node)
    shortest_path_maps.append(node.via_map)

    return shortest_path_nodes, shortest_path_maps


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
    priority_q = nodes_list.copy()
    shortest_path_nodes = []
    shortest_path_maps = []

    # finding starting node and ending node
    both_nodes_found_flagger = 0
    for node in nodes_list:
        if node.key == starting_node_name:
            starting_node = node
            node.weight = 0
            node.via_node = starting_node
            both_nodes_found_flagger += 1
            if both_nodes_found_flagger == 2:
                break

        if node.key == ending_node_name:
            ending_node = node
            both_nodes_found_flagger += 1
            if both_nodes_found_flagger == 2:
                break

    priority_q.sort(key=lambda x: x.weight, reverse=False)

    for i in range(len(priority_q)):
        node = priority_q[i]
        if not node.isVisited:
            for neighbour in node.neighbours:
                if not neighbour[0].isVisited:
                    weight = node.weight + neighbour[2]
                    if neighbour[0].weight >= weight:
                        neighbour[0].via_node = node
                        neighbour[0].via_map = neighbour[1]
                        neighbour[0].weight = weight
        node.isVisited = True

        priority_q_to_sort = priority_q[i + 1:]
        priority_q_to_sort.sort(key=lambda x: x.weight, reverse=False)
        priority_q = priority_q[0:i + 1] + priority_q_to_sort

    find_shortest_path(ending_node, starting_node, shortest_path_nodes, shortest_path_maps)

    return shortest_path_nodes, shortest_path_maps


def get_shortest_path(env_obj, starting_node_name, end_node_name):
    """
    Calls djikistra algorithm
    Args:
        env_obj: env_ obj
        starting_node_name: Node from which search should start
        end_node_name: Node at which search should end
    Returns: A list of nodes, which is the shortest path.
    """
    shortest_path_nodes, shortest_path_maps = djikstra(env_obj.nodes, starting_node_name, end_node_name)

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
