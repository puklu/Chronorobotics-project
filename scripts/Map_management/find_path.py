from cost_calculation import final_cost_calc


def find_shortest_path(node, starting_node, shortest_path_nodes, shortest_path_maps):
    """
    To traceback the path from end node to starting node using via
    Args:
        node:
        starting_node:
        shortest_path_nodes:
        shortest_path_maps:

    Returns:

    """
    if node is None:
        return

    if node.key == starting_node.key:
        shortest_path_nodes.append(node)
        return

    find_shortest_path(node.via_node, starting_node, shortest_path_nodes, shortest_path_maps)

    if node and node.via_map:
        shortest_path_nodes.append(node)
        shortest_path_maps.append(node.via_map)

    return shortest_path_nodes, shortest_path_maps


def dijkstra(nodes_list, starting_node_name, ending_node_name):
    """
    Dijkstra algorithm to find the shortest path between two nodes in the graph
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
            node.g_cost = 0
            node.via_node = starting_node
            both_nodes_found_flagger += 1
            if both_nodes_found_flagger == 2:
                break

        if node.key == ending_node_name:
            ending_node = node
            both_nodes_found_flagger += 1
            if both_nodes_found_flagger == 2:
                break

    priority_q.sort(key=lambda x: x.g_cost, reverse=False)

    for i in range(len(priority_q)):
        node = priority_q[i]
        if not node.isVisited:
            for neighbour in node.neighbours:
                if not neighbour[0].isVisited:
                    weight = node.g_cost + neighbour[2]
                    if neighbour[0].g_cost >= weight:
                        neighbour[0].via_node = node
                        neighbour[0].via_map = neighbour[1]
                        neighbour[0].g_cost = weight
        node.isVisited = True

        priority_q_to_sort = priority_q[i + 1:]
        priority_q_to_sort.sort(key=lambda x: x.g_cost, reverse=False)
        priority_q = priority_q[0:i + 1] + priority_q_to_sort

    find_shortest_path(ending_node, starting_node, shortest_path_nodes, shortest_path_maps)

    return shortest_path_nodes, shortest_path_maps


def a_star(nodes_list, starting_node_name, ending_node_name, final_cost):
    """
    Finds the shortest path between two nodes using A-star algorithm
    Args:
        nodes_list: A list of all the nodes ( instances of Node) in the graph
        starting_node_name: The name of the starting node
        ending_node_name: The name of the ending node
        final_cost: a dictionary containing the heuristic cost of all the maps

    Returns:

    """
    starting_node_name = starting_node_name
    ending_node_name = ending_node_name
    shortest_path_nodes = []
    shortest_path_maps = []
    open_nodes = []
    closed_nodes = []

    # finding starting node and ending node
    both_nodes_found_flagger = 0
    for node in nodes_list:
        if node.key == starting_node_name:
            starting_node = node
            node.g_cost = 0
            node.h_cost = 0
            node.via_node = starting_node
            both_nodes_found_flagger += 1
            if both_nodes_found_flagger == 2:
                break

        if node.key == ending_node_name:
            ending_node = node
            both_nodes_found_flagger += 1
            if both_nodes_found_flagger == 2:
                break

    open_nodes.append(starting_node)

    while len(open_nodes) > 0:
        current_node = open_nodes[0]
        del open_nodes[0]
        closed_nodes.append(current_node)

        if current_node.key == ending_node.key:
            break

        for neighbour in current_node.neighbours:
            if neighbour[0] not in closed_nodes:
                # weight = current_node.g_cost  # + neighbour[2]
                # cost = weight + neighbour[3]

                # # final_cost = {map_name: [cost, distance, map_timestamp, map_timestamp_local, final_cost]}
                cost = final_cost[neighbour[1]][-1]     # neighbour[1] <- map_name
                weight = current_node.g_cost + cost

                if neighbour[0] in open_nodes and weight < neighbour[0].f_cost:
                    idx = open_nodes.index(neighbour[0])
                    del open_nodes[idx]
                if neighbour[0] in closed_nodes and weight < neighbour[0].f_cost:
                    idx = closed_nodes.index(neighbour[0])
                    del closed_nodes[idx]

                if neighbour[0] not in closed_nodes and neighbour[0] not in open_nodes:
                    neighbour[0].via_node = current_node
                    neighbour[0].via_map = neighbour[1]
                    neighbour[0].g_cost = weight
                    neighbour[0].h_cost = cost
                    neighbour[0].f_cost = weight
                    if neighbour[0] not in open_nodes:
                        open_nodes.append(neighbour[0])

                    open_nodes.sort(key=lambda x: x.f_cost, reverse=False)

    find_shortest_path(ending_node, starting_node, shortest_path_nodes, shortest_path_maps)

    return shortest_path_nodes, shortest_path_maps


def get_path(env_obj, starting_node_name, end_node_name):
    """
    Calls A-star or dijkstra algorithm (based on the USE_A_STAR flag)
    Args:
        env_obj: env_ obj
        starting_node_name: Node from which search should start
        end_node_name: Node at which search should end
    Returns: A list of nodes, which is the shortest path.
    """
    USE_A_STAR = True  # Set to False if Dijkstra is to be used

    # if the starting or ending node do not exist in the graph
    if starting_node_name not in env_obj.nodes_names or end_node_name not in env_obj.nodes_names:
        if starting_node_name not in env_obj.nodes_names and end_node_name not in env_obj.nodes_names:
            print(f"The nodes {starting_node_name} and {end_node_name} do not exist!")
            return None, None
        elif starting_node_name not in env_obj.nodes_names:
            print(f"The node {starting_node_name} does not exist!")
            return None, None
        elif end_node_name not in env_obj.nodes_names:
            print(f"The node {end_node_name} does not exist!")
            return None, None

    # calculate the cost
    env_name = env_obj.name
    periodicities = env_obj.fremen_output['time_periods']
    amplitudes = env_obj.fremen_output['amplitudes']

    # Hardcoded values for testing in case needed
    # periodicities = [31831.57894737,      13440.,               24192.,               23261.53846154,        30240.]
    # amplitudes = [0.038102326504664503, 0.03572674840148863,  0.03943332208346288, 0.038365230317346496,  0.037481548730897735]

    final_cost = final_cost_calc(env_name, periodicities=periodicities, amplitudes=amplitudes)

    if USE_A_STAR:
        shortest_path_nodes, shortest_path_maps = a_star(env_obj.nodes, starting_node_name, end_node_name, final_cost)
    else:
        shortest_path_nodes, shortest_path_maps = dijkstra(env_obj.nodes, starting_node_name, end_node_name, final_cost)

    return shortest_path_nodes, shortest_path_maps


def print_path(shortest_path_nodes_list, shortest_path_maps_list):
    """
    Prints the keys of the names of the shortest path and the sequence of maps for the shortest path
    Args:
        shortest_path_maps_list: the sequence of maps for the shortest path
        shortest_path_nodes_list: List of nodes of shortest path
    """
    if not shortest_path_nodes_list or not shortest_path_maps_list:
        print("No path found!")
        return

    print("The nodes of shortest path is: ")
    for node in shortest_path_nodes_list:
        print(node.key, end=' -> ')
    print()

    print("The sequence of maps of shortest path is: ")
    for map_name in shortest_path_maps_list:
        print(map_name, end=' -> ')
    print()
