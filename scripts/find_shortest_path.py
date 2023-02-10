
def find_shortest_path(node, starting_node, shortest_path):

    if node.key == starting_node.key:
        shortest_path.append(node)
        return

    find_shortest_path(node.via, starting_node, shortest_path)

    shortest_path.append(node)

    return shortest_path


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
    shortest_path = []

    #finding starting node and ending node
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

    find_shortest_path(ending_node, starting_node, shortest_path)

    return shortest_path


def get_shortest_path(env_obj, starting_node_name, end_node_name):
    """
    Calls djikistra algorithm
    Args:
        env_obj: env_ obj
        starting_node_name: Node from which search should start
        end_node_name: Node at which search should end
    Returns: A list of nodes, which is the shortest path.
    """
    return djikstra(env_obj.nodes, starting_node_name, end_node_name)


def print_shortest_path(shortest_path_list):
    """
    Prints the keys of the names of the shortest path
    Args:
        shortest_path_list: List of nodes of shortest path
    """
    print("The shortest path is: ")
    for node in shortest_path_list:
        print(node.key, end=' -> ')
    print()

