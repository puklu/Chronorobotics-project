from constants import CLIENT, MAP_BUCKET, ENV_BUCKET
from upload_utils import env_upload, map_upload
from fetch_utils import fetch_environment
# from graph_creation import create_graph
from meta_data_extraction import extract_map_metadata


def delete_a_map(env_name, map_name):
    """
    Deletes a map for an environment and updated the corresponding env object
    Args:
        env_name: The name of the environment to which the map belongs
        map_name: Name of the map to be deleted
    """
    map_obj_name = f"{env_name}.{map_name}.zip"

    env_obj = fetch_environment(env_name)  # fetching the env details

    if env_obj:
        maps_in_env = env_obj.map_metadata['maps_names'].copy()  # maps list in env before updating
        if map_name in maps_in_env:

            CLIENT.remove_object(MAP_BUCKET, map_obj_name)  # MAP DELETED from the db

            idx = maps_in_env.index(map_name)  # index of the deleted map in map_metadata

            # deleting all the data at the index corresponding to the deleted map
            del env_obj.map_metadata['maps_names'][idx]
            del env_obj.map_metadata['distance'][idx]
            # del env_obj.map_metadata['distances'][idx]
            del env_obj.map_metadata['start_node'][idx]
            del env_obj.map_metadata['end_node'][idx]
            del env_obj.map_metadata['times'][idx]
            del env_obj.map_metadata['timestamp'][idx]

            # # updating the nodes of the environment
            # env_obj.nodes = []
            # for snode in env_obj.map_metadata['start_node']:
            #     if snode not in env_obj.nodes:
            #         env_obj.nodes.append(snode)
            # for enode in env_obj.map_metadata['end_node']:
            #     if enode not in env_obj.nodes:
            #         env_obj.nodes.append(enode)

            # updating the neighbours of the environment
            env_obj.nodes = []  # all the nodes of the graph have to created again to update
            env_obj.nodes_names = []
            
            remaining_maps_in_env = env_obj.map_metadata['maps_names'].copy()  # NEW maps list in env after updating
            remaining_start_nodes = env_obj.map_metadata['start_node'].copy()
            remaining_end_nodes = env_obj.map_metadata['end_node'].copy()
            # remaining_env_name = env_obj.name
            num_of_maps = len(remaining_maps_in_env)

            # CLIENT.remove_object(ENV_BUCKET, env_name)  # env object deleted

            for k in range(num_of_maps):
                map_upload(env_name=env_name,
                           map_name=remaining_maps_in_env[k],
                           start_node=remaining_start_nodes[k],
                           end_node=remaining_end_nodes[k])
                # s_node_name = env_obj.map_metadata['start_node'][k].key
                # e_node_name = env_obj.map_metadata['end_node'][k].key
                # dist = env_obj.map_metadata['distance'][k]
                # m_name = env_obj.map_metadata['maps_names'][k]
                # # # cost =
                # env_obj.nodes_names.append(s_node_name)
                # env_obj.nodes_names.append(e_node_name)
                #
                # env_obj = extract_map_metadata(env_obj, map_name=m_name, start_node_name=s_node_name, end_node_name=e_node_name)
                #
                # node1, node2 = create_graph(env_obj, start_node_name=s_node_name, end_node_name=e_node_name, map_name=m_name, distance=dist)
                #
                # env_obj.nodes.append(node1)
                # env_obj.nodes.append(node2)


            # if len(env_obj.nodes) > 0:  # if there are still nodes in the environment
            #     env_upload(env_data=env_obj)  # uploading the updated env object
            # else:  # if there are no nodes left after deleting the map, then delete the env_obj
            #     CLIENT.remove_object(ENV_BUCKET, env_name)  # env object deleted

            print(f"Map: {map_name} deleted and environment: {env_name} updated")

        else:
            print(f"map: {map_name} doesn't exist in db for the env: {env_name}")

    else:
        print(f"Environment: {env_name} doesn't exist in db")


def delete_all_maps_of_an_environment(env_name):
    """
    Deletes all the maps from an environment followed by the deletion of the env object
    Args:
        env_name: Name of the environment for which all maps need to be deleted

    """
    env_obj = fetch_environment(env_name)  # fetching the env details

    if env_obj:
        list_of_maps_in_the_env = env_obj.map_metadata['maps_names']

        # deleting all the maps
        for map_ in list_of_maps_in_the_env:
            map_obj_name = f"{env_name}.{map_}.zip"
            try:
                CLIENT.remove_object(MAP_BUCKET, map_obj_name)  # map deleted
                print(f"map: {map_} deleted from env: {env_name}")
            except:
                print(f"{map_} doesn't exist in the db, so nothing deleted ")

        # deleting the env_obj
        CLIENT.remove_object(ENV_BUCKET, env_name)  # env object deleted

    else:
        print(f"Environment: {env_name} doesn't exist in db")
