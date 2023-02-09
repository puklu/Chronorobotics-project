from utils import *
from fetch_utils import *


def delete_a_map(env_name, map_name):
    """
    Deletes a map for an environment and updated the corresponding env object
    Args:
        env_name: The name of the environment to which the map belongs
        map_name: Name of the map to be deleted
    """
    map_obj_name = f"{env_name}.{map_name}.zip"

    CLIENT.remove_object(MAP_BUCKET, map_obj_name)  # map deleted

    env_obj = fetch_environment(env_name)  # fetching the env details

    if env_obj:
        maps_in_env = env_obj.map_metadata['maps_names']  # maps list in env before updating (map deleted at this point)
        if map_name in maps_in_env:
            idx = maps_in_env.index(map_name)  # index of the deleted map in map_metadata

            # deleting all the data at the index corresponding to the deleted map
            del env_obj.map_metadata['maps_names'][idx]
            del env_obj.map_metadata['distance'][idx]
            del env_obj.map_metadata['start_node'][idx]
            del env_obj.map_metadata['end_node'][idx]

            # updating the nodes of the environment
            env_obj.nodes = []
            for snode in env_obj.map_metadata['start_node']:
                if snode not in env_obj.nodes:
                    env_obj.nodes.append(snode)
            for enode in env_obj.map_metadata['end_node']:
                if enode not in env_obj.nodes:
                    env_obj.nodes.append(enode)

            if len(env_obj.nodes) > 0:  # if there are still nodes in the environment
                env_upload(env_data=env_obj)  # uploading the updated env object
            else:  # if there are no nodes left after deleting the map, then delete the env_obj
                CLIENT.remove_object(ENV_BUCKET, env_name)  # env object deleted

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
