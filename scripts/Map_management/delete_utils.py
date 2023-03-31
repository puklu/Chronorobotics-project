import os
import logging

from constants import CLIENT, MAP_BUCKET, ENV_BUCKET, FIRST_IMAGE_BUCKET, IMAGES_PATH, RESULTS_PATH, LOGS_PATH
from upload_utils import env_upload, map_upload
from fetch_utils import fetch_environment, fetch_first_images
from cost_calculation import image_similarity_matrix_calc

logging.basicConfig(filename=f"{LOGS_PATH}/delete_utils.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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

            idx = maps_in_env.index(map_name)  # index of the deleted map in map_metadata
            start_node = env_obj.map_metadata['start_node'][idx]
            end_node = env_obj.map_metadata['end_node'][idx]

            # deleting all the data from map_metadata at the index corresponding to the deleted map
            del env_obj.map_metadata['maps_names'][idx]
            del env_obj.map_metadata['distance'][idx]
            # del env_obj.map_metadata['distances'][idx]
            del env_obj.map_metadata['start_node'][idx]
            del env_obj.map_metadata['end_node'][idx]
            del env_obj.map_metadata['times'][idx]
            del env_obj.map_metadata['timestamp'][idx]

            # updating the neighbours of the environment
            s_idx = env_obj.nodes.index(start_node)
            e_idx = env_obj.nodes.index(end_node)

            # index of end_node in the list of neighbours of start_node to update it
            neighbours_of_start_node = env_obj.nodes[s_idx].neighbours
            for m in range(len(neighbours_of_start_node)):
                if neighbours_of_start_node[m][1] == map_name:
                    idx_neigh_start_node = m
                    break
            del neighbours_of_start_node[idx_neigh_start_node]

            # index of start_node in the list of neighbours of end_node to update it
            neighbours_of_end_node = env_obj.nodes[e_idx].neighbours
            for n in range(len(neighbours_of_end_node)):
                if neighbours_of_end_node[n][1] == map_name:
                    idx_neigh_end_node = n
                    break
            del neighbours_of_end_node[idx_neigh_end_node]

            # if a node doesn't have any neighbours left, deleting the node
            if len(neighbours_of_start_node) == 0:
                del env_obj.nodes[s_idx]
            if len(neighbours_of_end_node) == 0:
                del env_obj.nodes[e_idx]

            # update env.nodes_names
            env_obj.nodes_names = []
            for node in env_obj.nodes:
                env_obj.nodes_names.append(node.key)

            CLIENT.remove_object(MAP_BUCKET, map_obj_name)  # MAP DELETED from the db

            CLIENT.remove_object(FIRST_IMAGE_BUCKET, f"{env_name}.{map_name}.jpg")  # FIRST IMAGE DELETED from the db

            # recalculating similarity matrix ---------------------------------------
            fetch_first_images(env_obj)

            maps_names = env_obj.map_metadata['maps_names']

            images_names = []
            for map_name in maps_names:
                images_names.append(f"{env_name}.{map_name}.jpg")

            recalculated_similarity_matrix, recalculated_softmax_similarity_matrix = image_similarity_matrix_calc(env_name, save_plot=False)

            env_obj.similarity_matrix = recalculated_similarity_matrix
            env_obj.softmax_similarity_matrix = recalculated_softmax_similarity_matrix

            # deleting all the downloaded images to save space
            for path, directories, files in os.walk(IMAGES_PATH):
                for file in files:
                    os.remove(f"{IMAGES_PATH}/{file}")

            # -----------------------------------------------------------------------

            # uploading the updated env object
            env_upload(env_data=env_obj)
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
            first_image_obj_name = f"{env_name}.{map_}.jpg"
            try:
                CLIENT.remove_object(MAP_BUCKET, map_obj_name)  # map deleted
                print(f"map: {map_} deleted from env: {env_name}")
                CLIENT.remove_object(FIRST_IMAGE_BUCKET, first_image_obj_name)  # first image deleted
                logging.info(f"first image: {first_image_obj_name} deleted from env: {env_name}")
                # print(f"first image: {first_image_obj_name} deleted from env: {env_name}")
            except:
                print(f"{map_} doesn't exist in the db, so nothing deleted ")

        # deleting the env_obj
        CLIENT.remove_object(ENV_BUCKET, env_name)  # env object deleted

    else:
        print(f"Environment: {env_name} doesn't exist in db")
