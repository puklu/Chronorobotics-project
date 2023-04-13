import argparse
import numpy as np
import matplotlib.pyplot as plt

from upload_utils import batch_upload, map_upload
from fetch_utils import save_env_details, fetch_maps, fetch_environment, fetch_maps_by_time_cost
from delete_utils import delete_a_map, delete_all_maps_of_an_environment
from find_path import get_path, print_path
from data_manipulation import manipulated_map_upload
from visualise import visualise_similarity_matrix, visualise_heatmap
from cost_calculation import image_similarity_matrix_update, time_cost_calc, final_cost_calc, \
    calculate_similarity_matrix_and_periodicities, calculate_timeseries, calculate_periodicities


def main():
    parser = argparse.ArgumentParser(description='Tool for uploading and downloading maps created by VT&R')
    parser.add_argument('-e', help="Name of the environment for which map needs to be fetched")
    parser.add_argument('-m', help="Name of the map that needs to be fetched")
    parser.add_argument('-oe', help="Name of the environment for which only env needs to be fetched")
    parser.add_argument('-u', help="Name of the map to upload")
    parser.add_argument('-snode', help="Name of the start node")
    parser.add_argument('-enode', help="Name of the end node")
    parser.add_argument('-delamap', help="Name of the map to be deleted from a environment")
    parser.add_argument('-delmaps', help="Name of the env from which all maps are to be deleted")
    parser.add_argument('-findpath', help="Node names for between which path is desired", nargs='+')
    parser.add_argument('-mani', help="To manipulate distance and cost between two nodes when uploading maps",
                        nargs='+')

    args = parser.parse_args()
    # args = vars(parser.parse_args())

    # when no arguments are provided, buckets are created and all the maps for all the envs are uploaded -------------
    # if not any(args.values()):
    if not args.e \
            and not args.m \
            and not args.u \
            and not args.oe \
            and not args.snode \
            and not args.enode \
            and not args.delamap \
            and not args.delmaps \
            and not args.findpath \
            and not args.mani:

        # batch_upload()  # upload to db # TODO: SHOULD BE UNCOMMENTED AFTER TESTING IS DONE. THE FOLLOWING LINES SHOULD BE REMOVED.
        env_name = 'env0'

            # TODO: Hardcoded values below for testing
        # maps_timestamps = [1680933600, 1680948000, 1680969600, 1680980400, 1681020000, 1681034400, 1681056000, 1681066800, 1681106400, 1681120800, 1681142400, 1681153200, 1681192800, 1681207200, 1681228800, 1681239600, 1681279200, 1681293600]
        #                   # 0800,      1200,       1800,       2100,       0800,       1200,        1800,       2100,       0800,      1200,       1800,        2100,       0800,       1200,      1800,        2100,       0800,       1200
        # softmax_similarity_matrix = np.array([[0.80, 0.50, 0.10, 0.02, 0.80, 0.50, 0.10, 0.02, 0.80, 0.50, 0.10, 0.02, 0.80, 0.50, 0.10, 0.02, 0.80, 0.50],
        #                                       [0.50, 0.80, 0.10, 0.01, 0.50, 0.80, 0.10, 0.01, 0.50, 0.80, 0.10, 0.01, 0.50, 0.80, 0.10, 0.01, 0.50, 0.80],
        #                                       [0.10, 0.10, 0.80, 0.50, 0.10, 0.10, 0.80, 0.50, 0.10, 0.10, 0.80, 0.50, 0.10, 0.10, 0.80, 0.50, 0.10, 0.10],
        #                                       [0.02, 0.01, 0.50, 0.80, 0.02, 0.01, 0.50, 0.80, 0.02, 0.01, 0.50, 0.80, 0.02, 0.01, 0.50, 0.80, 0.02, 0.01],
        #                                       [0.80, 0.50, 0.10, 0.02, 0.80, 0.50, 0.10, 0.02, 0.80, 0.50, 0.10, 0.02, 0.80, 0.50, 0.10, 0.02, 0.80, 0.50],
        #                                       [0.50, 0.80, 0.10, 0.01, 0.50, 0.80, 0.10, 0.01, 0.50, 0.80, 0.10, 0.01, 0.50, 0.80, 0.10, 0.01, 0.50, 0.80],
        #                                       [0.10, 0.10, 0.80, 0.50, 0.10, 0.10, 0.80, 0.50, 0.10, 0.10, 0.80, 0.50, 0.10, 0.10, 0.80, 0.50, 0.10, 0.10],
        #                                       [0.02, 0.01, 0.50, 0.80, 0.02, 0.01, 0.50, 0.80, 0.02, 0.01, 0.50, 0.80, 0.02, 0.01, 0.50, 0.80, 0.02, 0.01],
        #                                       [0.80, 0.50, 0.02, 0.02, 0.80, 0.50, 0.10, 0.02, 0.80, 0.50, 0.02, 0.02, 0.80, 0.50, 0.02, 0.02, 0.80, 0.50],
        #                                       [0.50, 0.80, 0.10, 0.01, 0.50, 0.80, 0.10, 0.01, 0.50, 0.80, 0.10, 0.01, 0.50, 0.80, 0.10, 0.01, 0.50, 0.80],
        #                                       [0.10, 0.10, 0.80, 0.50, 0.10, 0.10, 0.80, 0.50, 0.02, 0.10, 0.80, 0.50, 0.10, 0.10, 0.80, 0.50, 0.02, 0.10],
        #                                       [0.02, 0.01, 0.50, 0.80, 0.02, 0.01, 0.50, 0.80, 0.02, 0.01, 0.50, 0.80, 0.02, 0.01, 0.50, 0.80, 0.02, 0.01],
        #                                       [0.80, 0.50, 0.10, 0.02, 0.80, 0.50, 0.10, 0.02, 0.80, 0.50, 0.10, 0.02, 0.80, 0.50, 0.10, 0.02, 0.80, 0.50],
        #                                       [0.50, 0.80, 0.10, 0.01, 0.50, 0.80, 0.10, 0.01, 0.50, 0.80, 0.10, 0.01, 0.50, 0.80, 0.10, 0.01, 0.50, 0.80],
        #                                       [0.10, 0.10, 0.80, 0.50, 0.10, 0.10, 0.80, 0.50, 0.02, 0.10, 0.80, 0.50, 0.10, 0.10, 0.80, 0.50, 0.10, 0.10],
        #                                       [0.02, 0.01, 0.50, 0.80, 0.02, 0.01, 0.50, 0.80, 0.02, 0.01, 0.50, 0.80, 0.02, 0.01, 0.50, 0.80, 0.02, 0.01],
        #                                       [0.80, 0.50, 0.10, 0.02, 0.80, 0.50, 0.10, 0.02, 0.80, 0.50, 0.02, 0.02, 0.80, 0.50, 0.10, 0.02, 0.80, 0.50],
        #                                       [0.50, 0.80, 0.10, 0.01, 0.50, 0.80, 0.10, 0.01, 0.50, 0.80, 0.10, 0.01, 0.50, 0.80, 0.10, 0.01, 0.50, 0.80]])
        #


        # softmax_similarity_matrix = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        #                                       [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        #                                       [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        #                                       [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        #                                       [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        #                                       [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])

        # time_cost_calc(env_name, [3600], [1])  # [3600, 86400, 604800, 2592000, 31536000])
        _, softmax_similarity_matrix, amplitudes, omegas, time_periods, _ = calculate_similarity_matrix_and_periodicities(env_name)
        # time_periods = [3600, 3*3600]
        # amplitudes = [1, 0.5]
        # final_cost_calc(env_name, time_periods, amplitudes)

        # times, values = calculate_timeseries(softmax_similarity_matrix, maps_timestamps)
        # print(times/3600)
        # print(values)
        # calculate_periodicities(times, values)
        # visualise_heatmap(softmax_similarity_matrix, maps_timestamps, maps_timestamps, "title", "test")

        # visualise_similarity_matrix(env_name)
        # visualise_fft_for_env(env_name)
        # save_env_details(env_name)

    # ----------------------------------------------------------------------------------------------------------------

    # delete all maps from an environment ----------------------------------------------------------------------------
    elif args.delmaps \
            and not args.e \
            and not args.m \
            and not args.u \
            and not args.oe \
            and not args.snode \
            and not args.enode \
            and not args.delamap \
            and not args.findpath \
            and not args.mani:
        delete_all_maps_of_an_environment(args.delmaps)

    # -------------------------------------------------------------------------------------------------------------

    # deleting a map ----------------------------------------------------------------------------------------------
    elif args.delamap \
            and args.e \
            and not args.m \
            and not args.u \
            and not args.oe \
            and not args.snode \
            and not args.enode \
            and not args.delmaps \
            and not args.findpath \
            and not args.mani:
        map_name = args.delamap
        env_name = args.e
        delete_a_map(env_name, map_name)
        save_env_details(env_name)
    # -------------------------------------------------------------------------------------------------------------

    # when only -oe is provided, only env object is fetched --------------------------------------------------------
    elif args.oe \
            and not args.e \
            and not args.m \
            and not args.u \
            and not args.snode \
            and not args.enode \
            and not args.delamap \
            and not args.delmaps \
            and not args.findpath \
            and not args.mani:

        save_env_details(args.oe)  # printing the metadata
    # --------------------------------------------------------------------------------------------------------------

    # when only -e is provided, all the maps for that particular environment are fetched ----------------------------
    elif args.e \
            and not args.m \
            and not args.u \
            and not args.oe \
            and not args.snode \
            and not args.enode \
            and not args.delamap \
            and not args.delmaps \
            and not args.findpath \
            and not args.mani:
        fetch_maps(args.e)
    # --------------------------------------------------------------------------------------------------------------

    # when -e and -m are provided, that particular map is fetched for that environment -----------------------------
    elif args.e \
            and args.m \
            and not args.u \
            and not args.oe \
            and not args.snode \
            and not args.enode \
            and not args.delamap \
            and not args.delmaps \
            and not args.findpath \
            and not args.mani:
        fetch_maps(args.e, args.m)
    # --------------------------------------------------------------------------------------------------------------

    # when -u and -e are provided, map u is uploaded for env e from .ros --------------------------------------------
    elif args.u \
            and args.e \
            and args.snode \
            and args.enode \
            and not args.m \
            and not args.oe \
            and not args.delamap \
            and not args.delmaps \
            and not args.findpath \
            and not args.mani:

        map_name = f"{args.u}"
        start_node = args.snode
        end_node = args.enode

        map_upload(env_name=args.e, map_name=map_name, start_node=start_node, end_node=end_node)
        save_env_details(args.e)
    # --------------------------------------------------------------------------------------------------------------

    # Find the path between two nodes ---------------------------------------------------------------------
    elif args.e \
            and not args.m \
            and not args.u \
            and not args.oe \
            and not args.snode \
            and not args.enode \
            and not args.delamap \
            and not args.delmaps \
            and args.findpath \
            and not args.mani:

        env_obj = fetch_environment(args.e)
        starting_node_name = args.findpath[0]
        last_node_name = args.findpath[1]

        shortest_path_nodes, shortest_path_maps = get_path(env_obj=env_obj,
                                                           starting_node_name=starting_node_name,
                                                           end_node_name=last_node_name)

        print_path(shortest_path_nodes, shortest_path_maps)

        # # fetching the maps corresponding to shortest path
        # if shortest_path_maps:
        #     for map_ in shortest_path_maps:
        #         fetch_maps(args.e, map_)
    # -----------------------------------------------------------------------------------------------------------

    # ONLY FOR TESTING. SHOULD NOT BE CALLED OTHERWISE!! ---------------------------------------------------------
    # when -u, -e, -mani are provided, map u is uploaded for env e from .ros with the manipulated length of the map
    elif args.u \
            and args.e \
            and args.snode \
            and args.enode \
            and not args.m \
            and not args.oe \
            and not args.delamap \
            and not args.delmaps \
            and not args.findpath \
            and args.mani:

        map_name = args.u
        env_name = args.e
        start_node = args.snode
        end_node = args.enode
        manipulated_distance = int(args.mani[0])  # manipulating distance
        manipulated_cost = int(args.mani[1])  # manipulating cost

        manipulated_map_upload(env_name=env_name,
                               map_name=map_name,
                               start_node=start_node,
                               end_node=end_node,
                               manipulated_distance=manipulated_distance,
                               manipulated_cost=manipulated_cost)

    else:
        raise Exception("Please try again, something is missing..")

    # ------------------------------------------------------------------------------------------------------------


main()
