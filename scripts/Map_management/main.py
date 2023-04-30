import argparse
import numpy as np
import matplotlib.pyplot as plt

from upload_utils import  map_upload
from fetch_utils import save_env_details, fetch_maps, fetch_environment
from delete_utils import delete_a_map, delete_all_maps_of_an_environment
from find_path import get_path, print_path
from data_manipulation import manipulated_map_upload
from visualise import visualise_similarity_matrix, visualise_heatmap
from cost_calculation import image_similarity_matrix_update, time_cost_calc, final_cost_calc, \
    calculate_similarity_matrix_and_periodicities, calculate_timeseries, calculate_periodicities, calculate_timeseries2, strands_test, \
    calculate_periodicities2

TEST_DATA = {"path0_map0": [1628441543.1538866, "2021-08-08 18:52:23"],
             "path0_map1": [1628446944.9681673, "2021-08-08 20:22:24"],
             "path0_map2": [1628457620, "2021-08-08 23:20:20"],
             "path0_map3": [1628443562.5771434, "2021-08-08 19:26:02"],
             "path0_map4": [1628445730.1549253, "2021-08-08 20:02:10"],
             "path0_map5": [1628532555, "2021-08-09 20:09:15"],
             "path0_map6": [1628447864.6729546, "2021-08-08 20:37:44"],
             "path0_map7": [1628448852.1848671, "2021-08-08 20:54:12"],
             "path0_map8": [1628450613.950567, "2021-08-08 21:23:33"],
             "path0_map9": [1628537656, "2021-08-09 21:34:16"],
             "path0_map10": [1628543174, "2021-08-09 23:06:14"],
             "path0_map11": [1628629965, "2021-08-10 23:12:45"],
             "path0_map12": [1628625016, "2021-08-10 21:50:16"],
             "path0_map13": [1628715812, "2021-08-11 23:03:32"],
             "path0_map15": [1628698644, "2021-08-11 18:17:24"],
             "path0_map16": [1628612717, "2021-08-10 18:25:17"],
             "path0_map17": [1628527576, "2021-08-09 18:46:16"],
             "path0_map18": [1628533349, "2021-08-09 20:22:29"],
             "path0_map19": [1628534238, "2021-08-09 20:37:18"],
             "path0_map20": [1628429197, "2021-08-08 15:26:37"],
             "path0_map21": [1628516065, "2021-08-09 15:34:25"],
             "path0_map22": [1628603830, "2021-08-10 15:57:10"],
             "path0_map23": [1628532992, "2021-08-09 20:16:32"],
             "path0_map24": [1628620012, "2021-08-10 20:26:52"],
             # --------------------

             "path0_map25": [1628470800, "2021-08-09 03:00:00"],
             "path0_map26": [1628488800, "2021-08-09 08:00:00"],
             "path0_map27": [1628496000, "2021-08-09 10:00:00"],
             "path0_map28": [1628506800, "2021-08-09 13:00:00"],
             "path0_map29": [1628557200, "2021-08-10 03:00:00"],
             "path0_map30": [1628575200, "2021-08-10 08:00:00"],
             "path0_map31": [1628582400, "2021-08-10 10:00:00"],
             "path0_map32": [1628593200, "2021-08-10 13:00:00"],
             "path0_map33": [1628643600, "2021-08-11 03:00:00"],
             "path0_map34": [1628661600, "2021-08-11 08:00:00"],
             "path0_map35": [1628668800, "2021-08-11 10:00:00"],
             "path0_map36": [1628679600, "2021-08-11 13:00:00"],
             "path0_map37": [1628688600, "2021-08-11 15:30:00"],
             }

SIMILARITY_MATRIX_ROW_ZERO_TEST = [0.17513923996294217, 0.07643693015278027, 0.05838751048506356, 0.04879837915052157, 0.045896101617553285, 0.04129888598660418, 0.02312919889733218, 0.014024437536086789, 0.006171111523986735,  0.001342343245345,  0.043421523986735  ,0.063421523986735, 0.07324342424433, 0.08053933367438879, 0.05724999917157255, 0.039349340019764506, 0.017496833349878196, 0.01525514223015582, 0.013855063652514389, 0.013334498763412453, 0.006171111523986735, 0.001342343245345,  0.043421523986735  ,0.063421523986735, 0.07324342424433, 0.06565812262994783, 0.06790887708894487, 0.017706041330447266, 0.013391240306180087, 0.006171111523986735, 0.001342343245345,  0.043421523986735  ,0.063421523986735, 0.07324342424433, 0.06885703684008469, 0.015290145130495661, 0.0053933367438879]

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

        env_name = 'env0'

        # TODO: Hardcoded values below for testing -----------------------------------------------------------------------------
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

        # softmax_similarity_matrix = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        #                                       [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        #                                       [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        #                                       [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        #                                       [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        #                                       [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])

        # times, values = calculate_timeseries(softmax_similarity_matrix, maps_timestamps)
        # print(times/3600)
        # print(values)
        # calculate_periodicities(times, values)
        # visualise_heatmap(softmax_similarity_matrix, maps_timestamps, maps_timestamps, "title", "test")

        # ------------------------------------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------------------------------
        # Testing with a sinusoid ------------------------------------------------------------------------------------------
        # size = 1000
        # rate = 3600
        # time_period = 24*3600
        # softmax_similarity_matrix = np.zeros((size,size))
        # maps_timestamps = np.arange(1628427600,1628427600 + size*rate, rate)
        # t = np.arange(0, size*rate, rate)
        # first_row = 0.5*(np.cos((2*np.pi/(time_period))* t) + 1)
        # softmax_similarity_matrix[0] = first_row
        # print(len(maps_timestamps))
        # print(len(first_row))
        # plt.plot(t, first_row, marker='o')
        # plt.show()

        # times, values = calculate_timeseries_test(first_row, t, env_name)
        # amplitudes, omegas, time_periods, phis, fremen = calculate_periodicities(times, values, env_name)
        # current_time = 1681479000 # 2023-04-14 15:30:00
        # final_cost_calc(env_name, time_periods, amplitudes, phis=phis, current_time=current_time)

        # --------------------------------------------------------------------------------------------------------------
        # map_andTimestamp_andLocal = TEST_DATA
        #
        # map_andTimestamp_andLocal = dict(
        #     sorted(map_andTimestamp_andLocal.items(), key=lambda x: x[1][0]))  # sorting the dict by timestamps
        #
        # maps_timestamps = []
        #
        # for item_ in map_andTimestamp_andLocal.items():
        #     maps_timestamps.append(item_[1][0])
        #
        # times, values = calculate_timeseries_test(SIMILARITY_MATRIX_ROW_ZERO_TEST, maps_timestamps, env_name)
        # amplitudes, omegas, time_periods, phis, fremen = calculate_periodicities(times, values, env_name)
        # current_time = 1681479000 # 2023-04-14 15:30:00
        # current_time = 1681755300  # 2023-04-17 20:15:00
        # current_time =  1692296100 # 2023-08-17 20:15:00
        # current_time =  1692299700 # 2023-08-17 21:15:00
        # current_time = 1681779600
        # current_time= 1692284400
        # current_time = 1692248400
        # current_time = 1628412743  # 2021-08-08 10:52:23
        # final_cost_calc(env_name, time_periods, amplitudes, phis=phis, fremen=fremen, current_time=current_time)

        # ---------------------------------------------------------------------------------------------------------------
        _, softmax_similarity_matrix, amplitudes, omegas, time_periods, phis, fremen = calculate_similarity_matrix_and_periodicities(env_name)
        # time_periods = [24*3600, 7*24*3600]
        # amplitudes = [1, 1]
        # phis = [0, 0]
        # current_time = 1628412743  # 2021-08-08 10:52:23
        # current_time = 1681479000  # 2023-04-14 15:30:00
        # current_time = 1681480800  # 2023-04-14 16:00:00
        # current_time = 1681755300  # 2023-04-17 20:15:00
        current_time = 1681759800  # 2023-04-17 21:30:00
        # current_time = 1681779600
        # current_time = 1629464400
        final_cost_calc(env_name, time_periods, amplitudes, phis=phis, fremen="fremen", current_time=current_time)

        # save_env_details(env_name)

    # ----------------------------------------------------------------------------------------------------------------
    #     strands_test()
    # ----------------------------------------------------------------------------------

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

        # fetching the maps corresponding to shortest path
        if shortest_path_maps:
            for map_ in shortest_path_maps:
                fetch_maps(args.e, map_)
    # -----------------------------------------------------------------------------------------------------------

    else:
        raise Exception("Please try again, something is missing..")

    # ------------------------------------------------------------------------------------------------------------


main()
