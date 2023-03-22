import argparse

from upload_utils import batch_upload, map_upload, create_buckets, env_upload, first_image_upload
from fetch_utils import print_env_details, fetch_maps, fetch_environment, fetch_maps_by_time_cost
from delete_utils import delete_a_map, delete_all_maps_of_an_environment
from find_shortest_path import get_shortest_path, print_shortest_path
from data_manipulation import manipulated_map_upload
from visualise import visualise_similarity_matrix, visualise_fft_for_env
from cost_calculation import image_similarity_matrix_update


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-e', help="Name of the environment for which map needs to be fetched")
    parser.add_argument('-m', help="Name of the map that needs to be fetched")
    parser.add_argument('-oe', help="Name of the environment for which only env needs to be fetched")
    parser.add_argument('-u', help="Name of the map to upload")
    parser.add_argument('-snode', help="Name of the start node")
    parser.add_argument('-enode', help="Name of the end node")
    parser.add_argument('-delamap', help="Name of the map to be deleted from a environment")
    parser.add_argument('-delmaps', help="Name of the env from which all maps are to be deleted")
    parser.add_argument('-shortest', help="Starting nodes for shortest path between two nodes", nargs='+')

    parser.add_argument('-mani', help="To manipulate distance and cost between two nodes when uploading maps",
                        nargs='+')

    args = parser.parse_args()

    # when no arguments are provided, buckets are created and all the maps for all the envs are uploaded
    if not args.e \
            and not args.m \
            and not args.u \
            and not args.oe \
            and not args.snode \
            and not args.enode \
            and not args.delamap \
            and not args.delmaps \
            and not args.shortest \
            and not args.mani:

        # batch_upload()  # upload to db # TODO: SHOULD BE UNCOMMENTED AFTER TESTING IS DONE. THE FOLLOWING LINES SHOULD BE REMOVED.
        # find_map_according_to_time([3600, 86400, 604800, 31536000])
        # fetch_maps_according_to_time('env2', [3600, 86400, 604800, 2592000, 31536000])
        # fetch_maps_by_time_cost('env0', [3600])
        # image_similarity_matrix_update('env', 'ddd')
        visualise_similarity_matrix('env0')
        # visualise_fft_for_env('env0')


    # delete all maps from an environment
    elif args.delmaps \
            and not args.e \
            and not args.m \
            and not args.u \
            and not args.oe \
            and not args.snode \
            and not args.enode \
            and not args.delamap \
            and not args.shortest \
            and not args.mani:
        delete_all_maps_of_an_environment(args.delmaps)

    # deleting a map
    elif args.delamap \
            and args.e \
            and not args.m \
            and not args.u \
            and not args.oe \
            and not args.snode \
            and not args.enode \
            and not args.delmaps \
            and not args.shortest \
            and not args.mani:
        map_name = args.delamap
        env_name = args.e
        delete_a_map(env_name, map_name)

    # when only -oe is provided, only env object is fetched
    elif args.oe \
            and not args.e \
            and not args.m \
            and not args.u \
            and not args.snode \
            and not args.enode \
            and not args.delamap \
            and not args.delmaps \
            and not args.shortest \
            and not args.mani:

        print_env_details(args.oe)  # printing the metadata

    # when only -e is provided, all the maps for that particular environment are fetched
    elif args.e \
            and not args.m \
            and not args.u \
            and not args.oe \
            and not args.snode \
            and not args.enode \
            and not args.delamap \
            and not args.delmaps \
            and not args.shortest \
            and not args.mani:
        fetch_maps(args.e)

    # when -e and -m are provided, that particular map is fetched for that environment
    elif args.e \
            and args.m \
            and not args.u \
            and not args.oe \
            and not args.snode \
            and not args.enode \
            and not args.delamap \
            and not args.delmaps \
            and not args.shortest \
            and not args.mani:
        fetch_maps(args.e, args.m)

    # when -u and -e are provided, map u is uploaded for env e from .ros
    elif args.u \
            and args.e \
            and args.snode \
            and args.enode \
            and not args.m \
            and not args.oe \
            and not args.delamap \
            and not args.delmaps \
            and not args.shortest \
            and not args.mani:

        map_name = f"{args.u}"
        start_node = args.snode
        end_node = args.enode

        map_upload(env_name=args.e, map_name=map_name, start_node=start_node, end_node=end_node)

    # Find the shortest path between two nodes
    elif args.e \
            and not args.m \
            and not args.u \
            and not args.oe \
            and not args.snode \
            and not args.enode \
            and not args.delamap \
            and not args.delmaps \
            and args.shortest \
            and not args.mani:

        env_obj = fetch_environment(args.e)
        starting_node_name = args.shortest[0]
        last_node_name = args.shortest[1]

        shortest_path_nodes, shortest_path_maps = get_shortest_path(env_obj=env_obj,
                                                                    starting_node_name=starting_node_name,
                                                                    end_node_name=last_node_name)

        print_shortest_path(shortest_path_nodes, shortest_path_maps)

        # # fetching the maps corresponding to shortest path
        # if shortest_path_maps:
        #     for map_ in shortest_path_maps:
        #         fetch_maps(args.e, map_)


    # ONLY FOR TESTING. SHOULD NOT BE CALLED OTHERWISE!!
    # when -u, -e, -mani are provided, map u is uploaded for env e from .ros with the manipulated length of the map
    elif args.u \
            and args.e \
            and args.snode \
            and args.enode \
            and not args.m \
            and not args.oe \
            and not args.delamap \
            and not args.delmaps \
            and not args.shortest \
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


main()
