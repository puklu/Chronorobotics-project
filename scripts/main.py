from copyreg import pickle
import pickle
import os
import argparse
from pathlib import Path
from zipfile import ZipFile

from utils import *
# from utils import load_map, create_buckets, env_upload, map_upload, map_upload2, delete_a_map, delete_all_maps_of_an_environment,
from constants import ROOT, OBJECTS_PATH, CLIENT, ENV_BUCKET, MAP_BUCKET, FETCHED_MAP_OBJ_PATH, FETCHED_ENV_OBJ_PATH, \
    FETCHED_MAPS_PATH, DOT_ROS_PATH
from classes_ import Map, Environment


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

    args = parser.parse_args()

    # when no arguments are provided, buckets are created and all the maps for all the envs are uploaded
    if not args.e and not args.m and not args.u and not args.oe and not args.snode and not args.enode and not args.delamap and not args.delmaps:
        create_buckets()  # create the buckets
        # upload_objects2()  # upload to db

    # delete all maps from an environment
    elif args.delmaps and not args.e and not args.m and not args.u and not args.oe and not args.snode and not args.enode and not args.delamap:
        delete_all_maps_of_an_environment(args.delmaps)

    # deleting a map
    elif args.delamap and args.e and not args.m and not args.u and not args.oe and not args.snode and not args.enode and not args.delmaps:
        map_name = args.delamap
        env_name = args.e
        delete_a_map(env_name, map_name)

    # when only -oe is provided, only env object is fetched
    elif args.oe and not args.e and not args.m and not args.u and not args.snode and not args.enode and not args.delamap and not args.delmaps:
        env_obj = fetch_environment(args.oe)
        if env_obj:
            env_map_metadata = env_obj.map_metadata
            print(f"env name: {env_obj.name}")
            print(f"nodes in the env: {env_obj.nodes}")
            print(f"maps in the env: {env_map_metadata['maps_names']}")
            print(f"distance: {env_map_metadata['distance']}")
            print(f"start nodes: {env_map_metadata['start_node']}")
            print(f"end nodes: {env_map_metadata['end_node']}")
        else:
            print(f"{args.oe} doesn't exist in db")

    # when only -e is provided, all the maps for that particular environment are fetched
    elif args.e and not args.m and not args.u and not args.oe and not args.snode and not args.enode and not args.delamap and not args.delmaps:
        fetch_maps2(args.e, args.m)

    # when -e and -m are provided, that particular map is fetched for that environment
    elif args.e and args.m and not args.u and not args.oe and not args.snode and not args.enode and not args.delamap and not args.delmaps:
        fetch_maps2(args.e, args.m)

    # when -u and -e are provided, map u is uploaded for env e from .ros
    elif args.u and args.e and args.snode and args.enode and not args.m and not args.oe and not args.delamap and not args.delmaps:
        create_buckets()  # create the buckets

        path = Path(f"{DOT_ROS_PATH}/{args.u}")

        if not path.is_dir():
            print(f"The map {args.u} doesn't exist in local")
            return

        # zipping the map

        with ZipFile(f"{DOT_ROS_PATH}/{args.e}.{args.u}.zip", 'w') as zip:
            for path, directories, files in os.walk(path):
                for file in files:
                    file_name = os.path.join(path, file)
                    zip.write(file_name, arcname=f"{args.u}/{os.path.basename(file_name)}")  # zipping the file

        map_obj_name = f"{args.e}.{args.u}.zip"  # name to be used for the map object
        map_name = f"{args.u}"
        start_node = args.snode
        end_node = args.enode

        # Fetch env obj from db, append to it, then replace the one in db
        env_obj = fetch_environment(args.e)
        if env_obj:  # if the env exists in db then update it
            env_obj = extract_map_metadata2(env_obj=env_obj, map_name=map_name, start_node=start_node,
                                            end_node=end_node)
            if map_name not in env_obj.map_metadata['maps_names']:
                env_obj = extract_map_metadata2(env_obj=env_obj, map_name=map_name, start_node=start_node,
                                                end_node=end_node)

        else:  # else create the env obj
            env_obj = Environment(name=args.e, gps_position=None)  # env object for the current environment
            env_obj = extract_map_metadata2(env_obj=env_obj, map_name=map_name, start_node=start_node,
                                            end_node=end_node)

        map_path = f"{str(DOT_ROS_PATH)}/{args.e}.{args.u}.zip"
        map_upload2(env_name=args.e, obj_name=map_obj_name, map_name=map_name, path=map_path)
        env_upload(env_data=env_obj)

        # Delete the zip file
        os.remove(f"{str(DOT_ROS_PATH)}/{args.e}.{args.u}.zip")

    else:
        raise Exception("Please try again, something is missing..")


main()
