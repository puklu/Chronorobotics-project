import os
from dotenv import load_dotenv
from pathlib import Path
from minio import Minio
import time

CURRENT_SYSTEM_TIME = time.time() # TODO: this might need a change


load_dotenv()
END_POINT = os.getenv("END_POINT")
ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

ROOT = Path(__file__).absolute().parent.parent.parent
OBJECTS_PATH = ROOT / "objects/"
DOWNLOADED_MAPS_PATH = ROOT / 'downloaded_objects/maps/'
DOWNLOADED_ENVS_PATH = ROOT / 'downloaded_objects/envs/'
TO_UPLOAD_PATH = Path(ROOT / 'pickled_objects/')

DOT_ROS_PATH = os.path.expanduser('~/.ros')
FETCHED_MAPS_PATH = os.path.expanduser('~/.ros/fetched_maps')
FETCHED_MAP_OBJ_PATH = Path(f"{ROOT}/fetched_objects/maps")
FETCHED_ENV_OBJ_PATH = Path(f"{ROOT}/fetched_objects/environment")

IMAGES_PATH = ROOT / 'scripts/Map_management/images/'

SIAMESE_NETWORK_PATH = ROOT / 'scripts/Siamese_network_image_alignment/'
PATH_TO_SIAMESE_MODEL = SIAMESE_NETWORK_PATH/ 'model_eunord.pt'

# Buckets
ENV_BUCKET = "environment"
MAP_BUCKET = "maps"
FIRST_IMAGE_BUCKET = "first-image"

CLIENT = Minio(END_POINT,
               access_key=ACCESS_KEY,
               secret_key=SECRET_KEY,
               secure=False)
