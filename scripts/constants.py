import os
from dotenv import load_dotenv
from pathlib import Path
from minio import Minio

load_dotenv()
END_POINT = os.getenv("END_POINT")
ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

ROOT = Path(__file__).absolute().parent.parent
OBJECTS_PATH = ROOT / "objects/"
DOWNLOADED_MAPS_PATH = ROOT / 'downloaded_objects/maps/'
DOWNLOADED_ENVS_PATH = ROOT / 'downloaded_objects/envs/'
TO_UPLOAD_PATH = ROOT / 'pickled_objects/'

ENV_BUCKET = "environment"
MAP_BUCKET = "maps"

CLIENT = Minio( END_POINT,
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            secure=False)
