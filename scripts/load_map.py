"""
Author: Zdenek Rozsypalek
"""

import numpy as np
import os
# from classes_ import Features


class Features:
    def __init__(self) -> None:
        self.shape = None
        self.values = None


def load_map(mappaths):
    """
    Loads map from the .npy files
    Args:
        mappaths: Path of the map on local

    Returns: images, distances, trans, times
    """
    images = []
    distances = []
    trans = []
    times = []

    if "," in mappaths:
        mappaths = mappaths.split(",")
    else:
        mappaths = [mappaths]

    for map_idx, mappath in enumerate(mappaths):
        tmp = []
        for file in list(os.listdir(mappath)):
            if file.endswith(".npy"):
                tmp.append(file[:-4])
        print(str(len(tmp)) + " images found in the map")

        # rospy.logwarn(str(len(tmp)) + " images found in the map")
        tmp.sort(key=lambda x: float(x))

        # print(map_idx, tmp)

        tmp_images = []
        tmp_distances = []
        tmp_trans = []
        tmp_times = []

        for idx, dist in enumerate(tmp):

            tmp_distances.append(float(dist))
            feature = Features()
            with open(os.path.join(mappath, dist + ".npy"), 'rb') as fp:
                map_point = np.load(fp, allow_pickle=True, fix_imports=False).item(0)
                r = map_point["representation"]
                ts = map_point["timestamp"]
                diff_hist = map_point["diff_hist"]

                if map_idx > 0 and map_point["source_map_align"] != mappaths[0]:
                    print("Multimap with invalid target!" + str(mappath))
                    # rospy.logwarn("Multimap with invalid target!" + str(mappath))
                    raise Exception("Invalid map combination")

                feature.shape = r.shape
                feature.values = r.flatten()
                tmp_images.append(feature)
                tmp_times.append(ts)

                if diff_hist is not None:
                    tmp_trans.append(diff_hist)
                # rospy.loginfo("Loaded feature: " + dist + str(".npy"))
                # print("Loaded feature: " + dist + str(".npy"))

        tmp_times[-1] = tmp_times[-2]  # to avoid very long period before map end
        images.append(tmp_images)
        distances.append(tmp_distances)
        trans.append(tmp_trans)
        times.append(tmp_times)
        # rospy.logwarn("Whole map " + str(mappath) + " sucessfully loaded")
        print("Whole map " + str(mappath) + " successfully loaded")

        return images, distances, trans, times
