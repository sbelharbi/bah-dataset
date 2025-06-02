import copy
import math
import os
import random
import sys
from os.path import dirname, abspath, join, basename, expanduser, normpath
import pickle as pkl
from collections import Counter
import csv
import platform
import re
import datetime
from datetime import datetime as dt
from pprint import pprint
import numbers
from typing import Tuple, Union

import numpy as np
import yaml


import constants

root_dir = dirname((abspath(__file__)))
sys.path.append(root_dir)


_STATS_DIR = 'absolute-folder-containing-video_annotation.yaml'

_BASEURL = get_root_wsol_dataset()



def overload_real_frame_labels():
    ds = constants.BAH_DB
    data_dir = join(_BASEURL, ds, 'features/compacted_48')
    annotation_path = join(_STATS_DIR, 'video_annotation.yaml')
    assert os.path.isfile(annotation_path), annotation_path

    with open(annotation_path, 'r') as fx:
        fr_annot = yaml.full_load(fx)

    for k in fr_annot:
        new_anno_file = join(data_dir, k, 'EXPR_continuous_label.npy')
        assert os.path.isfile(new_anno_file), new_anno_file
        old_annot = np.load(new_anno_file, mmap_mode='c')
        new_data = fr_annot[k]['frame_annotation']
        new_data = [item[1] for item in new_data]
        new_annot = np.array(new_data, dtype=np.float32)
        assert new_annot.shape == old_annot.shape, f"{new_annot.shape} | " \
                                                   f"{old_annot.shape}"
        np.save(new_anno_file, new_annot)



if __name__ == "__main__":
    overload_real_frame_labels()
