import os
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

data_path = join('../../data')
dataset_name = 'pick_and_place/c'

dataset_path = join(__location__, data_path, dataset_name)

onlyfiles = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]
n_files = len(onlyfiles)

first_diff = None
last_diff = None

for f in onlyfiles:
    idx = int(f.split('.')[0].split('_')[1])
    if idx == 0 or idx == n_files - 1:
        continue

    prev_frame = cv2.imread(join(dataset_path, f'frame_{str(idx - 1).zfill(5)}.jpg')).astype(np.float32)
    next_frame = cv2.imread(join(dataset_path, f'frame_{str(idx + 1).zfill(5)}.jpg')).astype(np.float32)
    diff = np.sum(np.abs(next_frame - prev_frame), axis=2) / (255 * 3)
    print(np.min(diff), np.max(diff))
    cv2.imshow('diff', diff)
    cv2.waitKey(-1)

    if idx == 1:
        first_diff = diff
    elif idx == n_files - 2:
        last_diff = diff
    print(idx)
