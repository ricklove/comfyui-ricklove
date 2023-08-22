import cv2
import numpy as np
from pathlib import Path
import json
import os

def save_image(image, file_path):
    if not Path(file_path).parent.exists():
        Path(file_path).parent.mkdir(parents=True)
    cv2.imwrite(file_path, image)
    print('save_image', file_path)

def load_image(image_path):
    'returns numpy.shape == (H, W, 3{bgr})'
    im = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # print('load_image', im.shape)
    return im

def save_json(data, file_path):
    if not Path(file_path).parent.exists():
        Path(file_path).parent.mkdir(parents=True)
    f = open(file_path, 'w')
    f.write(json.dumps(data))
    f.close()
    print('save_json', file_path)

def save_numpy(data, file_path):
    if not Path(file_path).parent.exists():
        Path(file_path).parent.mkdir(parents=True)
    np.save(file_path, data)
    print('save_numpy', file_path)

def file_exists(file_path):
    return os.path.exists(file_path)