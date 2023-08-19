import cv2
import numpy as np
import os
import sys
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.io import read_video, read_image, ImageReadMode
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from torchvision.io import write_jpeg
import torchvision.transforms as T
import tempfile
from pathlib import Path
from urllib.request import urlretrieve
from scipy.interpolate import LinearNDInterpolator
from imageio import imread, imwrite
from torchvision.utils import flow_to_image
import math
import json

def save_image(image, file_path):
    if not Path(file_path).parent.exists():
        Path(file_path).parent.mkdir(parents=True)
    cv2.imwrite(file_path, image)

def load_image_bgr(image_path):
    image = Image.open(image_path)
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image_bgr

def save_json(data, file_path):
    if not Path(file_path).parent.exists():
        Path(file_path).parent.mkdir(parents=True)
    f = open(file_path, 'w')
    f.write(json.dumps(data))
    f.close()

# ref: https://learnopencv.com/image-quality-assessment-brisque/
# ref: https://github.com/rehanguha/brisque/blob/master/brisque/brisque.py
def analyze_image_quality(image_path, working_name, working_path, model_path):
    img_bgr = load_image_bgr(image_path) 

    def save_image_by_kind(img, kind):
        save_image(img, f'{working_path}/{kind}/{working_name}.png')

    def save_json_by_kind(data, kind):
        save_json(data, f'{working_path}/{kind}/{working_name}.json')

    save_image_by_kind(img_bgr, 'orig')

    def mean_subtracted_contrast_normalization(im):
        blurred = cv2.GaussianBlur(im, (7, 7), 1.166) # apply gaussian blur to the image
        blurred_sq = blurred * blurred
        sigma = cv2.GaussianBlur(im * im, (7, 7), 1.166)
        sigma = (sigma - blurred_sq) ** 0.5
        sigma = sigma + 1.0/255 # to make sure the denominator doesn't give DivideByZero Exception
        structdis = (im - blurred)/sigma # final MSCN(i, j) image
        return structdis

    img_mscn = mean_subtracted_contrast_normalization(img_bgr)
    save_image_by_kind(img_mscn, 'mscn')

    def pairwise_diffs(img_mscn):
        structdis = img_mscn
        # indices to calculate pair-wise products (H, V, D1, D2)
        shifts = [[0,1], [1,0], [1,1], [-1,1]]
        ShiftArrs = []
        # calculate pairwise components in each orientation
        for i in range(0, len(shifts)):
            OrigArr = structdis
            reqshift = shifts[i]
        
            # create affine matrix (to shift the image)
            M = np.float32([[1, 0, reqshift[1]], [0, 1, reqshift[0]]])
            ShiftArr = cv2.warpAffine(OrigArr, M, (structdis.shape[1], structdis.shape[0]))
            ShiftArrs.append(ShiftArr)

        return ShiftArrs
    
    imgs_ShiftArr = pairwise_diffs(img_mscn)
    for i, img_ShiftArr in enumerate(imgs_ShiftArr):
        save_image_by_kind(img_ShiftArr, f'ShiftArr{i}')
    
    # def generalized_gaussian_distribution(x, alpha, sigma):

    #     beta = sigma * np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))

    #     coefficient = alpha / (2 * beta() * special.gamma(1 / alpha))
    #     return coefficient * np.exp(-(np.abs(x) / beta) ** alpha)


    def run_brisque(im):
        'the lower the better'
        # https://stackoverflow.com/a/72906238/567524
        # models at https://github.com/opencv/opencv_contrib/tree/master/modules/quality/samples
        obj = cv2.quality.QualityBRISQUE_create(f"{model_path}/brisque/brisque_model_live.yml", f"{model_path}/brisque/brisque_range_live.yml")
        score = obj.compute(im)
        # score = brisque.score(im)
        return score

    score = run_brisque(img_bgr)
    print(f'[{working_name}] brisque score', score)
    save_json_by_kind({'brisque_score':score}, 'score')



    # def brisque(im):
    #     # load the model from allmodel file
    #     model = svmutil.svm_load_model("allmodel")
    #     # create svm node array from features list
    #     x, idx = gen_svm_nodearray(x[1:], isKernel=(model.param.kernel_type == PRECOMPUTED))
    #     nr_classifier = 1 # fixed for svm type as EPSILON_SVR (regression)
    #     prob_estimates = (c_double * nr_classifier)()
        
    #     # predict quality score of an image using libsvm module
    #     qualityscore = svmutil.libsvm.svm_predict_probability(model, x, dec_values)