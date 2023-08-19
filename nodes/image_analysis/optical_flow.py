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
import scipy.ndimage as ndimage

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

device = "cuda" if torch.cuda.is_available() else "cpu"
model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
model = model.eval()

def analyze_image_flow(imageA_path, imageB_path, working_nameA, working_nameB, working_path, model_path):
    def get_file_name(kind, working_name, ext):
        return f'{working_path}/{kind}/{working_name}.{ext}'
    
    def calculate_flow_data():
        flow_a2b_path = get_file_name('flow', f'{working_nameB}_{working_nameA}', 'flo.npy')
        flow_b2a_path = get_file_name('flow', f'{working_nameA}_{working_nameB}', 'flo.npy')

        if (os.path.exists(flow_a2b_path) 
            & os.path.exists(flow_b2a_path)):
            return np.load(flow_a2b_path), np.load(flow_b2a_path)

        imageA = load_image_bgr(imageA_path) 
        imageB = load_image_bgr(imageB_path)

        def infer(frameA, frameB):
            # print('infer: loading raft model')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
            model = model.eval()

            # print('infer: running')

            # Check if both frames have the same size
            if frameA.size != frameB.size:
                raise ValueError("Both input frames must have the same size")

            transform = T.ToTensor()

            img1_batch = transform(frameA)
            img2_batch = transform(frameB)
            img1_batch = torch.stack([img1_batch])
            img2_batch = torch.stack([img2_batch])
            weights = Raft_Large_Weights.DEFAULT
            transforms = weights.transforms()

            def preprocess(img1_batch, img2_batch):
                return transforms(img1_batch, img2_batch)

            img1_batch, img2_batch = preprocess(img1_batch, img2_batch)

            # print('infer: DONE')

            return img1_batch, img2_batch

        imageA_batch,imageB_batch = infer(imageA, imageB)
        imageA_batch,imageB_batch = imageA_batch.to(device), imageB_batch.to(device)

        def calculate_flow(a,b, path):
            if (os.path.exists(path)):
                return np.load(path)
        
            list_of_flows = model(a,b)
            predicted_flow = list_of_flows[-1][0]
            flow_img = flow_to_image(predicted_flow)
            flow_img = flow_img.to("cpu").numpy()
            # flow_img.shape (3, 640, 480) -> (3{rgb}, H, W)
            # print('flow_img', flow_img.shape)

            #  (3, H, W) => (H, W, 3)
            flow_img = np.transpose(flow_img, (1,2,0))
            flow_img = cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR)
            save_image(flow_img, f'{path}.png')

            predicted_flow = predicted_flow.detach().cpu().numpy()
            np.save(path, predicted_flow)

            return predicted_flow


        flow_a2b = calculate_flow(imageA_batch, imageB_batch, flow_a2b_path)
        flow_b2a = calculate_flow(imageB_batch, imageA_batch, flow_b2a_path)

        return flow_a2b,flow_b2a
    
    flow_a2b,flow_b2a = calculate_flow_data()
    # print('flow', flow_a2b.shape, flow_b2a.shape)
    
    def calculate_forward_flow(flow, path):
        'flow.shape = (2,H,W)'

        h = flow.shape[1]
        w = flow.shape[2]

        print('flow', flow.shape)

        flow = np.vstack((flow, 
                          np.indices((h,w)), 
                          np.zeros((2,h,w)),
                          np.zeros((2,h,w)),
                          ))

        # swap h,w indices
        flow[4] = flow[2]
        flow[2] = flow[3]
        flow[3] = flow[4]

        flow[4] = flow[0]+flow[2]
        flow[5] = flow[1]+flow[3]
        flow[6] = flow[0]*flow[0]+flow[1]*flow[1]
        flow[7] = flow[6]

        # [0] = xToRel
        # [1] = yToRel
        # [2] = xFrom
        # [3] = yFrom
        # [4] = xToAbs
        # [5] = yToAbs
        # [6] = dist^2
        # [7] = dist^2 (order)
        dtype = [
            ('xToRel', np.float32), 
            ('yToRel', np.float32), 
            ('xFrom', np.float32),
            ('yFrom', np.float32),
            ('xToAbs', np.float32),
            ('yToAbs', np.float32),
            ('dist2', np.float32),
            ('order', np.float32),
            ]


        # (n,H,W) -> (W,H,n) -> (W*H,n)
        flow_t = flow.transpose(2,1,0)
        flow_t = flow_t.reshape((flow_t.shape[0]*flow_t.shape[1], flow_t.shape[2]))
        # print('flow_t', flow_t.shape, flow_t )

        flow_t = np.rec.fromarrays(flow_t.T, dtype=dtype)
        # print('flow_t', flow_t.shape, flow_t )

        flow_t = np.sort(flow_t, order='order')
        print('flow_t', flow_t.shape, flow_t )

        # (2,H,W)
        flow_from = np.zeros((2,h,w), dtype=np.float32)
        print('flow_from', flow_from.shape, (h, w) )

        for v in flow_t:
            # print('v', v)
            xToRel,yToRel,xFrom,yFrom,xTo,yTo,_,_ = v

            x0 = int(xTo)
            y0 = int(yTo)
            if ((x0 < 0) | (x0 >= w)): continue
            if ((y0 < 0) | (y0 >= h)): continue

            # print('flow_from', flow_from.shape, x0, y0, xTo, yTo )
            flow_from[0,y0,x0] = -xToRel
            flow_from[1,y0,x0] = -yToRel

        print('flow_from', flow_from.shape, flow_from )
        flow_from = torch.from_numpy(flow_from)
        flow_from_img = flow_to_image(flow_from)
        flow_from_img = flow_from_img.to("cpu").numpy()
        flow_from_img = np.transpose(flow_from_img, (1,2,0))
        flow_from_img = cv2.cvtColor(flow_from_img, cv2.COLOR_RGB2BGR)
        save_image(flow_from_img, f'{path}..png')

        # def draw_flow(values):
        #     print(values)
        #     return values.sum()
        # footprint = np.array([[1,1,1],
        #       [1,0,1],
        #       [1,1,1]])
        # ndimage.generic_filter(flow, test_func, footprint=footprint)
        # flow.

        # print('flow', flow.shape, flow)

    
    calculate_forward_flow(flow_a2b, get_file_name('flow_from', f'{working_nameB}_{working_nameA}', 'flo.npy'))
    calculate_forward_flow(flow_b2a, get_file_name('flow_from', f'{working_nameA}_{working_nameB}', 'flo.npy'))