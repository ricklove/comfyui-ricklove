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
    print('save_image', file_path)

def load_image_bgr(image_path):
    'returns numpy.shape == (H, W, 3{bgr})'
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
    
    def save_flow_image(flow, path):
        if flow.dtype != torch.float:
            flow = torch.from_numpy(flow)
        flow_img = flow_to_image(flow)
        flow_img = flow_img.to("cpu").numpy()
        # flow_img.shape (3, 640, 480) -> (3{rgb}, H, W)
        # print('flow_img', flow_img.shape)

        #  (3, H, W) => (H, W, 3)
        flow_img = np.transpose(flow_img, (1,2,0))
        flow_img = cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR)
        save_image(flow_img, f'{path}')      
    
    def calculate_flow_data(noise=0):
        '''
        returns flow.shape = (2,H,W)
        '''

        flow_a2b_path = get_file_name('flow', f'{working_nameB}_{working_nameA}', f'flo.{noise}.npy')
        flow_b2a_path = get_file_name('flow', f'{working_nameA}_{working_nameB}', f'flo.{noise}.npy')

        if (os.path.exists(flow_a2b_path) 
            & os.path.exists(flow_b2a_path)):
            return np.load(flow_a2b_path), np.load(flow_b2a_path)

        imageA = load_image_bgr(imageA_path)
        imageB = load_image_bgr(imageB_path)
        h,w,_ = imageA.shape

        # if noise > 0:
        #     imageA = np.clip(imageA + noise, 0, 255)
        #     imageB = np.clip(imageB + noise, 0, 255)
        offset = noise
        if offset > 0:
            off8 = int(round(math.ceil(offset/8)*8))
            imageA = np.pad(imageA, ((0,off8),(0,off8),(0,0)))
            imageB = np.pad(imageB, ((offset,off8-offset),(offset,off8-offset),(0,0)))
        save_image(imageA, get_file_name('flow', f'{working_nameA}', f'offset_{offset:02}.a.png'))
        save_image(imageB, get_file_name('flow', f'{working_nameB}', f'offset_{offset:02}.b.png'))


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
            list_of_flows = model(a,b)
            predicted_flow = list_of_flows[-1][0]
            predicted_flow = predicted_flow.detach().cpu().numpy()
            return predicted_flow


        flow_a2b = calculate_flow(imageA_batch, imageB_batch, flow_a2b_path)
        flow_b2a = calculate_flow(imageB_batch, imageA_batch, flow_b2a_path)

        if offset > 0:
            # (2,H,W)
            flow_a2b = flow_a2b - offset
            flow_b2a = flow_b2a + offset
            flow_a2b = flow_a2b[0:2,0:h,0:w]
            flow_b2a = flow_b2a[0:2,offset:h+offset,offset:w+offset]

        save_flow_image(flow_a2b, f'{flow_a2b_path}.png')
        save_flow_image(flow_b2a, f'{flow_b2a_path}.png')
        np.save(flow_a2b_path, flow_a2b)
        np.save(flow_b2a_path, flow_b2a)

        return flow_a2b,flow_b2a
    
    def warp_with_inverse_flow(image_path, flow_inv, path):
        '''
        im.shape = (H,W,3bgr)
        flow.shape = (2,H,W)
        '''

        im = load_image_bgr(image_path)
        h,w,_ = im.shape
        # coords.shape = (2,w,h) => (h,w,2)
        coords = np.indices((w,h), dtype=np.float32).transpose(2, 1, 0)
        flow_inv = flow_inv.transpose(1, 2, 0)
        flow_inv_abs = coords + flow_inv

        warped_image = cv2.remap(im, flow_inv_abs, None, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        save_image(warped_image, f'{path}.png')
        return warped_image
    
    def warp_round_trip(image_path, flow_backward, flow_forward, path, MAX_DIFF=32):
        '''
        im.shape = (H,W,3bgr)
        flow.shape = (2,H,W)
        '''

        im = load_image_bgr(image_path)
        h,w,_ = im.shape
        # coords.shape = (2,w,h) => (h,w,2)
        def do_warp(im, flow_inv):
            coords = np.indices((w,h), dtype=np.float32).transpose(2, 1, 0)
            flow_inv = flow_inv.transpose(1, 2, 0)
            flow_inv_abs = coords + flow_inv

            return cv2.remap(im, flow_inv_abs, None, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        warped_image = do_warp(im, flow_backward)
        warped_image = do_warp(warped_image, flow_forward)
        save_image(warped_image, f'{path}.round_trip.png')

        # (H,W,3bgr)
        im_std = np.abs(1.0 * warped_image - im, dtype=np.float32)
        save_image(im_std, f'{path}.round_trip.std.png')

        warped_black = np.max(warped_image, axis=2)
        warped_black = np.where(warped_black>1,0,255)

        flow_mask = np.max(im_std, axis=2)
        flow_mask = np.where(flow_mask>MAX_DIFF,255,0)
        flow_mask = np.max([flow_mask, warped_black], axis=0)
        
        save_image(flow_mask, f'{path}.round_trip.flow_mask.png')

        return warped_image, flow_mask
    
    def warp_with_mask(image_from_path, image_to_path, flow_inv, flow_mask, path):
        '''
        im.shape = (H,W,3bgr)
        flow.shape = (2,H,W)
        '''

        # binary inverse mask
        flow_mask = np.where(flow_mask<=0,255,0)
        flow_mask = np.array([flow_mask,flow_mask,flow_mask]).transpose(1,2,0)

        warped = warp_with_inverse_flow(image_from_path, flow_inv, path)
        h,w,_ = warped.shape
        # im_default = load_image_bgr(image_to_path)
        im_default = np.zeros((h,w,3))
        im_masked = (im_default * (255.0-flow_mask) + warped * flow_mask) / 256.0

        save_image(im_masked, f'{path}.image_masked.png')

        return im_masked
    
    flow_data_count = 4
    flow_datas_a2b = []
    flow_datas_b2a = []
    flow_masks_a2b = []
    flow_masks_b2a = []
    for i in range(0, flow_data_count):
        flow_a2b_n,flow_b2a_n = calculate_flow_data(i)
        flow_datas_a2b.append(flow_a2b_n)
        flow_datas_b2a.append(flow_b2a_n)
        warp_with_inverse_flow(imageB_path, flow_a2b_n, get_file_name('warped', f'{working_nameA}_{working_nameB}', f'warped.{i:02}'))
        warp_with_inverse_flow(imageA_path, flow_b2a_n, get_file_name('warped', f'{working_nameB}_{working_nameA}', f'warped.{i:02}'))
        _, flow_mask_a2b = warp_round_trip(imageB_path, flow_a2b_n, flow_b2a_n, get_file_name('warp_rt', f'{working_nameB}_{working_nameA}_{working_nameB}', f'warp_rt.a.{i:02}'))
        _, flow_mask_b2a = warp_round_trip(imageA_path, flow_b2a_n, flow_a2b_n, get_file_name('warp_rt', f'{working_nameA}_{working_nameB}_{working_nameA}', f'warp_rt.b.{i:02}'))
        flow_masks_a2b.append(flow_mask_a2b)
        flow_masks_b2a.append(flow_mask_b2a)

    flow_a2b = np.average(flow_datas_a2b, axis=0)
    flow_b2a = np.average(flow_datas_b2a, axis=0)
    flow_a2b_std = 4 * np.max(np.std(flow_datas_a2b, axis=0), axis=0)
    flow_b2a_std = 4 * np.max(np.std(flow_datas_b2a, axis=0), axis=0)
    save_image(flow_a2b_std, get_file_name('flow_std', f'{working_nameB}_{working_nameA}', f'std_4.png'))
    save_image(flow_b2a_std, get_file_name('flow_std', f'{working_nameA}_{working_nameB}', f'std_4.png'))

    flow_mask_a2b = np.average(flow_masks_a2b, axis=0)
    flow_mask_b2a = np.average(flow_masks_b2a, axis=0)
    save_image(flow_mask_a2b, get_file_name('warp_rt', f'{working_nameB}_{working_nameA}_{working_nameB}', f'warp_rt.a.ave.round_trip.flow_mask.png'))
    save_image(flow_mask_b2a, get_file_name('warp_rt', f'{working_nameA}_{working_nameB}_{working_nameA}', f'warp_rt.b.ave.round_trip.flow_mask.png'))
    save_image(flow_mask_a2b, get_file_name('flow_mask', f'{working_nameB}_{working_nameA}_{working_nameB}', f'warp_rt.a.ave.round_trip.flow_mask.png'))
    save_image(flow_mask_b2a, get_file_name('flow_mask', f'{working_nameA}_{working_nameB}_{working_nameA}', f'warp_rt.b.ave.round_trip.flow_mask.png'))

    save_flow_image(torch.from_numpy(flow_a2b), get_file_name('flow', f'{working_nameB}_{working_nameA}', f'flo.npy.png'))
    save_flow_image(torch.from_numpy(flow_b2a), get_file_name('flow', f'{working_nameA}_{working_nameB}', f'flo.npy.png'))

    save_image(load_image_bgr(imageA_path), get_file_name('warped', f'{working_nameA}', f'orig.png'))
    save_image(load_image_bgr(imageB_path), get_file_name('warped', f'{working_nameB}', f'orig.png'))
    warp_with_inverse_flow(imageB_path, flow_a2b, get_file_name('warped', f'{working_nameA}_{working_nameB}', f'warped'))
    warp_with_inverse_flow(imageA_path, flow_b2a, get_file_name('warped', f'{working_nameB}_{working_nameA}', f'warped'))
    warp_round_trip(imageB_path, flow_a2b, flow_b2a, get_file_name('warp_rt', f'{working_nameB}_{working_nameA}_{working_nameB}', f'warp_rt.a'))
    warp_round_trip(imageA_path, flow_b2a, flow_a2b, get_file_name('warp_rt', f'{working_nameA}_{working_nameB}_{working_nameA}', f'warp_rt.b'))

    warp_with_mask(imageB_path, imageA_path, flow_a2b, flow_mask_a2b, get_file_name('warp_masked', f'{working_nameA}_{working_nameB}', f'warp_masked.a'))
    warp_with_mask(imageA_path, imageB_path, flow_b2a, flow_mask_b2a, get_file_name('warp_masked', f'{working_nameB}_{working_nameA}', f'warp_masked.b'))

    # print('flow', flow_a2b.shape, flow_b2a.shape)

    return

    debug_name = '.015'

    def evaluate_flow_quality(flow, path, image_from_path, image_to_path):
        'flow.shape = (2,H,W)'

        h = flow.shape[1]
        w = flow.shape[2]

        image_from = load_image_bgr(image_from_path)
        image_to = load_image_bgr(image_to_path)

        flow = np.vstack((flow, 
                    np.indices((h,w)), 
                    np.zeros((2,h,w)),
                    np.zeros((2,h,w)),
                    np.zeros((2,h,w)),
                    ))
        
        i_optical_distance = 9
        dim = 10
        dtype = [
            # [0] = xToRel
            # [1] = yToRel
            # [2] = xFrom
            # [3] = yFrom
            # [4] = xToAbs
            # [5] = yToAbs
            ('xToRel', np.float32), 
            ('yToRel', np.float32), 
            ('xFrom', np.float32),
            ('yFrom', np.float32),
            ('xToAbs', np.float32),
            ('yToAbs', np.float32),

            # [6] = dist^2
            # [7] = 
            # [8] = 
            # [9] = score = reconstruction_distance^2
            ('dist2', np.float32),
            ('_', np.float32),
            ('_', np.float32),
            ('optical_distance', np.float32),
        ]

        # swap h,w indices
        flow[4] = flow[2]
        flow[2] = flow[3]
        flow[3] = flow[4]

        flow[4] = flow[0]+flow[2]
        flow[5] = flow[1]+flow[3]
        flow[6] = flow[0]*flow[0]+flow[1]*flow[1]



        # default values (for pixels that point to out of bounds)
        flow[i_optical_distance] = 255

        # (n,H,W) -> (H,W,n) -> (H*W,n)
        flow_t = flow.transpose(1,2,0)
        flow_t = flow_t.reshape((flow_t.shape[0]*flow_t.shape[1], flow_t.shape[2]))
        # print('flow_t', flow_t.shape, flow_t )

        # flow_t = np.rec.fromarrays(flow_t.T, dtype=dtype)

        for v in flow_t:
            # print('v', v)
            xToRel,yToRel,xFrom,yFrom,xTo,yTo,_,_,_,_ = v

            x0 = int(xTo)
            y0 = int(yTo)

            if ((x0 < 0) | (x0 >= w)): continue
            if ((y0 < 0) | (y0 >= h)): continue

            x1 = min(x0 + 1, w-1)
            y1 = min(y0 + 1, h-1)

            xr0 = xTo - x0
            xr1 = 1 - xr0
            yr0 = yTo - y0
            yr1 = 1 - yr0

            pixel_from = image_from[int(yFrom),int(xFrom)]
            pixel_to = (
                yr0 * xr0 * image_to[y0,x0]
                + yr0 * xr1 * image_to[y0,x1]
                + yr1 * xr0 * image_to[y1,x0]
                + yr1 * xr1 * image_to[y1,x1]
            )

            delta_bgr = pixel_to - pixel_from
            optical_distance = np.linalg.norm(delta_bgr)

            v[i_optical_distance] = optical_distance

            # flow_from[0,y0,x0] = -xToRel
            # flow_from[1,y0,x0] = -yToRel
            # flow_from[0,y0,x1] = -xToRel
            # flow_from[1,y0,x1] = -yToRel
            # flow_from[0,y1,x0] = -xToRel
            # flow_from[1,y1,x0] = -yToRel
            # flow_from[0,y1,x1] = -xToRel
            # flow_from[1,y1,x1] = -yToRel

        # (H*W,n) => (H,W,n) => (n,H,W)
        flow_data = flow_t.reshape(h,w,dim).transpose(2,0,1)
        save_image(flow_data[i_optical_distance], f'{path}{debug_name}.optical_distance.png')

    evaluate_flow_quality(flow_a2b, get_file_name('quality', f'{working_nameB}_{working_nameA}', 'quality.npy'), imageA_path, imageB_path)
    evaluate_flow_quality(flow_b2a, get_file_name('quality', f'{working_nameA}_{working_nameB}', 'quality.npy'), imageB_path, imageA_path)

    return
    
    def calculate_inverse_flow(flow, path):
        'flow.shape = (2,H,W)'

        if (os.path.exists(path)):
            return np.load(path)

        h = flow.shape[1]
        w = flow.shape[2]

        print('flow', flow.shape)

        def save_large_image(f, f_path):
            fl_bg = np.zeros((2,h,w), dtype=np.float32)
            fl_bg[0] = cv2.resize(f[0],(w,h),interpolation=cv2.INTER_NEAREST)
            fl_bg[1] = cv2.resize(f[1],(w,h),interpolation=cv2.INTER_NEAREST)
            save_flow_image(torch.from_numpy(fl_bg), f'{f_path}.bg.png')

        def smooth_edge_preserving(fl,size,s,r):
            fl_edges = np.zeros((2,int(h/size),int(w/size)), dtype=np.float32)
            fl_edges[0] = cv2.edgePreservingFilter(fl[0], sigma_s=s, sigma_r=r)
            fl_edges[1] = cv2.edgePreservingFilter(fl[1], sigma_s=s, sigma_r=r)
            # print('fl_edges', fl_edges,fl_sm)
            save_large_image(fl_edges, f'{path}{debug_name}.smooth{size:02}.s{s:02}.r{int(r*100):02}')
            return fl_edges

        def resize_find_edges(fl, size):
            fl_sm = fl
            if size > 1:
                fl_sm = np.zeros((2,int(h/size),int(w/size)), dtype=np.float32)
                fl_sm[0] = cv2.resize(fl[0],(int(w/size),int(h/size)),interpolation=cv2.INTER_NEAREST)
                fl_sm[1] = cv2.resize(fl[1],(int(w/size),int(h/size)),interpolation=cv2.INTER_NEAREST)
            save_large_image(fl_sm, f'{path}{debug_name}.sm{size:02}')

            fl_sm = np.round(fl_sm) 
            save_large_image(fl_sm, f'{path}{debug_name}.sm{size:02}.round')

            for s_int in range(1,6):
                s = 2**s_int    
                for r_int in range(1,5+1):
                    r = r_int/5    
                    smooth_edge_preserving(fl_sm, size, s, r)

        # # resize_find_edges(flow, 32)
        # # resize_find_edges(flow, 16)
        # resize_find_edges(flow, 8)
        # # resize_find_edges(flow, 4)
        # resize_find_edges(flow, 1)

        should_smooth = True
        if should_smooth:
            flow = np.round(flow)
            flow = smooth_edge_preserving(flow, 1, 2, 1)
        # return


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
        # print('flow_t', flow_t.shape, flow_t )

        # (2,H,W)
        flow_from = np.zeros((2,h,w), dtype=np.float32)
        # print('flow_from', flow_from.shape, (h, w) )

        for v in flow_t:
            # print('v', v)
            xToRel,yToRel,xFrom,yFrom,xTo,yTo,_,_ = v

            x0 = round(xTo)
            y0 = round(yTo)

            if ((x0 < 0) | (x0 >= w)): continue
            if ((y0 < 0) | (y0 >= h)): continue

            x1 = min(x0 + 1, w-1)
            y1 = min(y0 + 1, h-1)

            flow_from[0,y0,x0] = -xToRel
            flow_from[1,y0,x0] = -yToRel
            flow_from[0,y0,x1] = -xToRel
            flow_from[1,y0,x1] = -yToRel
            flow_from[0,y1,x0] = -xToRel
            flow_from[1,y1,x0] = -yToRel
            flow_from[0,y1,x1] = -xToRel
            flow_from[1,y1,x1] = -yToRel

        print('flow_from', flow_from.shape, flow_from )
        save_flow_image(torch.from_numpy(flow_from), f'{path}{debug_name}.png')
        np.save(path, flow_from)

        # # # erode
        # # def erode(values):
        # #     nonzeros = values[values!=0]
        # #     return 0 if nonzeros.size <= 12 else values[12]
  
        # # flow_from[0] = ndimage.generic_filter(flow_from[0], erode, size=5)
        # # flow_from[1] = ndimage.generic_filter(flow_from[1], erode, size=5)
        # # # flow_from[0] = ndimage.generic_filter(flow_from[0], erode, size=3)
        # # # flow_from[1] = ndimage.generic_filter(flow_from[1], erode, size=3)
        # # # flow_from[0] = ndimage.generic_filter(flow_from[0], erode, size=3)
        # # # flow_from[1] = ndimage.generic_filter(flow_from[1], erode, size=3)
        # # save_flow_image(torch.from_numpy(flow_from), f'{path}.010.erode.png')

        # flow_from_sm = np.zeros((2,int(h/8),int(w/8)), dtype=np.float32)
        # flow_from_sm[0] = cv2.resize(flow_from[0],(int(w/8),int(h/8)),interpolation=cv2.INTER_NEAREST)
        # flow_from_sm[1] = cv2.resize(flow_from[1],(int(w/8),int(h/8)),interpolation=cv2.INTER_NEAREST)
        # save_flow_image(torch.from_numpy(flow_from_sm), f'{path}{debug_name}.sm.png')

        # # erode sm
        # def erode_3x3(values):
        #     nonzeros = values[values!=0]
        #     return 0 if nonzeros.size <= 4 else values[4]
  
        # flow_from_sm[0] = ndimage.generic_filter(flow_from_sm[0], erode_3x3, size=3)
        # flow_from_sm[1] = ndimage.generic_filter(flow_from_sm[1], erode_3x3, size=3)
        # save_flow_image(torch.from_numpy(flow_from_sm), f'{path}{debug_name}.sm.erode.png')

        # flow_from_bg = np.zeros((2,h,w), dtype=np.float32)
        # flow_from_bg[0] = cv2.resize(flow_from_sm[0],(w,h),interpolation=cv2.INTER_NEAREST)
        # flow_from_bg[1] = cv2.resize(flow_from_sm[1],(w,h),interpolation=cv2.INTER_NEAREST)
        # save_flow_image(torch.from_numpy(flow_from_bg), f'{path}{debug_name}.bg.png')

        # # # smear into blanks
        # # def smear(values):
        # #     nonzeros = values[values!=0]
        # #     return 0 if nonzeros.size == 0 else np.median(nonzeros)
  
        # # flow_from[0] = ndimage.generic_filter(flow_from[0], smear, size=7)
        # # flow_from[1] = ndimage.generic_filter(flow_from[1], smear, size=7)
        # # flow_from_torch = torch.from_numpy(flow_from)
        # # save_flow_image(flow_from_torch, f'{path}.006.smear.png')
        # # flow.

        # # print('flow', flow.shape, flow)

        return flow_from

    flow_inv_b_from_a = calculate_inverse_flow(flow_a2b, get_file_name('flow_inv', f'{working_nameB}_{working_nameA}', 'flo.inv.npy'))
    flow_inv_a_from_b = calculate_inverse_flow(flow_b2a, get_file_name('flow_inv', f'{working_nameA}_{working_nameB}', 'flo.inv.npy'))

    warped_a2b = warp_with_inverse_flow(imageA_path, flow_inv_b_from_a, get_file_name('warped', f'{working_nameB}_{working_nameA}', 'warped'))
    warped_b2a = warp_with_inverse_flow(imageB_path, flow_inv_a_from_b, get_file_name('warped', f'{working_nameA}_{working_nameB}', 'warped'))
