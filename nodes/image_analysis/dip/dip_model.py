# Dip
import cv2
import numpy as np
import torch
import time

from ....ext.DIP.nets.dip import DIP
from ....ext.DIP.nets.utils.utils import InputPadder, forward_interpolate
from ....ext.DIP.demo import flow_to_image
from ..image_warp import warp_with_inverse_flow
from ..utils import load_image, save_image, save_numpy, file_exists
import folder_paths

DEVICE = 'cuda'

class DipModel:
    def __init__(self, max_offset=256):
        model_path = f'{folder_paths.models_dir}/dip/DIP_sintel.pth'

        model = DIP(max_offset=max_offset, mixed_precision=False, test_mode=False)
        model = torch.nn.DataParallel(model)
        model.cuda()

        pre_train = torch.load(model_path)
        model.load_state_dict(pre_train, strict=False)
        model.eval()
        self.model = model

    def __call__(self, input_image_a, input_image_b, iters=30, max_offset_ratio=0.5):
        model = self.model

        print('input_image_a raw', input_image_a.shape, input_image_b.shape)

        # assert input_image_a.ndim == 3
        # input_image_a = input_image_a[:, :, ::-1].copy()

        # assert input_image_b.ndim == 3
        # input_image_b = input_image_b[:, :, ::-1].copy()

        with torch.no_grad():
            img1 = input_image_a[0]
            img2 = input_image_b[0]

            image1 = img1.permute(2, 0, 1).float()
            image1 = image1[None].to(DEVICE)
            image2 = img2.permute(2, 0, 1).float()
            image2 = image2[None].to(DEVICE)

            print('image1 permuted', image1.shape, image2.shape)

            padder = InputPadder(image1.shape)
            image1_padded, image2_padded = padder.pad(image1, image2)

            # flow_prev = None
            # if flow_prev is None:
            flow_up = model(image1_padded, image2_padded, iters=iters, init_flow=None)
            # else:
            #     flow_up = model(image1_padded, image2_padded, iters=iters, init_flow=flow_prev)

            flow_up = flow_up[-1]

            # print('flow_up', flow_up)
            flow_up = padder.unpad(flow_up)
            flo = flow_up[0].view(2, flow_up[0].shape[-2], flow_up[0].shape[-1])
            flo = flo.permute(1,2,0).cpu().numpy()

            # flo.shape = (H,W,2)

            color_flow = flow_to_image(flo, clip_flow=None, convert_to_bgr=True)
            # save_image(color_flow, out_flow_image_path)

            # # flow = (1 to 2)
            # # used as inv_flow = (each pixel 1 from where in 2)
            # out_warp_image_path = f'{out_path}/warped/{i:05}_{j:05}.warped{iter_name}.png'
            # warped_image = warp_with_inverse_flow(img_path2, out_flow_path, out_warp_image_path)

            return flo, color_flow
