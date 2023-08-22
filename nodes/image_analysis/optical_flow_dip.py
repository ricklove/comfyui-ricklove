import cv2
import numpy as np
import torch

from ...ext.DIP.nets.dip import DIP
from ...ext.DIP.nets.utils.utils import InputPadder, forward_interpolate
from ...ext.DIP.demo import flow_to_image
from .image_warp import warp_with_inverse_flow
from .utils import load_image, save_image, save_numpy, file_exists

DEVICE = 'cuda'

def analyze_flow(image_path_prefix, out_path, models_dir, i_start, i_end, i_range=5 ):
    model_path = f'{models_dir}/dip/DIP_sintel.pth'
    print('model_path', model_path)

    model = DIP(max_offset=512, mixed_precision=False, test_mode=False)
    model = torch.nn.DataParallel(model)
    model.cuda()

    pre_train = torch.load(model_path)
    model.load_state_dict(pre_train, strict=False)
    model.eval()

    warm_start  = True
    iters_cool = 20
    iters_warm = 12
    iters_warm_count = 3
    
    with torch.no_grad():
        for i in range(i_start, i_end+1):
            torch.cuda.empty_cache()

            img_path1 = f'{image_path_prefix}{i:05}.png'
            img1 = load_image(img_path1)
            save_image(img1, f'{out_path}/orig/{i:05}.orig.a.png')

            flow_prev = None
            for j in range(i_start, i_end+1):
                if abs(i-j) >= i_range: continue

                img_path2 = f'{image_path_prefix}{j:05}.png'
                img2 = load_image(img_path2)
                # save_image(img2, f'{out_path}/orig/{j:05}.orig.b.png')

                image1 = torch.from_numpy(img1).permute(2, 0, 1).float()
                image1 = image1[None].to(DEVICE)

                image2 = torch.from_numpy(img2).permute(2, 0, 1).float()
                image2 = image2[None].to(DEVICE)

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                for iter in range(iters_warm_count):
                    iter_name = f'.{iter:03}'
                    out_flow_path = f'{out_path}/flow/{j:05}_{i:05}.flow{iter_name}.npy'
                    out_flow_image_path = f'{out_path}/flow/{j:05}_{i:05}.flow{iter_name}.png'

                    if file_exists(out_flow_path): continue

                    if warm_start & iter > 0:
                        out_flow_path_prev = f'{out_path}/flow/{j:05}_{i:05}.flow.{(iter-1):03}.npy'
                        flow_up = np.load(out_flow_path_prev)
                        flow_up = torch.from_numpy(flow_up)
                        flow_up = padder.pad(flow_up)
                        flow_prev = forward_interpolate(flow_up[0])[None].cuda()

                    if flow_prev is None:
                        flow_up = model(image1, image2, iters=iters_cool, init_flow=None)[-1]
                    else:
                        flow_up = model(image1, image2, iters=iters_warm, init_flow=flow_prev)[-1]

                    # print('flow_up', flow_up)
                    flow_up = padder.unpad(flow_up)
                    flo = flow_up[0].view(2, flow_up[0].shape[-2], flow_up[0].shape[-1])
                    flo = flo.permute(1,2,0).cpu().numpy()

                    save_numpy(flo, out_flow_path)
                    print(out_flow_path)

                    color_flow = flow_to_image(flo, clip_flow=None, convert_to_bgr=True)
                    save_image(color_flow, out_flow_image_path)

                    # flow = (1 to 2)
                    # used as inv_flow = (each pixel 1 from where in 2)
                    out_warp_image_path = f'{out_path}/warped/{i:05}_{j:05}.warped{iter_name}.png'
                    warp_with_inverse_flow(img_path2, out_flow_path, out_warp_image_path)


                