import numpy as np
import cv2
from .utils import load_image, save_image

def warp_with_inverse_flow(image_path, flow_inv_path, out_image_path):
    '''
    im.shape = (H,W,3bgr)
    flow.shape = (H,W,2)
    '''

    im = load_image(image_path)
    flow_inv = np.load(flow_inv_path)

    h,w,_ = im.shape
    # coords.shape = (2,w,h) => (h,w,2)
    coords = np.indices((w,h), dtype=np.float32).transpose(2, 1, 0)

    flow_inv_abs = coords + flow_inv
    warped_image = cv2.remap(im, flow_inv_abs, None, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    save_image(warped_image, f'{out_image_path}')
    save_image(im, f'{out_image_path}.from.png')

    return warped_image