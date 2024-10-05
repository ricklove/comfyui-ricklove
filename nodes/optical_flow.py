import torch
import numpy as np
from .image_analysis.dip.dip_model import DipModel
from .image_analysis.image_warp import warp_with_inverse_flow_np

def np2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class RL_Optical_Flow_Dip:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_to": ("IMAGE",), 
                "image_from": ("IMAGE",),
                "iters": ("INT",{"default": 20, "min": 1, "max": 1000, "step": 1}),
                "max_offset_ratio": ("FLOAT",{"default": 0.5, "min": 0.01, "max": 1, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("FLO","IMAGE",)
    FUNCTION = "analyze_flow"

    CATEGORY = "ricklove/flow"

    def analyze_flow(self, image_to, image_from, iters=20, max_offset_ratio=0.5):
        max_offset = max(image_to.shape) * max_offset_ratio
        model = DipModel(max_offset)
        flo, color_flow = model(image_to, image_from, iters,max_offset_ratio)
        return (flo, np2tensor(color_flow),)
    
class RL_Warp_Image:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",), 
                "flo": ("FLO",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "warp_image"

    CATEGORY = "ricklove/flow"

    def warp_image(self, image, flo):
        # print('warp_image', image.shape, flo.shape, flo)
        image = (image.cpu().numpy().squeeze() * 255.0).astype(np.uint8)
        # print('warp_image', image.shape, image)

        warped_image = warp_with_inverse_flow_np(image, flo)
        # print('warped_image', warped_image.shape, warped_image)

        return (np2tensor(warped_image),)
    
class RL_Save_Flow:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "flo": ("FLO",), 
                "path": ("STRING",),
            },
        }

    RETURN_TYPES = ("FLO",)
    FUNCTION = "save_flow"

    CATEGORY = "ricklove/flow"

    def save_flow(self, flo, path):
        # print('save_flow', flo.shape, path)
        np.save(path, flo.cpu().numpy().squeeze())
        return (flo,)
    
class RL_Load_Flow:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING",),
            },
        }

    RETURN_TYPES = ("FLO",)
    FUNCTION = "load_flow"

    CATEGORY = "ricklove/flow"

    def load_flow(self, path):
        # print('load_flow', path)
        flo = np.load(path)
        return (torch.from_numpy(flo).unsqueeze(0),)