### custom_nodes.comfy_controlnet_preprocessors.v11.zoe

# ZoeDepth
# https://github.com/isl-org/ZoeDepth

import os
import cv2
import numpy as np
import torch

from einops import rearrange
from custom_nodes.comfy_controlnet_preprocessors.v11.zoe.zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepth
from custom_nodes.comfy_controlnet_preprocessors.v11.zoe.zoedepth.utils.config import get_config
from custom_nodes.comfy_controlnet_preprocessors.util import annotator_ckpts_path, load_file_from_url
import comfy.model_management

from custom_nodes.comfy_controlnet_preprocessors.v11.zoe.zoedepth.utils.misc import colorize, save_raw_16bit
from PIL import Image

def processZoeDept(depth, normMin, normMax, cutoffMin=None, cutoffMax=None):
    # --- manually create rgb
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().cpu().numpy()

    # norm = False
    # normMin = 2
    # normMax = 85
    
    norm = True
    # normMin = 2
    # normMax = 60
    if norm:
        # normalize
        depth = depth.squeeze()
        # invalid_mask = depth == -99
        # mask = np.logical_not(invalid_mask)
        mask = np.logical_not(False)

        # invert colors because everything is negative?
        nMin = normMax
        nMax = normMin

        vmin = np.percentile(depth[mask],nMin)
        vmax = np.percentile(depth[mask],nMax)
        if vmin != vmax:
            depth = (depth - vmin) / (vmax - vmin)  # vmin..vmax
        else:
            # Avoid 0-division
            depth = depth * 0.

    if cutoffMin is not None and cutoffMax is not None:
        depth = (depth - cutoffMin) / (cutoffMax - cutoffMin)
        depth = np.clip(depth, 0, 1)

    # extract precision into channels
    depth1 = depth.astype(np.uint8)
    depth1 = Image.fromarray(depth1).convert('RGB')
    depth1.save('./temp-test-depth1.png')
    r1,g1,b1 = depth1.split()

    depth = depth * 256 # scale for lower 16-bit
    depth2 = depth.astype(np.uint8)
    depth2 = Image.fromarray(depth2).convert('RGB')
    depth2.save('./temp-test-depth2.png')
    r2,g2,b2 = depth2.split()

    depth = depth * 256 # scale for lower 16-bit
    depth3 = depth.astype(np.uint8)
    depth3 = Image.fromarray(depth3).convert('RGB')
    depth3.save('./temp-test-depth3.png')
    r3,g3,b3 = depth3.split()

    depth = depth * 256 # scale for lower 16-bit
    depth4 = depth.astype(np.uint8)
    depth4 = Image.fromarray(depth4).convert('RGBA')
    depth4.save('./temp-test-depth4.png')
    r4,g4,b4,a4 = depth4.split()

    colored_image = Image.merge('RGBA', (r1, g2, b3, a4))
    colored_image.save('./temp-test-depth-c.png')

    colored_rgb = np.array(colored_image)
    # colored_rgb = colored_rgb.astype(np.float32) / 255.0

    # testing
    colored_rgb = rearrange(colored_rgb, 'h w c -> c h w')


    # failed: assert x.dtype == np.uint8

    return colored_rgb

    # # --- from save_raw_16bit
    # if isinstance(depth, torch.Tensor):
    #     depth = depth.squeeze().cpu().numpy()
    # depth = depth * 256  # scale for 16-bit png
    # depth1 = depth.astype(np.uint16)
    # depth1 = Image.fromarray(depth1)
    # depth1.save('./temp-test.png')

    # # depth = depth / 256  # scale back for 8-bit png
    # # depth2 = depth.astype(np.uint8)
    # # depth2 = Image.fromarray(depth2).convert('RGB')
    # # depth2.save('./temp-test-2.png')

    # depth_np = np.array(depth)
    # return depth_np

    # # --- creates 3 copies of grayscale
    # # numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    # colored = colorize(depth)
    # colored_image = Image.fromarray(colored)
    # colored_image.save('./temp-test.png')

    # colored_rgb = colored_image.convert("RGB")
    # colored_rgb = np.array(colored_rgb)
    # # colored_rgb = colored_rgb.astype(np.float32) / 255.0

    # # testing
    # colored_rgb = rearrange(colored_rgb, 'h w c -> c h w')


    # # failed: assert x.dtype == np.uint8

    # return colored_rgb


    ### orig
    # image_depth = torch.from_numpy(image_depth).float().to(comfy.model_management.get_torch_device())
    # image_depth = image_depth / 255.0
    # image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
    # depth = self.model.infer(image_depth)

    # image_depth = image_depth / 255.0
    # image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
    # depth = self.model.infer(image_depth)

    # depth = depth[0, 0].cpu().numpy()

    # vmin = np.percentile(depth, 2)
    # vmax = np.percentile(depth, 85)

    # depth -= vmin
    # depth /= vmax - vmin
    # depth = 1.0 - depth
    # depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8)

    # return depth_image

class ZoeDetector:
    def __init__(self):
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/ZoeD_M12_N.pt"
        modelpath = os.path.join(annotator_ckpts_path, "ZoeD_M12_N.pt")
        if not os.path.exists(modelpath):
            load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)
        conf = get_config("zoedepth", "infer")
        model = ZoeDepth.build_from_config(conf)
        model.load_state_dict(torch.load(modelpath, map_location=comfy.model_management.get_torch_device())['model'])
        model.eval()
        self.model = model.to(comfy.model_management.get_torch_device())

    def __call__(self, input_image):
        assert input_image.ndim == 3
        image_depth = input_image
        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).float().to(comfy.model_management.get_torch_device())
            image_depth = image_depth / 255.0
            image_depth = rearrange(image_depth, 'h w c -> 1 c h w')

            depth = self.model.infer(image_depth)

            return depth

# def save_raw_16bit(depth, fpath="raw.png"):
#     if isinstance(depth, torch.Tensor):
#         depth = depth.squeeze().cpu().numpy()
    
#     assert isinstance(depth, np.ndarray), "Depth must be a torch tensor or numpy array"
#     assert depth.ndim == 2, "Depth must be 2D"
#     depth = depth * 256  # scale for 16-bit png
#     depth = depth.astype(np.uint16)
#     depth = Image.fromarray(depth)
#     depth.save(fpath)
#     print("Saved raw depth to", fpath)


from custom_nodes.comfy_controlnet_preprocessors.nodes.util import common_annotator_call, img_np_to_tensor, img_tensor_to_np
from custom_nodes.comfy_controlnet_preprocessors.util import resize_image, HWC3

class RL_Zoe_Depth_Map_Preprocessor_Raw_Save:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "zoeRaw": ("ZOE_RAW",),
            },
        }

    RETURN_TYPES = ("ZOE_RAW",)
    FUNCTION = "infer"

    CATEGORY = "preprocessors/normal_depth_map"

    def infer(self, image):
        # Ref: https://github.com/lllyasviel/ControlNet-v1-1-nightly/blob/main/gradio_depth.py
        tensor_image_list = img_tensor_to_np(image)
        annotator_callback = ZoeDetector()
        out_list = []
        for tensor_image in tensor_image_list:
            call_result_raw = annotator_callback(resize_image(HWC3(tensor_image)))
            H, W, C = tensor_image.shape
            out_list.append((call_result_raw, H, W, C,))
            
        return (out_list,)
    
class RL_Zoe_Depth_Map_Preprocessor_Raw_Infer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("ZOE_RAW","IMAGE",)
    FUNCTION = "infer"

    CATEGORY = "preprocessors/normal_depth_map"

    def infer(self, image):
        # Ref: https://github.com/lllyasviel/ControlNet-v1-1-nightly/blob/main/gradio_depth.py
        tensor_image_list = img_tensor_to_np(image)
        annotator_callback = ZoeDetector()
        out_list = []
        byte_images = []
        for tensor_image in tensor_image_list:
            call_result_raw = annotator_callback(resize_image(HWC3(tensor_image)))

            H, W, C = tensor_image.shape
            out_list.append((call_result_raw, H, W, C,))

            # print(f'RL_Zoe_Depth_Map_Preprocessor_Raw_Infer {call_result_raw.shape} {call_result_raw} {call_result_raw*256} {call_result_raw*256*256} {call_result_raw*256*256*256}')
            byte0 = ((call_result_raw)%256)//1
            byte1 = ((call_result_raw*256)%256)//1
            byte2 = ((call_result_raw*256*256)%256)//1
            byte3 = ((call_result_raw*256*256*256)%256)//1
            print(f'RL_Zoe_Depth_Map_Preprocessor_Raw_Infer {call_result_raw.shape} {byte0} {byte1} {byte2} {byte3}')

            byte_images.append(byte0.squeeze(dim=0)/255)
            byte_images.append(byte1.squeeze(dim=0)/255)
            byte_images.append(byte2.squeeze(dim=0)/255)
            byte_images.append(byte3.squeeze(dim=0)/255)
            
        return (out_list,torch.cat(byte_images, dim=0).cpu(),)
    
class RL_Zoe_Depth_Map_Preprocessor_Raw_Process:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "zoeRaw": ("ZOE_RAW",),
                "normMin": ("INT", {
                    "default": 2, 
                    "min": 0,
                    "max": 100,
                    "step": 1               
                }),
                "normMax": ("INT", {
                    "default": 85, 
                    "min": 0,
                    "max": 100,
                    "step": 1                   
                }),
            },
            "optional": {
                "cutoffMid": ("FLOAT", {
                    "default": 0.5, 
                    "step": 0.001,
                }),
                "cutoffRadius": ("FLOAT", {
                    "default": 0.001, 
                    "step": 0.001,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate_depth"

    CATEGORY = "preprocessors/normal_depth_map"

    def estimate_depth(self, zoeRaw, normMin, normMax, cutoffMid = None, cutoffRadius = None):
        # Ref: https://github.com/lllyasviel/ControlNet-v1-1-nightly/blob/main/gradio_depth.py
        out_list = []
        for (call_result_raw, H, W, C,) in zoeRaw:
            call_result = processZoeDept(call_result_raw, normMin, normMax, 
                                         None if cutoffMid is None or cutoffRadius is None else cutoffMid-cutoffRadius, 
                                         None if cutoffMid is None or cutoffRadius is None else cutoffMid+cutoffRadius)

            # resized = call_result

            # resized = cv2.resize(HWC3(call_result), (W, H), interpolation=cv2.INTER_AREA)
            # resized = cv2.resize(call_result, (W, H), interpolation=cv2.INTER_AREA)

            out_list.append(torch.from_numpy(call_result.astype(np.float32) / 255.0))
            
        return torch.stack(out_list)
    

class RL_Zoe_Depth_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "normMin": ("INT", {
                    "default": 2, 
                    "min": 0,
                    "max": 100,
                    "step": 1               
                }),
                "normMax": ("INT", {
                    "default": 85, 
                    "min": 0,
                    "max": 100,
                    "step": 1                   
                }),
            },
            "optional": {
                "cutoffMin": ("FLOAT", {
                    "default": 0, 
                    "step": 0.000001,
                }),
                "cutoffMax": ("FLOAT", {
                    "default": 255, 
                    "step": 0.000001,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate_depth"

    CATEGORY = "preprocessors/normal_depth_map"

    def estimate_depth(self, image, normMin, normMax, cutoffMin = None, cutoffMax = None):
        # Ref: https://github.com/lllyasviel/ControlNet-v1-1-nightly/blob/main/gradio_depth.py
        tensor_image_list = img_tensor_to_np(image)
        annotator_callback = ZoeDetector()
        out_list = []
        for tensor_image in tensor_image_list:
            call_result_raw = annotator_callback(resize_image(HWC3(tensor_image)))
            call_result = processZoeDept(call_result_raw, normMin, normMax, cutoffMin, cutoffMax)
            H, W, C = tensor_image.shape

            resized = call_result

            # resized = cv2.resize(HWC3(call_result), (W, H), interpolation=cv2.INTER_AREA)
            # resized = cv2.resize(call_result, (W, H), interpolation=cv2.INTER_AREA)

            out_list.append(torch.from_numpy(call_result.astype(np.float32) / 255.0))
            
        return torch.stack(out_list)

    # def estimate_depth(self, image):
    #     # Ref: https://github.com/lllyasviel/ControlNet-v1-1-nightly/blob/main/gradio_depth.py
    #     np_detected_map = common_annotator_call(ZoeDetector(), image)
    #     return (img_np_to_tensor(np_detected_map),)

# def HWC3(x):
#     assert x.dtype == np.uint8
#     if x.ndim == 2:
#         x = x[:, :, None]
#     assert x.ndim == 3
#     H, W, C = x.shape
#     assert C == 1 or C == 3 or C == 4
#     if C == 3:
#         return x
#     if C == 1:
#         return np.concatenate([x, x, x], axis=2)
#     if C == 4:
#         color = x[:, :, 0:3].astype(np.float32)
#         alpha = x[:, :, 3:4].astype(np.float32) / 255.0
#         y = color * alpha + 255.0 * (1.0 - alpha)
#         y = y.clip(0, 255).astype(np.uint8)
#         return y

# def common_annotator_call(annotator_callback, tensor_image, *args):
#     tensor_image_list = img_tensor_to_np(tensor_image)
#     out_list = []
#     out_info_list = []
#     for tensor_image in tensor_image_list:
#         call_result = annotator_callback(resize_image(HWC3(tensor_image)), *args)
#         H, W, C = tensor_image.shape
#         if type(annotator_callback) is openpose_v1.OpenposeDetector:
#             out_list.append(cv2.resize(HWC3(call_result[0]), (W, H), interpolation=cv2.INTER_AREA))
#             out_info_list.append(call_result[1])
#         elif type(annotator_callback) is midas.MidasDetector:
#             out_list.append(cv2.resize(HWC3(call_result[0]), (W, H), interpolation=cv2.INTER_AREA))
#             out_info_list.append(cv2.resize(HWC3(call_result[1]), (W, H), interpolation=cv2.INTER_AREA))
#         else:
#             out_list.append(cv2.resize(HWC3(call_result), (W, H), interpolation=cv2.INTER_AREA))
#     if type(annotator_callback) is openpose_v1.OpenposeDetector:
#         return (out_list, out_info_list)
#     elif type(annotator_callback) is midas.MidasDetector:
#         return (out_list, out_info_list)
#     else:
#         return out_list

# def img_np_to_tensor(img_np_list):
#     out_list = []
#     for img_np in img_np_list:
#         out_list.append(torch.from_numpy(img_np.astype(np.float32) / 255.0))
#     return torch.stack(out_list)

# NODE_CLASS_MAPPINGS = {
#     "RL_Depth16": Zoe_Depth_Map_Preprocessor
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "RL_Depth16": "RL_Depth16"
# }

