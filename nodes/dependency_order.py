import torch
import numpy as np
from duckduckgo_search import DDGS
from duckduckgo_search.cli import _download_file
from PIL import Image, ImageSequence
import requests

from comfy import model_management
import gc
from server import PromptServer

from .crop_resize import tensor2pil
from nodes import interrupt_processing

max_free_memory = 0
leaked_old = 0

def unload_all_models():
    global max_free_memory, leaked_old
    print(f'\n\n---UNLOADER---\n')
    print(f'unload_all_models START {model_management.get_free_memory()} max_free_memory:{max_free_memory}')
    model_management.unload_all_models()
    model_management.soft_empty_cache(True)
    try:
        print("unload_all_models - empty torch cache")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except:
        print("unload_all_models - empty torch cache FAILED")
    free_memory = model_management.get_free_memory()
    leaked_actual = max_free_memory - free_memory
    leaked_new = leaked_actual - leaked_old
    leaked_old = leaked_actual
    leaked_message = f' LEAKED:{leaked_new}' if leaked_new > 0 else ''
    print(f'unload_all_models DONE {free_memory} max_free_memory:{max_free_memory}{leaked_message}')
    max_free_memory = max(max_free_memory, free_memory)
    print(f'\n\n')
    return leaked_actual



# class RL_ForceDependencyOrder2:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "optional": {
#                 "image": ("IMAGE", {}),
#                 "mask": ("MASK", {}),
#                 "boolean_value": ("BOOLEAN", {"forceInput": True}),
#                 "int_value": ("INT", {"forceInput": True}),
#                 "float_value": ("FLOAT", {"forceInput": True}),
#                 "string_value": ("STRING", {"forceInput": True}),
#                 "a": ("*", {}),
#                 "b": ("*", {}),
#                 "c": ("*", {}),
#             },
#         }

#     RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN", "INT", "FLOAT", "STRING", "*", "*", "*",)
#     RETURN_NAMES = ("image", "mask", "boolean_value", "int_value", "float_value", "string_value", "a", "b", "c",)
#     FUNCTION = "passthrough"

#     CATEGORY = "ricklove/dependencies"

#     def passthrough(self, image, mask, boolean_value, int_value, float_value, string_value, a, b, c):

#         return (image, image, mask, boolean_value, int_value, float_value, string_value, a, b, c,)
    

# class RL_ForceDependencyOrder3:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "optional": {
#                 "image": ("IMAGE", {}),
#                 "mask": ("MASK", {}),
#                 "latent": ("LATENT", {}),
#                 "a": ("*", {}),
#                 "b": ("*", {}),
#                 "c": ("*", {}),
#             },
#         }

#     RETURN_TYPES = ("IMAGE", "MASK", "LATENT", "*", "*", "*",)
#     RETURN_NAMES = ("image", "mask", "latent", "a", "b", "c",)
#     FUNCTION = "passthrough"

#     CATEGORY = "ricklove/dependencies"

#     def passthrough(self, image=None, mask=None, latent=None, a=None, b=None, c=None):

#         return (image, image, mask, latent, a, b, c,)
    

class RL_ForceDependencyOrder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "unload_models": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "a": ("*", {}),
                "b": ("*", {}),
                "c": ("*", {}),
                "d": ("*", {}),
                "e": ("*", {}),
                "f": ("*", {}),
            },
        }

    RETURN_TYPES = ("*", "*", "*", "*", "*", "*",)
    RETURN_NAMES = ("a", "b", "c", "d", "e", "f",)
    FUNCTION = "passthrough"

    CATEGORY = "ricklove/dependencies"

    def passthrough(self, unload_models=True, a=None, b=None, c=None, d=None, e=None, f=None):
        if unload_models: 
            unload_all_models()
        return (a, b, c, d, e, f)
    
class RL_ForceDependencyOrder_Latents:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "unload_models": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "a": ("LATENT", {}),
                "b": ("LATENT", {}),
                "c": ("LATENT", {}),
                "d": ("LATENT", {}),
                "e": ("LATENT", {}),
                "f": ("LATENT", {}),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "LATENT", "LATENT", "LATENT", "LATENT",)
    RETURN_NAMES = ("a", "b", "c", "d", "e", "f",)
    FUNCTION = "passthrough"

    CATEGORY = "ricklove/dependencies"

    def passthrough(self, unload_models=True, a=None, b=None, c=None, d=None, e=None, f=None):
        if unload_models: 
            unload_all_models()
        return (a, b, c, d, e, f)
    
class RL_ForceDependencyOrder_Images:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "unload_models": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "a": ("IMAGE", {}),
                "b": ("IMAGE", {}),
                "c": ("IMAGE", {}),
                "d": ("IMAGE", {}),
                "e": ("IMAGE", {}),
                "f": ("IMAGE", {}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("a", "b", "c", "d", "e", "f",)
    FUNCTION = "passthrough"

    CATEGORY = "ricklove/dependencies"

    def passthrough(self, unload_models=True, a=None, b=None, c=None, d=None, e=None, f=None):
        if unload_models: 
            unload_all_models()
        return (a, b, c, d, e, f)
    
class RL_ForceDependencyOrder_String:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "unload_models": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "a": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("a", )
    FUNCTION = "passthrough"

    CATEGORY = "ricklove/dependencies"

    def passthrough(self, unload_models=True, a=None, ):
        if unload_models: 
            unload_all_models()
        return (a, )
    
class RL_ForceDependencyOrder_Strings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "unload_models": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "a": ("STRING", {"forceInput": True}),
                "b": ("STRING", {"forceInput": True}),
                "c": ("STRING", {"forceInput": True}),
                "d": ("STRING", {"forceInput": True}),
                "e": ("STRING", {"forceInput": True}),
                "f": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("a", "b", "c", "d", "e", "f",)
    FUNCTION = "passthrough"

    CATEGORY = "ricklove/dependencies"

    def passthrough(self, unload_models=True, a=None, b=None, c=None, d=None, e=None, f=None):
        if unload_models: 
            unload_all_models()
        return (a, b, c, d, e, f)

class RL_ForceDependencyOrder_ImageString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "unload_models": ("BOOLEAN", {"default": True}),
                "a": ("IMAGE", {}),
                "b": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("a", "b",)
    FUNCTION = "passthrough"

    CATEGORY = "ricklove/dependencies"

    def passthrough(self, unload_models=True, a=None, b=None):
        if unload_models: 
            unload_all_models()
        return (a, b,)
    

class RL_StopIfBlack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { "image": ("IMAGE",), },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "passthrough"
    RETURN_NAMES = ("image",)
    CATEGORY = "ricklove/dependencies"

    OUTPUT_NODE = True

    def passthrough(self, image):
        image_pils = tensor2pil(image)
        is_black_image = np.all(np.array(image_pils) == 0)
        print(f"RL_StopIfBlack: is_black_image: {is_black_image}")
        if is_black_image:
            print(f"\n\n---STOP!!!---\nRL_StopIfBlack: is_black_image: {is_black_image}\n---STOP!!!---\n\n")
            interrupt_processing()

        return (image,)
    

# import restart from '../../ComfyUI-Manager/glob/manager_util.py'
# code taken from ComfyUI-Manager/glob/manager_util.py
import os
import sys
def restart_comfy():
    try:
        sys.stdout.close_log()
    except Exception as e:
        pass

    if '__COMFY_CLI_SESSION__' in os.environ:
        with open(os.path.join(os.environ['__COMFY_CLI_SESSION__'] + '.reboot'), 'w') as file:
            pass

        print(f"\nRestarting...\n\n")
        exit(0)

    print(f"\nRestarting... [Legacy Mode]\n\n")
    if sys.platform.startswith('win32'):
        return os.execv(sys.executable, ['"' + sys.executable + '"', '"' + sys.argv[0] + '"'] + sys.argv[1:])
    else:
        return os.execv(sys.executable, [sys.executable] + sys.argv)

class RL_RebootComfyIfLeaky:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "image": ("IMAGE",),
                "max_leaked_gb": ("FLOAT", {"default": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "passthrough"
    RETURN_NAMES = ("image",)
    CATEGORY = "ricklove/dependencies"

    OUTPUT_NODE = True

    def passthrough(self, image, max_leaked_gb):
        leaked = unload_all_models()
        leaked_gb = leaked / 1024 / 1024 / 1024

        if leaked_gb > max_leaked_gb:
            print(f"\n\n---ABANDON SHIP!!!---\nRL_RebootComfyIfLeaky: leaked_gb: {leaked_gb}\n---ABANDON SHIP!!!---\n\n")
            restart_comfy()
            

        return (image,)