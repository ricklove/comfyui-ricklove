import torch
import numpy as np
from duckduckgo_search import DDGS
from duckduckgo_search.cli import _download_file
from PIL import Image, ImageSequence
import requests

from comfy import model_management
import gc

def unload_all_models():
    print(f'unload_all_models START {model_management.get_free_memory()}')
    model_management.unload_all_models()
    gc.collect()
    model_management.soft_empty_cache()
    print(f'unload_all_models DONE {model_management.get_free_memory()}')


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