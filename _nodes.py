from .nodes.crop_resize import RL_Crop_Resize

NODE_CLASS_MAPPINGS = {
    "RL_Crop_Resize": RL_Crop_Resize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RL_Crop_Resize": "Image Crop and Resize by Mask",
}
