from .nodes.crop_resize import RL_Crop_Resize
from .nodes.image_shadow import RL_Image_Shadow

NODE_CLASS_MAPPINGS = {
    "RL_Crop_Resize": RL_Crop_Resize,
    "RL_Image_Shadow": RL_Image_Shadow,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RL_Crop_Resize": "Image Crop and Resize by Mask",
    "RL_Image_Shadow": "Image Shadow",
}
