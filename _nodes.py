from .nodes.crop_resize import RL_Crop_Resize
from .nodes.image_shadow import RL_Image_Shadow
from .nodes.optical_flow import RL_Optical_Flow_Dip, RL_Warp_Image
from .nodes.image_threshold import RL_Image_Threshold_Channels


NODE_CLASS_MAPPINGS = {
    "RL_Crop_Resize": RL_Crop_Resize,
    "RL_Image_Shadow": RL_Image_Shadow,
    "RL_Optical_Flow_Dip": RL_Optical_Flow_Dip,
    "RL_Warp_Image": RL_Warp_Image,
    "RL_Image_Threshold_Channels": RL_Image_Threshold_Channels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RL_Crop_Resize": "Image Crop and Resize by Mask",
    "RL_Image_Shadow": "Image Shadow",
    "RL_Optical_Flow_Dip": "Optical Flow (Dip)",
    "RL_Warp_Image": "Warp Image with Flow",
    "RL_Image_Threshold_Channels": "Image Threshold (Channels)",
}
