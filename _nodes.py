from .nodes.crop_resize import RL_Crop_Resize, RL_Uncrop
from .nodes.image_shadow import RL_Image_Shadow
from .nodes.optical_flow import RL_Optical_Flow_Dip, RL_Warp_Image
from .nodes.image_threshold import RL_Image_Threshold_Channels
from .nodes.finetune import RL_Finetune_Variable, RL_Finetune_Analyze, RL_Finetune_Analyze_Batch
from .nodes.files import RL_SaveImageSequence, RL_LoadImageSequence
from .nodes.depth_16 import RL_Zoe_Depth_Map_Preprocessor, RL_Zoe_Depth_Map_Preprocessor_Raw_Infer, RL_Zoe_Depth_Map_Preprocessor_Raw_Process
from .nodes.internet_search import RL_Internet_Search



NODE_CLASS_MAPPINGS = {
    "RL_Crop_Resize": RL_Crop_Resize,
    "RL_Uncrop": RL_Uncrop,
    # "RL_Crop_Resize_Batch": RL_Crop_Resize_Batch,
    "RL_Image_Shadow": RL_Image_Shadow,
    "RL_Optical_Flow_Dip": RL_Optical_Flow_Dip,
    "RL_Warp_Image": RL_Warp_Image,
    "RL_Image_Threshold_Channels": RL_Image_Threshold_Channels,
    "RL_Finetune_Variable": RL_Finetune_Variable,
    "RL_Finetune_Analyze": RL_Finetune_Analyze,
    "RL_Finetune_Analyze_Batch": RL_Finetune_Analyze_Batch,
    "RL_SaveImageSequence": RL_SaveImageSequence,
    "RL_LoadImageSequence": RL_LoadImageSequence,
    "RL_Zoe_Depth_Map_Preprocessor": RL_Zoe_Depth_Map_Preprocessor,
    "RL_Zoe_Depth_Map_Preprocessor_Raw_Infer": RL_Zoe_Depth_Map_Preprocessor_Raw_Infer,
    "RL_Zoe_Depth_Map_Preprocessor_Raw_Process": RL_Zoe_Depth_Map_Preprocessor_Raw_Process,
    "RL_Internet_Search": RL_Internet_Search,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RL_Crop_Resize": "Image Crop and Resize by Mask",
    "RL_Uncrop": "Image Uncrop",
    # "RL_Crop_Resize_Batch": "Image Crop and Resize by Mask (Batch)",
    "RL_Image_Shadow": "Image Shadow",
    "RL_Optical_Flow_Dip": "Optical Flow (Dip)",
    "RL_Warp_Image": "Warp Image with Flow",
    "RL_Image_Threshold_Channels": "Image Threshold (Channels)",
    "RL_Finetune_Variable": "Finetune Variable",
    "RL_Finetune_Analyze": "Finetune Analyze",
    "RL_Finetune_Analyze_Batch": "Finetune Analyze Batch",
    "RL_SaveImageSequence": "RL_SaveImageSequence",
    "RL_LoadImageSequence": "RL_LoadImageSequence",
    "RL_Zoe_Depth_Map_Preprocessor": "RL_Zoe_Depth_Map_Preprocessor",
    "RL_Zoe_Depth_Map_Preprocessor_Raw_Infer": "RL_Zoe_Depth_Map_Preprocessor_Raw_Infer",
    "RL_Zoe_Depth_Map_Preprocessor_Raw_Process": "RL_Zoe_Depth_Map_Preprocessor_Raw_Process",
    "RL_Internet_Search": "RL_Internet_Search",
}
