from .nodes.crop_resize import RL_Crop_Resize, RL_Uncrop, RL_BBox
from .nodes.image_shadow import RL_Image_Shadow
from .nodes.optical_flow import RL_Optical_Flow_Dip, RL_Warp_Image
from .nodes.image_threshold import RL_Image_Threshold_Channels
from .nodes.finetune import RL_Finetune_Variable, RL_Finetune_Analyze, RL_Finetune_Analyze_Batch
from .nodes.files import RL_SaveImageSequence, RL_LoadImageSequence, RL_IfFileExists
from .nodes.depth_16 import RL_Zoe_Depth_Map_Preprocessor, RL_Zoe_Depth_Map_Preprocessor_Raw_Infer, RL_Zoe_Depth_Map_Preprocessor_Raw_Process
from .nodes.internet_search import RL_Internet_Search
from .nodes.dependency_order import RL_ForceDependencyOrder, RL_ForceDependencyOrder_Latents, RL_ForceDependencyOrder_Images, RL_ForceDependencyOrder_String, RL_ForceDependencyOrder_Strings, RL_ForceDependencyOrder_ImageString, RL_StopIfBlack, RL_RebootComfyIfLeaky
from .nodes.lora_tools import RL_LoraTextExtractTags
from .nodes.civitai_tools import RL_CivitaiTopImagePrompts

NODE_CLASS_MAPPINGS = {
    "RL_BBox": RL_BBox,
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
    "RL_IfFileExists": RL_IfFileExists,
    "RL_Zoe_Depth_Map_Preprocessor": RL_Zoe_Depth_Map_Preprocessor,
    "RL_Zoe_Depth_Map_Preprocessor_Raw_Infer": RL_Zoe_Depth_Map_Preprocessor_Raw_Infer,
    "RL_Zoe_Depth_Map_Preprocessor_Raw_Process": RL_Zoe_Depth_Map_Preprocessor_Raw_Process,
    "RL_Internet_Search": RL_Internet_Search,
    "RL_ForceDependencyOrder": RL_ForceDependencyOrder,
    "RL_ForceDependencyOrder_Latents": RL_ForceDependencyOrder_Latents,
    "RL_ForceDependencyOrder_Images": RL_ForceDependencyOrder_Images,
    "RL_ForceDependencyOrder_String": RL_ForceDependencyOrder_String,   
    "RL_ForceDependencyOrder_Strings": RL_ForceDependencyOrder_Strings,   
    "RL_ForceDependencyOrder_ImageString": RL_ForceDependencyOrder_ImageString,
    "RL_RebootComfyIfLeaky": RL_RebootComfyIfLeaky,
    "RL_StopIfBlack": RL_StopIfBlack,
    "RL_LoraTextExtractTags": RL_LoraTextExtractTags,
    "RL_CivitaiTopImagePrompts": RL_CivitaiTopImagePrompts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RL_BBox": "RL BBox",
    "RL_Crop_Resize": "RL Image Crop and Resize by Mask",
    "RL_Uncrop": "RL Image Uncrop",
    # "RL_Crop_Resize_Batch": "Image Crop and Resize by Mask (Batch)",
    "RL_Image_Shadow": "RL Image Shadow",
    "RL_Optical_Flow_Dip": "RL Optical Flow (Dip)",
    "RL_Warp_Image": "RL Warp Image with Flow",
    "RL_Image_Threshold_Channels": "RL Image Threshold (Channels)",
    "RL_Finetune_Variable": "RL Finetune Variable",
    "RL_Finetune_Analyze": "RL Finetune Analyze",
    "RL_Finetune_Analyze_Batch": "RL Finetune Analyze Batch",
    "RL_SaveImageSequence": "RL SaveImageSequence",
    "RL_LoadImageSequence": "RL LoadImageSequence",
    "RL_IfFileExists": "RL IfFileExists",
    "RL_Zoe_Depth_Map_Preprocessor": "RL Zoe Depth Map Preprocessor",
    "RL_Zoe_Depth_Map_Preprocessor_Raw_Infer": "RL Zoe Depth Map Preprocessor Raw Infer",
    "RL_Zoe_Depth_Map_Preprocessor_Raw_Process": "RL Zoe Depth Map Preprocessor Raw Process",
    "RL_Internet_Search": "RL Internet Search",
    "RL_ForceDependencyOrder": "RL Force",
    "RL_ForceDependencyOrder_Latents": "RL Force Latents",
    "RL_ForceDependencyOrder_Images": "RL Force Images",
    "RL_ForceDependencyOrder_String": "RL Force String",
    "RL_ForceDependencyOrder_Strings": "RL Force Strings",
    "RL_ForceDependencyOrder_ImageString": "RL Force ImageString",
    "RL_StopIfBlack": "RL StopIfBlack",
    "RL_RebootComfyIfLeaky": "RL RebootComfyIfLeaky",
    "RL_LoraTextExtractTags": "RL LoraTextExtractTags",
    "RL_CivitaiTopImagePrompts": "RL CivitaiTopImagePrompts",
}
