import torch
import numpy as np
from PIL import Image, ImageFilter

from typing import Union, List

def clamp(value, minvalue, maxvalue):
    return max(minvalue, min(value, maxvalue))

# def pil2tensor(image: Image.Image) -> torch.Tensor:
#     return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def pil2tensor(image: Image.Image | List[Image.Image]) -> torch.Tensor:
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)

    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(image: torch.Tensor) -> List[Image.Image]:
    batch_count = 1
    if len(image.shape) > 3:
        batch_count = image.size(0)

    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image[i]))
        return out

    return [
        Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )
    ]

def crop_resize_with_mask(image_pil:Image.Image, mask_pil:Image.Image, padding:int, max_side_length:int, width:int, height:int):
    [w,h] = mask_pil.size

    (l,t,r,b) = mask_pil.getbbox()
    l = clamp(l - padding, 0, w)
    t = clamp(t - padding, 0, h)
    r = clamp(r + padding, 0, w)
    b = clamp(b + padding, 0, h)

    w_pad = r-l
    h_pad = b-t

    wr = max_side_length / w_pad
    hr = max_side_length / h_pad
    ratio = min(wr,hr)

    if width > 0:
        ratio = width / w_pad
    if height > 0:
        ratio = height / h_pad
    if (width > 0) & (height > 0):
        wr = width / w_pad
        hr = height / h_pad
        ratio = min(wr,hr)

    w_resized = int(w_pad * ratio / 32) * 32
    h_resized = int(h_pad * ratio / 32) * 32

    if width > 0:
        w_resized = width
    if height > 0:
        h_resized = height

    w_source = round(w_resized / ratio)
    h_source = round(h_resized / ratio)

    l_source = l
    t_source = t
    r_source = l_source + w_source
    b_source = t_source + h_source

    if r_source > w:
        r_diff = r_source - w
        r_source = w
        l_source = l_source - r_diff
    if b_source > h:
        b_diff = b_source - h
        b_source = h
        t_source = t_source - b_diff

    r_source = r_source - 1 
    b_source = b_source - 1 

    mask_cropped = mask_pil.crop((l_source,t_source,r_source,b_source))
    mask_resized = mask_cropped.resize((w_resized, h_resized), Image.Resampling.LANCZOS)

    image_cropped = image_pil.crop((l_source,t_source,r_source,b_source))
    image_resized = image_cropped.resize((w_resized, h_resized), Image.Resampling.LANCZOS)

    # mask_pil.crop(bbox)

    out_mask = mask_resized.convert("L")
    out_image = image_resized
    # mask_tensor = pil2tensor(mask_resized.convert("L"))
    # image_tensor = pil2tensor(image_resized)

    # # return 1.0 - mask
    # region_mask, crop_data = self.WT.Masking.crop_region(mask_pil, region_type, padding)
    # region_tensor = pil2mask(ImageOps.invert(region_mask)).unsqueeze(0).unsqueeze(1)
    
    bbox_source = (t_source, l_source, r_source, b_source)

    return (out_image, out_mask, bbox_source, t_source, l_source, r_source, b_source, w_source, h_source, w_resized, h_resized)

def uncrop_image(original_image_pil:Image.Image, cropped_image_pil:Image.Image, t_source:int, l_source:int, r_source:int, b_source:int, blend:float):
    # cropped_image_pil.
    print("original_image_pil.size", original_image_pil.size)
    print("cropped_image_pil.size", cropped_image_pil.size)
    print("original_image_pil.mode", original_image_pil.mode)
    print("cropped_image_pil.mode", cropped_image_pil.mode)

    w = r_source-l_source+1
    h = b_source-t_source+1

    cropped_image_pil = cropped_image_pil.resize((w,h), Image.Resampling.LANCZOS)
    paste_image = Image.new('RGB', original_image_pil.size, (0,0,0))
    paste_image.paste(cropped_image_pil, [l_source, t_source])

    # mask
    RADIUS = int(min(blend * w/2,blend * h/2))
    mask = Image.new('L', (w,h), 0)
    mask.paste(Image.new('L', (w-2*RADIUS,h-2*RADIUS), 255), (RADIUS,RADIUS))
    mask = mask.filter(ImageFilter.GaussianBlur(RADIUS/2))
    paste_mask = Image.new('L', original_image_pil.size, 0)
    paste_mask.paste(mask, [l_source, t_source])

    original_image_pil.paste(paste_image, paste_mask)
    return (original_image_pil,paste_mask)


class RL_Crop_Resize:
    def __init__(self):
        return
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "padding": ("INT",{"default": 128, "min": 0, "max": 4096, "step": 1}),
                "max_side_length": ("INT",{"default": 512, "min": 0, "max": 4096, "step": 1}),
            },
            "optional": {
                "width": ("INT",{"default": 0, "min": 0, "max": 4096, "step": 1}),
                "height": ("INT",{"default": 0, "min": 0, "max": 4096, "step": 1}),
                "skip": ("BOOLEAN",{"default": False,}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "BBOX", "INT", "INT", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "bbox_source", "top_source", "left_source", "right_source", "bottom_source", "width_source", "height_source", "width_result", "height_result")
    FUNCTION = "crop_resize"
    
    CATEGORY = "ricklove/image"
    
    def crop_resize(self, image, mask, padding=24, max_side_length=512, width=0, height=0, skip=False):

        mask_pil = tensor2pil(mask)[0]

        [w,h] = mask_pil.size

        if skip:
            return (image, mask, (0, 0, w, h), 0, 0, w, h, w, h, w, h)
        
        images_pil = tensor2pil(image)
        out_images = []
        out_masks = []
        
        for image_pil in images_pil:
            (out_image, out_mask, bbox_source, t_source, l_source, r_source, b_source, w_source, h_source, w_resized, h_resized) = crop_resize_with_mask(image_pil, mask_pil, padding, max_side_length, width, height)
            out_images.append(out_image)
            out_masks.append(out_mask)
        
        return (pil2tensor(out_images), pil2tensor(out_masks[0]), bbox_source, t_source, l_source, r_source, b_source, w_source, h_source, w_resized, h_resized)

class RL_Uncrop:
    def __init__(self):
        return
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "cropped_image": ("IMAGE",),
                "box": ("BBOX",)
            },
            "optional": {
                "blend": ("FLOAT",{"default": 0.10, "min": 0, "max": 1, "step": 0.01}),
            },
        }
    
    RETURN_TYPES = ("IMAGE","IMAGE",)
    RETURN_NAMES = ("image","blend_mask",)
    FUNCTION = "uncrop_image"
    
    CATEGORY = "ricklove/image"
    
    def uncrop_image(self, image, cropped_image, box, blend=0.0):

        images_pil = tensor2pil(image)
        cropped_images_pil = tensor2pil(cropped_image)
        out_images = []
        out_blend_masks = []
        
        for i,image_pil in enumerate(images_pil):
            cropped_image_pil = cropped_images_pil[i]
            (t,l,r,b) = box
            (out_image, blend_mask) = uncrop_image(image_pil, cropped_image_pil, t,l,r,b, blend)
            out_images.append(out_image)
            out_blend_masks.append(blend_mask)
        
        return (pil2tensor(out_images),pil2tensor(out_blend_masks),)


# class RL_Crop_Resize_Batch:
#     def __init__(self):
#         return
        
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "images": ("IMAGE",),
#                 "masks": ("MASK",),
#                 "padding": ("INT",{"default": 128, "min": 0, "max": 4096, "step": 1}),
#                 "max_side_length": ("INT",{"default": 512, "min": 0, "max": 4096, "step": 1}),
#             },
#             "optional": {
#                 "width": ("INT",{"default": 0, "min": 0, "max": 4096, "step": 1}),
#                 "height": ("INT",{"default": 0, "min": 0, "max": 4096, "step": 1}),
#                 # "skip": ("BOOLEAN",{"default": False,}),
#             },
#         }

#     RETURN_TYPES = ("IMAGE", "MASK", "BBOXLIST", "BBOX")
#     RETURN_NAMES = ("cropped_images", "cropped_masks", "bbox_sources", "bbox_source_0")
#     FUNCTION = "crop_resize"

#     CATEGORY = "ricklove/image"

#     def crop_resize(self, image, mask, padding=24, max_side_length=512, width=0, height=0, skip=False):

#         images = tensor2pil(image)
#         masks = tensor2pil(mask)

#         out_images = []
#         out_masks = []
#         out_boxes = []

#         for i,image_pil in enumerate(images):

#             mask_pil = masks[i]
#             result = resize_image(image_pil, mask_pil, padding, max_side_length, width, height)
#             (image_tensor, mask_tensor, bbox_source) = result[0]
#             out_images.append(image_tensor)
#             out_masks.append(mask_tensor)
#             out_boxes.append(bbox_source)

#         # o=out_results[0] 
#         return (
#             torch.cat(out_images, dim=0),
#             torch.cat(out_masks, dim=0),
#             out_boxes,
#         )
        
