import torch
import numpy as np
from PIL import Image

def clamp(value, minvalue, maxvalue):
    return max(minvalue, min(value, maxvalue))

def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

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
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "BBOX", "INT", "INT", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "bbox_source", "top_source", "left_source", "right_source", "bottom_source", "width_source", "height_source", "width_result", "height_result")
    FUNCTION = "crop_resize"
    
    CATEGORY = "ricklove/image"
    
    def crop_resize(self, image, mask, padding=24, max_side_length=512, width=0, height=0):

        mask_pil = Image.fromarray(np.clip(255. * mask.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        image_pil = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

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


        mask_cropped = mask_pil.crop((l_source,t_source,r_source,b_source))
        mask_resized = mask_cropped.resize((w_resized, h_resized), Image.Resampling.LANCZOS)

        image_cropped = image_pil.crop((l_source,t_source,r_source,b_source))
        image_resized = image_cropped.resize((w_resized, h_resized), Image.Resampling.LANCZOS)

        # mask_pil.crop(bbox)

        mask_tensor = pil2tensor(mask_resized.convert("L"))
        image_tensor = pil2tensor(image_resized)

        # # return 1.0 - mask
        # region_mask, crop_data = self.WT.Masking.crop_region(mask_pil, region_type, padding)
        # region_tensor = pil2mask(ImageOps.invert(region_mask)).unsqueeze(0).unsqueeze(1)
        
        bbox_source = (t_source, l_source, r_source, b_source)

        return (image_tensor, mask_tensor, bbox_source, t_source, l_source, r_source, b_source, w_source, h_source, w_resized, h_resized)
        