import torch
import numpy as np
from PIL import Image
import skimage
from skimage.color import (convert_colorspace)
from skimage import exposure, img_as_float, img_as_ubyte
from skimage.morphology import disk
from skimage.filters import rank

# from custom_nodes.comfy_mtb.utils import (pil2tensor, np2tensor, tensor2np)
def pil2tensor(image):
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)

    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def np2tensor(img_np):
    if isinstance(img_np, list):
        return torch.cat([np2tensor(img) for img in img_np], dim=0)

    return torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0)

def tensor2np(tensor: torch.Tensor) -> [np.ndarray]:
    batch_count = 1
    if len(tensor.shape) > 3:
        batch_count = tensor.size(0)
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2np(tensor[i]))
        return out

    return [np.clip(255.0 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)]



def clamp(value, minvalue, maxvalue):
    return max(minvalue, min(value, maxvalue))

class RL_Image_Shadow:
    def __init__(self):
        return
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "IMAGE",

        "IMAGE",
        "IMAGE",
        "IMAGE",

        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",

        "IMAGE",
        "IMAGE",
        "IMAGE",
        )
    RETURN_NAMES = (
        "img_all",
        "img_orig",
        "img_orig_all",

        "img_intensity",
        "img_gamma",
        "img_log",

        "img_rescale",
        "out_shadows",
        "out_highlights",
        "out_mid",

        "img_eq",
        "img_adaptive",
        "img_eq_local",
        )
    FUNCTION = "create_shadow_image"
    
    CATEGORY = "ricklove/image"
    
    def create_shadow_image(self, image):

        imgs = tensor2np(image)
        out_orig = []
        out_orig_all = []

        out_intensity = []
        out_gamma = []
        out_log = []

        out_rescale = []
        out_shadows = []
        out_highlights = []
        out_mid = []

        out_eq = []
        out_adaptive = []
        out_eq_local = []

        out_all = []

        for img in imgs:
            out_orig.append(img)
            out_intensity.append(exposure.rescale_intensity(img))
            out_gamma.append(exposure.adjust_gamma(img, 2))
            out_log.append(exposure.adjust_log(img, 1))

            # Contrast stretching
            v_min, v_max = np.percentile(img, (0.2, 99.8))
            out_rescale.append(exposure.rescale_intensity(img, in_range=(v_min, v_max)))

            v_min, v_max = np.percentile(img, (0, 30))
            out_shadows.append(exposure.rescale_intensity(img, in_range=(v_min, v_max)))

            v_min, v_max = np.percentile(img, (70, 100))
            out_highlights.append(exposure.rescale_intensity(img, in_range=(v_min, v_max)))
            v_min, v_max = np.percentile(img, (30,70))
            out_mid.append(exposure.rescale_intensity(img, in_range=(v_min, v_max)))

            # Global equalize
            out_eq.append(img_as_ubyte(exposure.equalize_hist(img)))

            # Adaptive Equalization
            out_adaptive.append(img_as_ubyte(exposure.equalize_adapthist(img, clip_limit=0.03)))

            # Equalization
            out_eq_local.append(img)
            # out_eq_local.append(rank.equalize(img, footprint=disk(30)))

        out_all = ([]
            + out_orig

            + out_intensity
            + out_gamma
            + out_log

            + out_rescale
            + out_shadows
            + out_highlights
            + out_mid

            + out_eq
            + out_adaptive
            + out_eq_local
        )

        out_orig_all = ([]
            + out_orig

            + out_orig
            + out_orig
            + out_orig

            + out_orig
            + out_orig
            + out_orig
            + out_orig

            + out_orig
            + out_orig
            + out_orig
        )
        

        return (
            np2tensor(out_all),
            np2tensor(out_orig),
            np2tensor(out_orig_all),

            np2tensor(out_intensity),
            np2tensor(out_gamma),
            np2tensor(out_log),

            np2tensor(out_rescale),
            np2tensor(out_shadows),
            np2tensor(out_highlights),
            np2tensor(out_mid),

            np2tensor(out_eq),
            np2tensor(out_adaptive),
            np2tensor(out_eq_local),
        )

        # hsv = convert_colorspace(img, 'RGB', 'HSV')
        # hsv[np.logical_not(True)] = []



        # image_pil = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

        # image_pil.convert('')  

        # [w,h] = mask_pil.size

        # (l,t,r,b) = mask_pil.getbbox()
        # l = clamp(l - padding, 0, w)
        # t = clamp(t - padding, 0, h)
        # r = clamp(r + padding, 0, w)
        # b = clamp(b + padding, 0, h)

        # w_pad = r-l
        # h_pad = b-t

        # wr = max_side_length / w_pad
        # hr = max_side_length / h_pad
        # ratio = min(wr,hr)

        # w_resized = int(w_pad * ratio / 8) * 8
        # h_resized = int(h_pad * ratio / 8) * 8

        # w_source = round(w_resized / ratio)
        # h_source = round(h_resized / ratio)

        # r_source = l + w_source
        # b_source = t + h_source

        # mask_cropped = mask_pil.crop((l,t,r_source,b_source))
        # mask_resized = mask_cropped.resize((w_resized, h_resized), Image.Resampling.LANCZOS)

        # image_cropped = image_pil.crop((l,t,r_source,b_source))
        # image_resized = image_cropped.resize((w_resized, h_resized), Image.Resampling.LANCZOS)

        # # mask_pil.crop(bbox)

        # mask_tensor = pil2tensor(mask_resized.convert("L"))
        # image_tensor = pil2tensor(image_resized)

        # # # return 1.0 - mask
        # # region_mask, crop_data = self.WT.Masking.crop_region(mask_pil, region_type, padding)
        # # region_tensor = pil2mask(ImageOps.invert(region_mask)).unsqueeze(0).unsqueeze(1)
        
        # bbox_source = (t, l, r_source, b_source)

        # return (image_tensor, mask_tensor, bbox_source, t, l, r_source, b_source, w_source, h_source, w_resized, h_resized)
        