import os
import re
import torch
import numpy as np
import hashlib
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
import folder_paths
from pathlib import Path
import json

# Based on mtb
class RL_SaveImageSequence:
    """Save an image sequence to a folder. The current frame is used to determine which image to save.

    This is merely a wrapper around the `save_images` function with formatting for the output folder and filename.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "path": ("STRING", {"default": "videos/#####.png"}),
                "current_frame": ("INT", {"default": 0, "min": 0, "max": 9999999}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "ricklove/IO"

    def save_images(
        self,
        images,
        path,
        current_frame=0,
        prompt=None,
        extra_pnginfo=None,
    ):

        if len(images) > 1:
            raise ValueError("Can only save one image at a time")

        resolved_img = Path(self.output_dir) / resolve_path(path, current_frame)
        resolved_dir = resolved_img.parent
        resolved_dir.mkdir(parents=True, exist_ok=True)

        output_image = images[0].cpu().numpy()
        img = Image.fromarray(np.clip(output_image * 255.0, 0, 255).astype(np.uint8))
        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))

        img.save(resolved_img, pnginfo=metadata, compress_level=4)
        return {
            "ui": {
                "images": [
                    {
                        "filename": resolved_img.name,
                        "subfolder": resolved_dir.name,
                        "type": self.type,
                    }
                ]
            }
        }


class RL_LoadImageSequence:
    """Load an image sequence from a folder. The current frame is used to determine which image to load.

    This will also check if the file exists and will always execute as if it is output
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "videos/#####.png"}),
                "current_frame": (
                    "INT",
                    {"default": 0, "min": 0, "max": 9999999},
                ),
            }
        }

    CATEGORY = "ricklove/IO"
    FUNCTION = "load_image"
    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "INT",
    )
    RETURN_NAMES = (
        "image",
        "mask",
        "current_frame",
    )
    OUTPUT_NODE = True

    def load_image(self, path, current_frame=0):
        print(f"Loading image: {path}, {current_frame}")
        resolved_path = resolve_path(path, current_frame)
        image_path = folder_paths.get_annotated_filepath(resolved_path)
        image, mask = img_from_path(image_path)
        return (
            image,
            mask,
            current_frame,
        )

    @staticmethod
    def IS_CHANGED(path="", current_frame=0):
        print(f"Checking if changed: {path}, {current_frame}")
        resolved_path = resolve_path(path, current_frame)
        image_path = folder_paths.get_annotated_filepath(resolved_path)
        if os.path.exists(image_path):
            m = hashlib.sha256()
            with open(image_path, "rb") as f:
                m.update(f.read())
            return m.digest().hex()
        return "NONE"


import glob

def img_from_path(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    image = img.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    if "A" in img.getbands():
        mask = np.array(img.getchannel("A")).astype(np.float32) / 255.0
        mask = 1.0 - torch.from_numpy(mask)
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
    return (
        image,
        mask,
    )

def resolve_path(path, frame):
    hashes = path.count("#")
    padded_number = str(frame).zfill(hashes)
    return re.sub("#+", padded_number, path)
