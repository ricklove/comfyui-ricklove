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

from typing import Union, List

# Based on mtb
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

def pil2tensor(image: Image.Image | List[Image.Image]) -> torch.Tensor:
    if isinstance(image, list):
        items = [pil2tensor(img) for img in image]
        for item in items:
            print(f'pil2tensor: item {item.shape}')
        return torch.cat(items, dim=0)

    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

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
            "optional": {
                "count": (
                    "INT",
                    {"default": 1, "min": 1, "max": 9999999},
                ),
                "select_every_nth": (
                    "INT",
                    {"default": 1, "min": 1, "max": 9999999},
                ),
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
        # count is ignored since it will get the length from the images
        count=1, 
        select_every_nth=1,
        prompt=None,
        extra_pnginfo=None,
    ):

        images = tensor2pil(images)
        for i,img in enumerate(images):
            c = i * select_every_nth + current_frame
            resolved_path = resolve_path(path, c)
            resolved_img = Path(self.output_dir) / resolved_path
            resolved_dir = resolved_img.parent
            resolved_dir.mkdir(parents=True, exist_ok=True)

            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            img.save(resolved_img, pnginfo=metadata, compress_level=4)
            
        return ()
        # return {
        #     "ui": {
        #         "images": [
        #             {
        #                 "filename": resolved_img.name,
        #                 "subfolder": resolved_dir.name,
        #                 "type": self.type,
        #             }
        #         ]
        #     }
        # }


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
            },
            "optional": {
                "count": (
                    "INT",
                    {"default": 1, "min": 1, "max": 9999999},
                ),
                "select_every_nth": (
                    "INT",
                    {"default": 1, "min": 1, "max": 9999999},
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
    # OUTPUT_NODE = True

    def load_image(self, path, current_frame=0, count=1, select_every_nth=1):
        print(f"Loading image: {path}, {current_frame}")

        if(count > 1):
            images = []
            masks = []
            for i in range(0, count):
                c = i * select_every_nth + current_frame
                resolved_path = resolve_path(path, c)
                image_path = folder_paths.get_annotated_filepath(resolved_path)
                if not os.path.exists(image_path):
                    continue
                image, mask = img_from_path(image_path)
                images.append(image)
                masks.append(mask)
                
            return (torch.cat(images, dim=0), torch.cat(masks, dim=0), current_frame)

        resolved_path = resolve_path(path, current_frame)
        image_path = folder_paths.get_annotated_filepath(resolved_path)
        image, mask = img_from_path(image_path)
        return (
            image,
            mask,
            current_frame,
        )

    # @classmethod
    # def IS_CHANGED(self, path='', current_frame=0, count=1, select_every_nth=1):
    #     print(f"Checking if changed: {path}, {current_frame}")
    #     resolved_path = resolve_path(path, current_frame)
    #     image_path = folder_paths.get_annotated_filepath(resolved_path)
    #     m = hashlib.sha256()
    #     with open(image_path, 'rb') as f:
    #         m.update(f.read())
    #     return m.digest().hex()

    #     # resolved_path = resolve_path(path, current_frame)
    #     # image_path = folder_paths.get_annotated_filepath(resolved_path)
    #     # if os.path.exists(image_path):
    #     #     m = hashlib.sha256()
    #     #     with open(image_path, "rb") as f:
    #     #         m.update(f.read())
    #     #     return m.digest().hex()
    #     # return "NONE"


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


class RL_IfFileExists:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        
        return {
                    "required":
                    {
                        "path": ("STRING", {"default": "videos/#####.png"}),
                        "true_value": ("INT", {"default": 1, },),
                        "false_value": ("INT", {"default": 2, },),
                    },
                    "optional": {
                        "current_frame": (
                            "INT",
                            {"default": 0, "min": 0, "max": 9999999},
                        ),
                    },
                }


    CATEGORY = "ricklove/IO"
    FUNCTION = "check_file"
    RETURN_TYPES = (
        "BOOLEAN",
        "INT",
    )
    RETURN_NAMES = (
        "exists",
        "value",
    )

    def check_file(self, path, true_value, false_value, current_frame=0):
        resolved_path = resolve_path(path, current_frame)
        image_path = folder_paths.get_annotated_filepath(resolved_path)
        print(f"Checking if file exists: {resolved_path}")
        if os.path.exists(image_path):
            return (True, true_value)
        else:
            return (False, false_value)
        

class RL_CacheImageSequence:
    @classmethod
    def INPUT_TYPES(s):
        inputs = {            
            "required": {
                "images": ("IMAGE", {"lazy": True}),
                "path": ("STRING", {"default": "videos/#####.png"}),
                "current_frame": ("INT", {"default": 0, "min": 0, "max": 9999999}),
            },
            "optional": {
                "count": (
                    "INT",
                    {"default": 1, "min": 1, "max": 9999999},
                ),
                "select_every_nth": (
                    "INT",
                    {"default": 1, "min": 1, "max": 9999999},
                ),
                "subpath": ("STRING", {"default": ""}),
                "force": ("BOOLEAN",{"default": False,}),
            },
        }

        return inputs

    CATEGORY = "ricklove/IO"
    FUNCTION = "doit"
    RETURN_TYPES = (
        "IMAGE",
    )
    RETURN_NAMES = (
        "images",
    )

    def check_lazy_status(self, 
        images,
        path,
        current_frame=0,
        count=1, 
        select_every_nth=1,
        force=False,
        subpath="",
        ):

        if subpath != "":
            path = path.replace("/#", f'/{subpath}/#')

        resolved_path = resolve_path(path, current_frame)
        image_path = folder_paths.get_annotated_filepath(resolved_path)
        if not force and os.path.exists(image_path):
            print(f"File already exists: {resolved_path}")
            return []
        else:
            print(f"File does not exists, need images: {resolved_path}")
            return ['images']

    @staticmethod
    def doit(
        images,
        path,
        current_frame=0,
        count=1, 
        select_every_nth=1,
        force=False,
        subpath="",
        ):

        if subpath != "":
            path = path.replace("/#", f'/{subpath}/#')

        resolved_path = resolve_path(path, current_frame)
        image_path = folder_paths.get_annotated_filepath(resolved_path)
        if not force and os.path.exists(image_path):
            return RL_LoadImageSequence().load_image(path, current_frame, count, select_every_nth)

        RL_SaveImageSequence().save_images(images, path, current_frame, count, select_every_nth)
        return (images,)


class RL_CacheMaskSequence:
    @classmethod
    def INPUT_TYPES(s):
        inputs = {            
            "required": {
                "masks": ("MASK", {"lazy": True}),
                "path": ("STRING", {"default": "videos/#####.png"}),
                "current_frame": ("INT", {"default": 0, "min": 0, "max": 9999999}),
            },
            "optional": {
                "count": (
                    "INT",
                    {"default": 1, "min": 1, "max": 9999999},
                ),
                "select_every_nth": (
                    "INT",
                    {"default": 1, "min": 1, "max": 9999999},
                ),
                "subpath": ("STRING", {"default": ""}),
                "force": ("BOOLEAN",{"default": False,}),
            },
        }

        return inputs

    CATEGORY = "ricklove/IO"
    FUNCTION = "doit"
    RETURN_TYPES = (
        "MASK",
    )
    RETURN_NAMES = (
        "masks",
    )

    def check_lazy_status(self, 
        masks,
        path,
        current_frame=0,
        count=1, 
        select_every_nth=1,
        force=False,
        subpath="",
        ):

        result = RL_CacheImageSequence().check_lazy_status(masks, path, current_frame, count, select_every_nth, force, subpath)
        if len(result) == 0:
            return []
        return ['masks']

    @staticmethod
    def doit(
        masks,
        path,
        current_frame=0,
        count=1, 
        select_every_nth=1,
        force=False,
        subpath="",
        ):

        images = None
        if masks is not None:
            mask_pil = tensor2pil(masks.unsqueeze(-1))
            images_pil = [mask.convert("RGB") for mask in mask_pil]
            images = pil2tensor(images_pil)

        result = RL_CacheImageSequence().doit(images, path, current_frame, count, select_every_nth, force, subpath)
        images = result[0]
        masks_pil = tensor2pil(images)
        masks_pil = [mask.convert("L") for mask in masks_pil]
        return (pil2tensor(masks_pil),)
        

class RL_SequenceContext:
    def __init__(self):
        return
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {

            },
            "optional": {
                "ctx": ("SEQ_CONTEXT",),
                "path": ("STRING", {"forceInput": True}),
                "current_frame": ("INT", {"forceInput": True}),
                "count": ("INT", {"forceInput": True}),
                "select_every_nth": ("INT", {"forceInput": True}),
            },
        }
    
    RETURN_TYPES = (
        "SEQ_CONTEXT", 
        "STRING",
        "INT",
        "INT",
        "INT",
       )
    RETURN_NAMES = (
        "ctx",
        "path",
        "current_frame",
        "count",
        "select_every_nth",
        )
    FUNCTION = "doit"
    
    CATEGORY = "ricklove/image"
    
    def doit(self, 
        ctx = None,
        path = '',
        current_frame = -1,
        count = -1,
        select_every_nth = -1,
        ):
        
        _path = ""
        _current_frame = 0
        _count = 1
        _select_every_nth = 1

        if ctx is not None:
            (_path,_current_frame,_count,_select_every_nth) = ctx
        
        if path != '':
            _path = path
        
        if current_frame > -1:
            _current_frame = current_frame
        
        if count > -1:
            _count = count
        
        if select_every_nth > -1:
            _select_every_nth = select_every_nth
            
        return ((_path,_current_frame,_count,_select_every_nth), _path,_current_frame,_count,_select_every_nth)

class RL_Sequence_ToFilePathList:
    def __init__(self):
        return
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ctx": ("SEQ_CONTEXT",),
            },
             "optional": {
                "subpath": ("STRING", {"default": ""}),
                "seq_index": ("INT", {"default": 0, "min": 0}),
                "seq_count": ("INT", {"default": 1, "min": 1}),
            },
        }
    
    RETURN_TYPES = (
        "STRING",
       )
    RETURN_NAMES = (
        "path",
        )
    FUNCTION = "doit"
    
    CATEGORY = "ricklove/image"
    
    def doit(self, 
        ctx,
        subpath = "",
        seq_index = 0,
        seq_count = 0,
        ):
        
        (path, current_frame, count, select_every_nth) = ctx
        if subpath != "":
            path = path.replace("/#", f'/{subpath}/#')

        paths = []
        for i in range(0, count):
            c = i * select_every_nth + current_frame
            resolved_path = resolve_path(path, c)
            paths.append(resolved_path)

        if seq_index > 0:
            paths = paths[seq_index:]

        if seq_count <= 1:
            return (paths[0],)
        
        paths = paths[:seq_count]

        return (paths,)
    
