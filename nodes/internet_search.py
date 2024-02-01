import torch
import numpy as np
from duckduckgo_search import DDGS
from duckduckgo_search.cli import _download_file
from PIL import Image, ImageSequence
import requests

def pil2tensor(image):
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)

    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class RL_Internet_Search:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "search": ("STRING", {"default": "cat"}),
                "type_image": (["photo", "clipart", "gif", "transparent", "line"],),
                "safesearch": (['on', 'moderate', 'off'],),
                "max_results": ("INT", {"default": 1}),
                "skip_results": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "search"

    CATEGORY = "ricklove/internet"

    def search(self, search, type_image = None, safesearch = 'on', max_results = 1, skip_results=0):

        result = []
        ddgs_images_gen = DDGS().images(search, None, safesearch, None, None, None, type_image, None, None, max_results + skip_results)
        iSkip = 0
        for r in ddgs_images_gen:
            if iSkip < skip_results:
                iSkip = iSkip + 1
                continue
            
            url = r["image"]

            im = Image.open(requests.get(url, stream=True).raw)
            # frames = []
            for frame in ImageSequence.Iterator(im):
                # frame = np.array(frame.copy().convert('RGB').getdata(), dtype=np.uint8).reshape(frame.size[1],frame.size[0], 3)
                # frames.append(frame)
                result.append(frame.copy().convert('RGB'))

            # if type_image == 'gif':
            #     im.seek(1)
            #     try:
            #         while 1:
            #             im.seek(im.tell() + 1)
            #             result.append(im.copy())
            #             # do something to im
            #     except EOFError:
            #         pass  # end of sequence
            # else:
            #     result.append(im)

            print(f'RL_Internet_Search {im} {url}')

        return (pil2tensor(result),)
    
