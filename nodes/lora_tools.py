import torch
import numpy as np
from duckduckgo_search import DDGS
from duckduckgo_search.cli import _download_file
from PIL import Image, ImageSequence
import requests

from comfy import model_management
import gc

import re


class RL_LoraTextExtractTags:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}), 
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("text_tags", "text_tags_names", "text_no_tags")
    FUNCTION = "run"

    CATEGORY = "ricklove/lora"

    def run(self, text):
        # Regex find all tags of lora form <tag:tag_name>
        # <lora:LoraName:1> <lora:[label] name [more]:1>
        tags = re.findall(r'(<lora:[^>]+>)', text)
        tags = list(set(tags))
        text_tags = '\n'.join(tags)

        tag_names = re.findall(r'<lora:([^>]+):[\d.]+>', text)
        tag_names = list(set(tag_names))
        text_tag_names = '\n'.join(tag_names)

        # Remove tags from text
        text_no_tags = re.sub(r'<lora:[^>]+>', '', text)
        
        return (text_tags, text_tag_names, text_no_tags,)
    