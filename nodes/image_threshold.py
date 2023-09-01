import torch
import numpy as np

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

class RL_Image_Threshold_Channels:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "operation": (['max', 'average', 'distance'],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_threshold"

    CATEGORY = "ricklove/Image"

    def image_threshold(self, image, threshold=0.5, operation='max'):
        return (np2tensor(self.apply_threshold(tensor2np(image), threshold, operation)), )

    def apply_threshold(self, input_image, threshold=0.5, operation='max'):

        results = []
        for im in input_image:
            t = 255*threshold

            # print('apply_threshold start', im.shape)

            if operation == 'max':
                im = np.max(im, axis=2)
            if operation == 'average':
                im = np.average(im, axis=2)
            if operation == 'distance':
                im = np.sqrt(np.sum(np.square(im, dtype=np.float32), axis=2))

            # print('apply_threshold before threshold', im.shape, im)

            im = np.where(im < t, 0, 255)

            im = np.array([im,im,im]).transpose(1,2,0)
            # print('apply_threshold done', im.shape, im)

            results.append(im)

        return results
