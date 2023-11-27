import torch
import numpy as np
import random

class RL_Finetune_Variable:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "currentIndex": ("INT", {"forceInput": True,}),
                "startIndex": ("INT", {"forceInput": True,}),
                "outLabel": ("STRING", {"forceInput": True,}),

                "valueName": ("STRING", {"multiline": False,}),
                "stepCount": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1,}),
                "minValue": ("FLOAT", {"default": 0, "min": -1000, "max": 1000, "step": 0.01,}),
                "maxValue": ("FLOAT", {"default": 0, "min": -1000, "max": 1000, "step": 0.01,}),
                "defaultValue": ("FLOAT", {"default": 0, "min": -1000, "max": 1000, "step": 0.01,}),
            },
            "optional": {
                "skip": ("BOOLEAN",{"default": False,}),
                "randomValue": ("BOOLEAN",{"default": False,}),
                # "randomSeed": ("INT", {"default": 42,"forceInput": True,}),
            },
        }

    RETURN_TYPES = (
        "INT",
        "INT",
        "STRING",
        "FLOAT",
        "INT",
        )
    RETURN_NAMES = (
        "currentIndex",
        "nextStartIndex",
        "outLabel",
        "value",
        "valueInt",
        )
    FUNCTION = "finetune"
    CATEGORY = "ricklove/Process"

    def finetune(self, currentIndex, startIndex, outLabel, valueName, stepCount, minValue, maxValue, defaultValue, skip=False, randomValue=False,):

        print(f'finetune: {valueName}', sep='\n')

        if skip:
            ratio = (defaultValue - minValue) / (maxValue - minValue)
            bar = '█' * int(ratio * 10) + '-' * int((1-ratio) * 10)
            v = defaultValue
            nextStartIndex = startIndex
            symbol = '-'
            return (currentIndex, nextStartIndex, 
                f'{outLabel}\n{symbol} {bar} ({ratio:6.2f}) {v:6.2f} {valueName}', v, int(v))
        
        if randomValue:
            # random.seed(randomSeed)
            randomRange = random.gauss(0.5,0.25)
            ratio = randomRange
            bar = '█' * int(ratio * 10) + '-' * int((1-ratio) * 10)
            v = minValue + (maxValue-minValue)*(ratio)
            nextStartIndex = startIndex + stepCount
            symbol = '~'
            return (currentIndex, nextStartIndex, 
                f'{outLabel}\n{symbol} {bar} ({ratio:6.2f}) {v:6.2f} {valueName}', v, int(v))

        nextStartIndex = startIndex + stepCount
        if currentIndex < startIndex or currentIndex >= nextStartIndex:
            ratio = (defaultValue - minValue) / (maxValue - minValue)
            bar = '█' * int(ratio * 10) + '-' * int((1-ratio) * 10)
            v = defaultValue
            symbol = ' '
            return (currentIndex, nextStartIndex, 
                f'{outLabel}\n{symbol} {bar} ({ratio:6.2f}) {v:6.2f} {valueName}', v, int(v))
        

        i = currentIndex - startIndex
        ratio = 0 if stepCount < 2 else (i*1.0/(stepCount-1))
        bar = '█' * int(ratio * 10) + '-' * int((1-ratio) * 10)
        v = minValue + (maxValue-minValue)*ratio
        nextStartIndex = startIndex + stepCount
        symbol = '*'
        return (currentIndex, nextStartIndex, 
            f'{outLabel}\n{symbol} {bar} ({ratio:6.2f}) {v:6.2f} {valueName}', v, int(v))



class RL_Finetune_Analyze:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = (
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "FLOAT",
        )
    RETURN_NAMES = (
        "ave",
        "min",
        "max",
        "median",
        )
    FUNCTION = "analyze"
    CATEGORY = "ricklove/Process"

    def analyze(self, image,):

        # to numpy
        image = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

        ave = np.average(image)
        min = np.min(image)
        max = np.max(image)
        median = np.median(image)

        return (ave,min,max,median,)

        # image = image.convert('L')
        # hist = image.histogram()

        # sum = 0
        # count = 0
        # for i in range(256):
        #     sum += hist[i] * i
        #     count += hist[i]
        
        # ave_val = sum / count


class RL_Finetune_Analyze_Batch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "imagesMain": ("IMAGE",),
                "imagesCompare": ("IMAGE",),
                "direction": (["both", "reduce", "expand"], {"default": "both"}),
            },
        }

    RETURN_TYPES = (
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "INT",
        "INT",
        "STRING",
        )
    RETURN_NAMES = (
        "score",
        "consistency",
        "difference",
        "imageMainCount",
        "imageCompareCount",
        "summary",
        )
    FUNCTION = "analyze"
    CATEGORY = "ricklove/Process"

    def analyze(self, imagesMain, imagesCompare, direction,):

        imgMainCount = imagesMain.shape[0]
        imgCompCount = imagesCompare.shape[0]

        print(f'''RL_Finetune_Analyze_Batch: shapes 
            {imagesMain.shape} 
            {imagesCompare.shape}''')

        tensor_diff = None
        tensor_consistency = None
        last_frame_diff = None
        for i in range(imgCompCount):
            xMain = imagesMain[0] if i >= imgMainCount else imagesMain[i]
            xComp = imagesCompare[i]

            frameDiff = None
            if direction == 'both':
                frameDiff = torch.abs(xMain-xComp) / imgCompCount  
            if direction == 'reduce':              
                frameDiff = (xMain-xComp) / imgCompCount 
            else:
                frameDiff = (xComp-xMain) / imgCompCount

            frameCons = 0 if last_frame_diff is None else torch.abs(frameDiff - last_frame_diff)
            last_frame_diff = frameDiff

            tensor_consistency = frameCons if tensor_consistency is None else tensor_consistency + frameCons
            tensor_diff = frameDiff if tensor_diff is None else tensor_diff + frameDiff


        # to numpy
        np_diff = np.clip(255.0 * tensor_diff.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        np_consistency = np.clip(255.0 * tensor_consistency.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

        ave_diff = np.average(np_diff)
        ave_consistency = np.average(np_consistency)

        # score = ave_diff if imgCount <= 1 else ave_diff / (ave_consistency+0.01)
        score = ave_diff - ave_consistency

        sBar = min(int(score / 50.0 * 50), 50)
        sBarDec = int(score / 1.00 * 50) % 50
        
        bar = '█' * sBar + '-' * (50-sBar)
        barDec = '▲' * sBarDec + ' ' * (50-sBarDec)
        summary = f'{score:.6f} = {ave_diff:.6f} - {ave_consistency:.6f}\n|{bar}|\n|{barDec}|'

        return (score,ave_consistency,ave_diff,imgMainCount,imgCompCount,summary,)

        # image = image.convert('L')
        # hist = image.histogram()

        # sum = 0
        # count = 0
        # for i in range(256):
        #     sum += hist[i] * i
        #     count += hist[i]
        
        # ave_val = sum / count
