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
        "IMAGE",
        "IMAGE",
        "IMAGE",
        )
    RETURN_NAMES = (
        "score",
        "inconsistencyScore",
        "avePixelDifference",
        "countImageMain",
        "countImageCompare",
        "summary",
        "imageDifferences",
        "imageDifferenceAverage",
        "deprecatedImage",
        )
    FUNCTION = "analyze"
    CATEGORY = "ricklove/Process"

    def analyze(self, imagesMain, imagesCompare, direction,):

        imgMainCount = imagesMain.shape[0]
        imgCompCount = imagesCompare.shape[0]

        print(f'''RL_Finetune_Analyze_Batch: shapes 
            {direction}
            {imagesMain.shape} 
            {imagesCompare.shape}''')

        # Get the difference of each pixel to the main
        tensors_diff = []
        for i in range(imgCompCount):
            xMain = imagesMain[0] if i >= imgMainCount else imagesMain[i]
            xComp = imagesCompare[i]

            frameDiff = None
            if direction == 'both':
                frameDiff = 0.5 + 0.5 * torch.abs(xMain-xComp)
            elif direction == 'reduce':
                frameDiff = 0.5 + 0.5 * (xMain-xComp)
            else:
                frameDiff = 0.5 + 0.5 * (xComp-xMain)

            # tensor_diff = (frameDiff / imgCompCount) if tensor_diff is None else tensor_diff + (frameDiff / imgCompCount)
            tensors_diff.append(frameDiff)
        diff_stack = torch.stack(tensors_diff)

        # Calculate the average pixel diff
        ave_diff = 255 * (torch.mean(diff_stack).item() - 0.5)


        inconsistencyScore = 0

        if imgCompCount > 1:
            # TODO: this does not make sense if the main image is changing, since the location of the changes is changing
            # # get the average standard deviation of the pixel diffs
            # stdOfPixelDiffs = (diff_stack-diff_stack) if imgCompCount <= 1 else torch.std(diff_stack, dim=0)
            # inconsistencyScore = 255 * torch.mean(stdOfPixelDiffs).item()

            # Calculate histograms for each image and each channel
            # then get the average std dev of each bin across the images

            # Reshape the stack to (num_images, num_channels, num_pixels)
            reshaped_stack = diff_stack.permute(0, 3, 1, 2).view(diff_stack.shape[0], diff_stack.shape[3], -1)
            # print("Original shape:", diff_stack.shape)
            # print("Reshaped shape:", reshaped_stack.shape)
            
            # Calculate histograms for each image and each channel
            pixelCount = diff_stack.shape[1] * diff_stack.shape[2]
            histogramRatios = []
            for i in range(diff_stack.shape[0]):
                for c in range(diff_stack.shape[3]):
                    histogram = torch.histc(reshaped_stack[i, c], bins=256, min=0, max=1)
                    histogramRatio = histogram / pixelCount
                    histogramRatios.append(histogramRatio)

            # Stack the histograms for (image, channel, bins)
            histogramRatios = torch.stack(histogramRatios, dim=0).view(diff_stack.shape[0], diff_stack.shape[3], 256)
            # print("histograms shape:", histogramRatios.shape)


            stdOfHistogramBins = torch.std(histogramRatios, dim=0)
            inconsistencyScore = 255 * torch.mean(stdOfHistogramBins).item()


        score = ave_diff - inconsistencyScore

        sBar = min(int(score / 50.0 * 50), 50)
        sBarDec = int(score / 1.00 * 50) % 50
        
        bar = '█' * sBar + '-' * (50-sBar)
        barDec = '▲' * sBarDec + ' ' * (50-sBarDec)
        summary = f'{score:.6f} = {ave_diff:.6f} - {inconsistencyScore:.6f}\n|{bar}|\n|{barDec}|'

        result = (score,inconsistencyScore,ave_diff,imgMainCount,imgCompCount,summary,diff_stack,torch.mean(diff_stack, dim=0),imagesMain,)

        # print(f'''RL_Finetune_Analyze_Batch: result 
        #     {result}
        #     ''')
        return result

