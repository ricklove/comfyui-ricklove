import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageFilter
import cv2

# thanks to https://github.com/jinnsp ❤
def base64_to_texture(base64_string):
    if base64_string.lower().endswith('png'):
        image = Image.open(base64_string)
    else:
        decoded_data = base64.b64decode(base64_string)
        buffer = BytesIO(decoded_data)
        image = Image.open(buffer)
    texture = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return texture

def avg_edge_pixels(img):
    height, width = img.shape[:2]
    edge_pixels = []
    
    # top and bottom edges
    edge_pixels.extend(img[0,:])
    edge_pixels.extend(img[height-1,:])
    
    # left and right edges
    edge_pixels.extend(img[:,0])
    edge_pixels.extend(img[:,width-1])
    
    # calculate average of edge pixels
    avg_edge_pixel = np.mean(edge_pixels)
    
    return avg_edge_pixel


def create_hole_mask(flow_map):
    h, w, _ = flow_map.shape
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

    # Compute the new coordinates of each pixel after the optical flow is applied
    new_x_coords = np.clip(x_coords + flow_map[..., 0], 0, w - 1).astype(int)
    new_y_coords = np.clip(y_coords + flow_map[..., 1], 0, h - 1).astype(int)

    # Create a 2D array to keep track of whether a pixel is occupied or not
    occupied = np.zeros((h, w), dtype=bool)

    # Mark the pixels that are occupied after the optical flow is applied
    occupied[new_y_coords, new_x_coords] = True

    # Create the hole mask by marking unoccupied pixels as holes (value of 1)
    hole_mask = np.logical_not(occupied).astype(np.uint8)

    

    expanded = filter_mask(hole_mask) * 255
    #expanded = hole_mask * 255
    #blurred_hole_mask = box_(expanded, sigma=3)
    toblur = Image.fromarray(expanded).convert('L')
    blurred_hole_mask = np.array(toblur.filter(ImageFilter.GaussianBlur(3)))

    #blurred_numpy = np.array( Image.fromarray(expanded).filter(ImageFilter.GaussianBlur(3)))
    #blurred_hole_mask[blurred_hole_mask > 150] = 255
    filtered_smol = filter_mask(hole_mask,4,0.4,0.3) * 255
    return blurred_hole_mask + filtered_smol


# there are pixels all over the place that are not holes, so this only gets the holes with a high concentration
def filter_mask(mask, kernel_size=4, threshold_ratio=0.3,grayscale_intensity=1.0):
    # Create a custom kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Convolve the mask with the kernel
    conv_result = cv2.filter2D(mask, -1, kernel)

    # Calculate the threshold based on the ratio
    threshold = int(kernel.size * threshold_ratio)

    # Filter the mask using the calculated threshold
    filtered_mask = np.where(conv_result >= threshold, mask, 0)

    # thanks to https://github.com/jinnsp ❤
    grayscale_mask = np.where(conv_result >= threshold, int(255 * grayscale_intensity), 0).astype(np.uint8)

    # Combine the filtered mask and grayscale mask
    combined_mask = np.maximum(filtered_mask, grayscale_mask)


    return combined_mask

