import cv2
import numpy as np
import os
import sys
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.io import read_video, read_image, ImageReadMode
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from torchvision.io import write_jpeg
import torchvision.transforms as T
from . import berry_utility as utilityb
import tempfile
from pathlib import Path
from urllib.request import urlretrieve
from scipy.interpolate import LinearNDInterpolator
from imageio import imread, imwrite
from torchvision.utils import flow_to_image
import math

device = "cuda" if torch.cuda.is_available() else "cpu"
model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
model = model.eval()

#no clue if this works
def flow_to_rgb(flow):
    """
    Convert optical flow to RGB image
    
    :param flow: optical flow map
    :return: RGB image
    
    """
    # forcing conversion to float32 precision
    flow = flow.numpy()
    hsv = np.zeros(flow.shape, dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    #bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    #cv2.imshow("colored flow", bgr)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return hsv

def write_flo(flow, filename):
    """
    Write optical flow in Middlebury .flo format
    
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    
    from https://github.com/liruoteng/OpticalFlowToolkit/
    
    """
    # forcing conversion to float32 precision
    flow = flow.cpu().data.numpy()
    flow = flow.astype(np.float32)
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = flow.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    flow.tofile(f)
    f.close()


#def infer_old (frameA,frameB)

def infer(frameA, frameB):
    # print('infer: loading raft model')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model = model.eval()

    # print('infer: running')

    # Check if both frames have the same size
    if frameA.size != frameB.size:
        raise ValueError("Both input frames must have the same size")

    transform = T.ToTensor()

    img1_batch = transform(frameA)
    img2_batch = transform(frameB)
    img1_batch = torch.stack([img1_batch])
    img2_batch = torch.stack([img2_batch])
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    def preprocess(img1_batch, img2_batch):
        return transforms(img1_batch, img2_batch)

    img1_batch, img2_batch = preprocess(img1_batch, img2_batch)

    # print('infer: DONE')

    return img1_batch, img2_batch
 


def apply_flow_based_on_images (image1_path, image2_path, provided_image_path, output_path):
    output_dir = f'{Path(output_path).parent}'
    output_filename = Path(output_path).name
    print('warped_image_dir', output_dir, output_filename)

    print(f'apply_flow_based_on_images: START\n    {image1_path}\n->  {image2_path}')

    def calculate_flows():
        img1_texture = utilityb.base64_to_texture(image1_path)
        img2_texture = utilityb.base64_to_texture(image2_path)
        # max_dimension = max(img1_texture.shape) * scale
        max_dimension = max(img1_texture.shape)
        while max_dimension > 1024:
            max_dimension = int(max_dimension / 2)

        print('apply_flow_based_on_images: max_dimension', max_dimension, img1_texture.shape, img2_texture.shape)

        w,h = get_target_size(img1_texture, max_dimension)
        w =  int(w / 8) * 8
        h =  int(h / 8) * 8
        image1 =  resize_image(img1_texture,h,w)
        h, w = image1.shape[:2]
        image2 =  cv2.resize(img2_texture, (w,h), interpolation=cv2.INTER_LINEAR)

        save_image(image1, os.path.join(output_dir, 'resized', f'{output_filename}.1.png' ))
        save_image(image2, os.path.join(output_dir, 'resized', f'{output_filename}.2.png' ))

        img1_lap = cv2.Laplacian(image1, -1, ksize=3)
        img2_lap = cv2.Laplacian(image1, -1, ksize=3)
        save_image(img1_lap, os.path.join(output_dir, 'img_lap', f'{output_filename}.1.png' ))
        save_image(img2_lap, os.path.join(output_dir, 'img_lap', f'{output_filename}.2.png' ))



        img1_batch,img2_batch = infer(image1,image2)
        img1_batch,img2_batch = img1_batch.to(device), img2_batch.to(device)

        # for masking
        list_of_flows_rev = model(img1_batch, img2_batch)
        # for i, f in enumerate(list_of_flows_rev):
        #     f_img = flow_to_image(f[0]).to("cpu")
        #     write_jpeg(f_img, os.path.join(output_dir, 'w', f'{output_filename}.flow_rev.{i}.png' ))

        predicted_flow_rev = list_of_flows_rev[-1][0]
        flow_img_rev = flow_to_image(predicted_flow_rev).to("cpu")
        predicted_flow_rev = predicted_flow_rev.detach().cpu().numpy()

        # reverse order
        list_of_flows_inv = model(img2_batch, img1_batch)
        # for i, f in enumerate(list_of_flows_inv):
        #     f_img = flow_to_image(f[0]).to("cpu")
        #     write_jpeg(f_img, os.path.join(output_dir, 'w', f'{output_filename}.flow_inv.{i}.png' ))
        predicted_flow_inv = list_of_flows_inv[-1][0]
        flow_inv_img = flow_to_image(predicted_flow_inv).to("cpu")
        predicted_flow_inv = predicted_flow_inv.detach().cpu().numpy()


        # resize
        h2,w2,_ = img1_texture.shape
        pixel_scale = max(img1_texture.shape) / max_dimension
        # print('predicted_flow_rev', predicted_flow_rev.shape)
        #  (2, H, W) => (H, W, 2)
        predicted_flow_rev = np.transpose(predicted_flow_rev, (1,2,0))
        # predicted_flow_rev = cv2.resize(predicted_flow_rev, (int(w/8),int(h/8)), interpolation = cv2.INTER_LANCZOS4)
        predicted_flow_rev = cv2.resize(predicted_flow_rev, (w2,h2), interpolation = cv2.INTER_LANCZOS4)
        # predicted_flow_rev = cv2.GaussianBlur(predicted_flow_rev, (3,3), 0)
        # predicted_flow_rev = cv2.GaussianBlur(predicted_flow_rev, (5,5), 0)
        # predicted_flow_rev = cv2.GaussianBlur(predicted_flow_rev, (7,7), 0)
        predicted_flow_rev = np.transpose(predicted_flow_rev, (2,0,1))
        predicted_flow_rev[0] = pixel_scale * predicted_flow_rev[0]
        predicted_flow_rev[1] = pixel_scale * predicted_flow_rev[1]

        predicted_flow_inv = np.transpose(predicted_flow_inv, (1,2,0))
        # predicted_flow_inv = cv2.resize(predicted_flow_inv, (int(w/8),int(h/8)), interpolation = cv2.INTER_LANCZOS4)
        predicted_flow_inv = cv2.resize(predicted_flow_inv, (w2,h2), interpolation = cv2.INTER_LANCZOS4)
        # predicted_flow_inv = cv2.GaussianBlur(predicted_flow_inv, (3,3), 0)
        # predicted_flow_inv = cv2.GaussianBlur(predicted_flow_inv, (5,5), 0)
        # predicted_flow_inv = cv2.GaussianBlur(predicted_flow_inv, (7,7), 0)
        predicted_flow_inv = np.transpose(predicted_flow_inv, (2,0,1))
        predicted_flow_inv[0] = pixel_scale * predicted_flow_inv[0]
        predicted_flow_inv[1] = pixel_scale * predicted_flow_inv[1]

        return predicted_flow_inv, predicted_flow_rev, flow_inv_img, flow_img_rev


    predicted_flow_inv, predicted_flow_rev, flow_inv_img, flow_img_rev = calculate_flows()

    provided_image = utilityb.base64_to_texture(provided_image_path)
    max_dimension = max(provided_image.shape)
    w,h = get_target_size(provided_image, max_dimension)
    provided_image = cv2.resize(provided_image, (w,h), interpolation=cv2.INTER_LINEAR)

    def apply_flow_to_image_with_unused_mask_inv(image, flow):
        """
        Apply an optical flow tensor to a NumPy image by moving the pixels based on the flow and create a mask where the remap meant there was nothing there.
        
        Args:
            image (np.ndarray): Input image with shape (height, width, channels).
            flow (np.ndarray): Optical flow tensor with shape (height, width, 2).
            
        Returns:
            tuple: Warped image with the same shape as the input image, and a mask where the remap meant there was nothing there.
        """

        print('apply_flow_to_image_with_unused_mask_inv: START')

        height, width, _ = image.shape
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        coords = np.stack([x_coords, y_coords], axis=-1).astype(np.float32)

        # Add the flow to the original coordinates
        if isinstance(flow, torch.Tensor):
            flow = flow.detach().cpu().numpy()

        # --- flow (2, H, W)

        # mask = flow.copy()
        # mask_x, mask_y = mask
        # mask[0] = np.subtract(mask_x, np.average(mask_x))
        # mask[1] = np.subtract(mask_y, np.average(mask_y))
        # white_pixels = np.sum(mask != 0)

        flow = flow.transpose(1, 2, 0)
        # new_coords = np.subtract(coords, flow)
        new_coords = np.add(coords, flow)
        avg = utilityb.avg_edge_pixels(image)
        warped_image = cv2.remap(image, new_coords, None, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        # mask = utilityb.create_hole_mask_inv(flow)
        # white_pixels = np.sum(mask > 0)



        # Create a mask where the remap meant there was nothing there
        # if isinstance(flow_img, torch.Tensor):
        #     flow_img = flow_img.detach().cpu().numpy()
        # mask = Image.fromarray(flow_img).convert('L')

        # Create a mask where the remap meant there was nothing there
        # mask = utilityb.create_hole_mask(flow)
        # white_pixels = np.sum(mask > 0)
        #print(f'white pixels {white_pixels}')

        #remove later
        #warped_image = warp_image2(image,flow)

        # return warped_image, mask, white_pixels
        return warped_image

    warped_image = apply_flow_to_image_with_unused_mask_inv(provided_image,predicted_flow_inv)

    # img1 = utilityb.base64_to_texture(image1_path)
    # id1_img = apply_flow_to_image_with_unused_mask_inv(img1,predicted_flow_inv)
    # id1_img = apply_flow_to_image_with_unused_mask_inv(id1_img,predicted_flow_rev)
    # save_image(id1_img, os.path.join(output_dir, 'w', f'{output_filename}.identity1.png' ))
    # id1_cmp_img = cv2.absdiff(img1, id1_img)
    # save_image(id1_cmp_img, os.path.join(output_dir, 'w', f'{output_filename}.identity1.cmp.png' ))
    # id1_img = apply_flow_to_image_with_unused_mask_inv(id1_img,predicted_flow_inv)
    # id1_img = apply_flow_to_image_with_unused_mask_inv(id1_img,predicted_flow_rev)
    # save_image(id1_img, os.path.join(output_dir, 'w', f'{output_filename}.identity1-2.png' ))
    # id1_cmp_img = cv2.absdiff(img1, id1_img)
    # save_image(cv2.absdiff(img1, id1_img), os.path.join(output_dir, 'w', f'{output_filename}.identity1-2.cmp.png' ))

    # img2 = utilityb.base64_to_texture(image2_path)
    # id2_img = apply_flow_to_image_with_unused_mask_inv(img2,predicted_flow_rev)
    # id2_img = apply_flow_to_image_with_unused_mask_inv(id2_img,predicted_flow_inv)
    # save_image(id2_img, os.path.join(output_dir, 'w', f'{output_filename}.identity2.png' ))
    # id2_cmp_img = cv2.absdiff(img2, id2_img)
    # save_image(id2_cmp_img, os.path.join(output_dir, 'w', f'{output_filename}.identity2.cmp.png' ))
    # id2_img = apply_flow_to_image_with_unused_mask_inv(id2_img,predicted_flow_rev)
    # id2_img = apply_flow_to_image_with_unused_mask_inv(id2_img,predicted_flow_inv)
    # save_image(id2_img, os.path.join(output_dir, 'w', f'{output_filename}.identity2-2.png' ))
    # id2_cmp_img = cv2.absdiff(img2, id2_img)
    # save_image(cv2.absdiff(img2, id2_img), os.path.join(output_dir, 'w', f'{output_filename}.identity2-2-cmp.png' ))

    img1 = utilityb.base64_to_texture(image1_path)
    img2 = utilityb.base64_to_texture(image2_path)

    id1to2_img = apply_flow_to_image_with_unused_mask_inv(img1,predicted_flow_inv)
    save_image(id1to2_img, os.path.join(output_dir, 'identity1to2', f'{output_filename}.png' ))
    id1to2_cmp_img = cv2.absdiff(img2, id1to2_img)
    save_image(id1to2_cmp_img, os.path.join(output_dir, 'identity1to2_cmp', f'{output_filename}.png' ))


    # reverse flow for masking
    predicted_flow_rev = predicted_flow_rev.transpose(1, 2, 0)
    unused_mask_0 = utilityb.create_hole_mask(predicted_flow_rev)

    unused_mask = id1to2_cmp_img.copy()
    unused_mask[unused_mask>=32] = 255
    unused_mask = cv2.cvtColor(unused_mask, cv2.COLOR_BGR2GRAY)
    unused_mask[unused_mask>=32] = 255
    unused_mask[unused_mask!=255] = 0
    white_pixels = np.sum(unused_mask > 0)

    # First create the image with alpha channel
    warped_used_image = warped_image.copy()
    warped_used_image = cv2.cvtColor(warped_used_image, cv2.COLOR_RGB2RGBA)

    # Then assign the mask to the last channel of the image
    used_mask = unused_mask.copy()
    used_mask[used_mask>=128] = 255
    used_mask[used_mask!=255] = 0
    used_mask = 255 - used_mask

    warped_used_image[:, :, 3] = used_mask

    # warped_image_path = os.path.join(output_folder, f'warped_provided_image_{index + 1}.png')

    warped_image_path=output_path
    save_image(warped_used_image, output_path)
    save_image(warped_image, os.path.join(output_dir, 'warped_raw', f'{output_filename}.png' ))
    save_image(unused_mask, os.path.join(output_dir, 'unused_mask', f'{output_filename}.png' ))
    save_image(used_mask, os.path.join(output_dir, 'used_mask', f'{output_filename}.png' ))
    save_flow_image(flow_img_rev, os.path.join(output_dir, 'flow_rev', f'{output_filename}.png' ))
    save_flow_image(flow_inv_img, os.path.join(output_dir, 'flow_inv', f'{output_filename}.png' ))

    print('apply_flow_based_on_images: DONE, saved', warped_image_path)
    return warped_image_path,predicted_flow_inv,unused_mask,white_pixels,flow_inv_img

def apply_flow_to_image(image, flow):
    """
    Apply optical flow transforms to an input image
    
    :param image: input image
    :param flow: optical flow map
    :return: warped image
    
    """
    
    # forcing conversion to float32 precision
    #flow = flow.numpy()
    flow = flow.astype(np.float32)

    # Get the height and width of the input image
    height, width = image.shape[:2]

    # Create a grid of (x, y) coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Apply the optical flow to the coordinates
    x_warped = (x + flow[..., 0]).astype(np.float32)
    y_warped = (y + flow[..., 1]).astype(np.float32)

    # Remap the input image using the warped coordinates
    warped_image = cv2.remap(image, x_warped, y_warped, cv2.INTER_LINEAR)

    return warped_image

def warp_image(image, flow):
    h, w = image.shape[:2]

    flow_map = np.array([[x, y] for y in range(h) for x in range(w)], dtype=np.float32) - flow.reshape(-1, 2)
    flow_map = flow_map.reshape(h, w, 2).astype(np.float32)  # Ensure the flow_map is in the correct format

    # Clip the flow_map to the image bounds
    flow_map[:, :, 0] = np.clip(flow_map[:, :, 0], 0, w - 1)
    flow_map[:, :, 1] = np.clip(flow_map[:, :, 1], 0, h - 1)

    warped_image = cv2.remap(image, flow_map, None, cv2.INTER_LANCZOS4)
    return warped_image

def save_image(image, file_path):
    if not Path(file_path).parent.exists():
        Path(file_path).parent.mkdir(parents=True)
    cv2.imwrite(file_path, image)

def save_flow_image(image, file_path):
    if not Path(file_path).parent.exists():
        Path(file_path).parent.mkdir(parents=True)
    write_jpeg(image, file_path)

def resize_image(image, new_height,new_width):
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image

def get_target_size (image,max_dimension):
    h, w = image.shape[:2]
    aspect_ratio = float(w) / float(h)
    if h > w:
        new_height = max_dimension
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = max_dimension
        new_height = int(new_width / aspect_ratio)
    return new_width,new_height

        
def apply_flow_to_image_try3(image,flow):
    """
    Apply an optical flow tensor to a NumPy image by moving the pixels based on the flow.
    
    Args:
        image (np.ndarray): Input image with shape (height, width, channels).
        flow (np.ndarray): Optical flow tensor with shape (height, width, 2).
        
    Returns:
        np.ndarray: Warped image with the same shape as the input image.
    """
    height, width, _ = image.shape
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    coords = np.stack([x_coords, y_coords], axis=-1).astype(np.float32)

    # Add the flow to the original coordinates
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().numpy()
    flow = flow.transpose(1, 2, 0)
    new_coords = np.subtract(coords, flow)


    # Map the new coordinates to the pixel values in the original image
    warped_image = cv2.remap(image, new_coords, None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return warped_image

# https://stackoverflow.com/a/65205082/567524
def calculate_flow_inverse(flow):
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().numpy()

    print('calculate_flow_inverse: START')

    x_ave = np.average(flow[0])
    y_ave = np.average(flow[1])

    #  (2, H, W) => (H, W, 2)
    # print('calculate_flow_inverse: flow.shape before', flow.shape)
    flow = flow.transpose(1, 2, 0)
    # print('calculate_flow_inverse: flow.shape after', flow.shape)

    height, width, _ = flow.shape
    # print('calculate_flow_inverse: flow', flow)

    # generate the map_pts (for each target pixel, which src_pt should be used)
    # create a flow_inv, [h,w,{x,y, magnitude_so_far}] 
    # iterate through the flow pixels
    # find the dest pixel for that flow pixel (4 pixels interpolation => ratio of application)
    # apply this flow pixel coord based on distance magnitude

    f = np.zeros((height,width,3))
    for y in range(0,height):
        for x in range(0,width):
            # print('x,y',x,y)
            dest = flow[y, x]
            # print(dest)
            x_delta = dest[0]
            y_delta = dest[1]
            cx = x + x_delta
            cy = y + y_delta

            # --- use max movement
            x_mag = x_delta - x_ave
            y_mag = y_delta - y_ave
            mag = math.sqrt(x_mag*x_mag + y_mag*y_mag)

            s = 1
            sf = 1

            for fx in range(-s,s+1):
                fx0 = int(cx+fx)
                fxd = fx0 - cx

                if(fx0<0 or fx0>width-1): continue

                for fy in range(-s,s+1):
                    fy0 = int(cy+fy)
                    fyd = fy0 - cy

                    if(fy0<0 or fy0>height-1): continue

                    fdist = math.sqrt(fxd*fxd+fyd*fyd)
                    if(fdist>sf): continue

                    fv = f[fy0, fx0]
                    fmag = mag - 4 * fdist

                    # fmag = mag * (1-(dx-dx0)) * (1-(dy-dy0))
                    if(fmag>fv[2]):
                        fv[0] = -x_delta
                        fv[1] = -y_delta
                        fv[2] = fmag

            # fv = f[dy1, dx0]
            # # fmag = mag * (1-(dx-dx0)) * (1-(dy1-dy))
            # if(fmag>fv[2]):
            #     fv[0] = -deltax
            #     fv[1] = -deltay
            #     fv[2] = fmag

            # fv = f[dy0, dx1]
            # # fmag = mag * (1-(dx1-dx)) * (1-(dy-dy0))
            # if(fmag>fv[2]):
            #     fv[0] = -deltax
            #     fv[1] = -deltay
            #     fv[2] = fmag

            # fv = f[dy1, dx1]
            # # fmag = mag * (1-(dx1-dx)) * (1-(dy1-dy))
            # if(fmag>fv[2]):
            #     fv[0] = -deltax
            #     fv[1] = -deltay
            #     fv[2] = fmag

            # --- average forces based on distance
            # mag = math.sqrt(deltax*deltax + deltay*deltay)

            # fv = f[dy0, dx0]
            # # print('fv00 before', fv)
            # fmag = mag * (1-(dx-dx0)) * (1-(dy-dy0))
            # fv[0] = (fv[0] * fv[2] + -deltax * fmag) / (fv[2] + fmag + 0.000001)
            # fv[1] = (fv[1] * fv[2] + -deltay * fmag) / (fv[2] + fmag + 0.000001)
            # fv[2] = (fv[2] + fmag)
            # # print('fv00', fv, fmag, (dx, dy), (deltax, deltay), (dx0, dx1, dy0, dy1))

            # fv = f[dy1, dx0]
            # # print('fv01 before', fv)
            # fmag = mag * (1-(dx-dx0)) * (1-(dy1-dy))
            # fv[0] = (fv[0] * fv[2] + -deltax * fmag) / (fv[2] + fmag + 0.000001)
            # fv[1] = (fv[1] * fv[2] + -deltay * fmag) / (fv[2] + fmag + 0.000001)
            # fv[2] = (fv[2] + fmag)
            # # print('fv01', fv, fmag, (dx, dy), (deltax, deltay), (dx0, dx1, dy0, dy1))

            # fv = f[dy0, dx1]
            # # print('fv10 before', fv)
            # fmag = mag * (1-(dx1-dx)) * (1-(dy-dy0))
            # fv[0] = (fv[0] * fv[2] + -deltax * fmag) / (fv[2] + fmag + 0.000001)
            # fv[1] = (fv[1] * fv[2] + -deltay * fmag) / (fv[2] + fmag + 0.000001)
            # fv[2] = (fv[2] + fmag)
            # # print('fv10', fv, fmag, (dx, dy), (deltax, deltay), (dx0, dx1, dy0, dy1))

            # fv = f[dy1, dx1]
            # # print('fv11 before', fv)
            # fmag = mag * (1-(dx1-dx)) * (1-(dy1-dy))
            # fv[0] = (fv[0] * fv[2] + -deltax * fmag) / (fv[2] + fmag + 0.000001)
            # fv[1] = (fv[1] * fv[2] + -deltay * fmag) / (fv[2] + fmag + 0.000001)
            # fv[2] = (fv[2] + fmag)
            # # print('fv11', fv, fmag, (dx, dy), (deltax, deltay), (dx0, dx1, dy0, dy1))

            # break
        # break

    # remove islands
    def remove_islands(size = 1):
        f_orig = f.copy()
        for y in range(size,height-size):
            for x in range(size,width-size):
                if f[y,x][2] <= 0: continue

                count = 0
                for i in range(-size,size+1):
                    for j in range(-size,size+1):
                        fx0 = x+i
                        fy0 = y+j

                        if f_orig[fy0,fx0][2] <= 0: continue
                        count = count + 1
                        if count > 4: break
                    if count > 4: break
                if count > 4: break
                
                f[y,x] = (0,0,0)

    # fill holes
    def fill_holes(size = 4):
        f_orig = f.copy()
        
        for y in range(size,height-size):
            for x in range(size,width-size):
                if f[y,x][2] > 0: continue

                done = False
                for r in range(1, size):
                    for i in range(-r,r+1):
                        for j in range(-r,r+1):
                            if abs(i)+abs(j) != r: continue

                            fx0 = x+i
                            fy0 = y+j

                            if f_orig[fy0,fx0][2] <= 0: continue
                            f[y,x]=f_orig[fy0,fx0]
                            done = True
                            break
                        if done: break
                    if done: break
        return f
    
    # # expand holes
    # def consume_islands(size = 4):
    #     f_orig = f.copy()
        
    #     for y in range(size,height-size):
    #         for x in range(size,width-size):
    #             if f[y,x][2] > 0: continue

    #             empties = False
    #             for i in range(-size,size+1):
    #                 for j in range(-size,size+1):
    #                     if abs(i)+abs(j) != r: continue

    #                     fx0 = x+i
    #                     fy0 = y+j

    #                     if f_orig[fy0,fx0][2] <= 0: continue
    #                     f[y,x]=f_orig[fy0,fx0]
    #                     done = True
    #                     break
    #                 if done: break
    #     return f

    remove_islands()
    # fill_holes(2)
    # remove_islands()
    # fill_holes(2)
    # remove_islands()
    # fill_holes(2)
    # remove_islands()
    # fill_holes(2)
    # fill_holes(2)
    # fill_holes(2)
    # fill_holes(2)
    # fill_holes(2)
    # consume_islands(4)

    print('f', f)
    flow_inv = np.split(f, [2,], 2)[0]
    print('flow_inv', flow_inv)



    # (H, W, 2) => (2, H, W)
    flow_inv = flow_inv.transpose(2, 0, 1)
    flow_inv = torch.from_numpy(flow_inv).float()
    print('flow_inv.shape', flow_inv.shape)
    print('flow_inv', flow_inv)


    # def transform(src_pts, H):
    #     # src = [src_pts 1]
    #     src = np.pad(src_pts, [(0, 0), (0, 1)], constant_values=1)
    #     # pts = H * src
    #     pts = np.dot(H, src.T).T
    #     # normalize and throw z=1
    #     pts = (pts / pts[:, 2].reshape(-1, 1))[:, 0:2]
    #     return pts

    # x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    # coords = np.stack([x_coords, y_coords], axis=-1).astype(np.float32)
    # print('calculate_flow_inverse: coords', coords.shape)
    # print('calculate_flow_inverse: coords', coords)

    # src_pts = coords.reshape(-1, 2)
    # delta_pts = flow.reshape(-1, 2)
    # dst_pts = src_pts+delta_pts
    # print('calculate_flow_inverse: src_pts', src_pts.shape)
    # print('calculate_flow_inverse: src_pts', src_pts)
    # print('calculate_flow_inverse: dst_pts', dst_pts.shape)
    # print('calculate_flow_inverse: dst_pts', dst_pts)

    # H, status = cv2.findHomography(src_pts, dst_pts)
    # print('calculate_flow_inverse: H', H.shape)
    # print('calculate_flow_inverse: H', H)

    # idx_pts = np.mgrid[0:width, 0:height].reshape(2, -1).T
    # print('calculate_flow_inverse: idx_pts', idx_pts.shape)
    # print('calculate_flow_inverse: idx_pts', idx_pts)

    # map_pts = transform(idx_pts, np.linalg.inv(H))
    # map_pts = map_pts.reshape(width, height, 2).astype(np.float32)
    # # warped = cv.remap(img, map_pts, None, cv.INTER_CUBIC).transpose(1, 0, 2)

    # # (W, H, 2) => (2, H, W)
    # flow_inv = map_pts.transpose(2, 1, 0)
    # flow_inv = torch.from_numpy(flow_inv).float()

    # print('calculate_flow_inverse: flow_inv', flow_inv)

    return flow_inv


def apply_flow_to_image_with_unused_mask(image, flow):
    """
    Apply an optical flow tensor to a NumPy image by moving the pixels based on the flow and create a mask where the remap meant there was nothing there.
    
    Args:
        image (np.ndarray): Input image with shape (height, width, channels).
        flow (np.ndarray): Optical flow tensor with shape (height, width, 2).
        
    Returns:
        tuple: Warped image with the same shape as the input image, and a mask where the remap meant there was nothing there.
    """
    height, width, _ = image.shape
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    coords = np.stack([x_coords, y_coords], axis=-1).astype(np.float32)

    # Add the flow to the original coordinates
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().numpy()
    flow = flow.transpose(1, 2, 0)
    new_coords = np.subtract(coords, flow)
    avg = utilityb.avg_edge_pixels(image)
    warped_image = cv2.remap(image, new_coords, None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Create a mask where the remap meant there was nothing there
    mask = utilityb.create_hole_mask(flow)
    white_pixels = np.sum(mask > 0)
    #print(f'white pixels {white_pixels}')

    #remove later
    #warped_image = warp_image2(image,flow)

    return warped_image, mask,white_pixels

def warp_image2(image, flow):
    h, w = image.shape[:2]
    flow_map = np.array([[x, y] for y in range(h) for x in range(w)], dtype=np.float32) - flow.reshape(-1, 2)
    flow_map = flow_map.reshape(h, w, 2).astype(np.float32)  # Ensure the flow_map is in the correct format

    # Clip the flow_map to the image bounds
    flow_map[:, :, 0] = np.clip(flow_map[:, :, 0], 0, w - 1)
    flow_map[:, :, 1] = np.clip(flow_map[:, :, 1], 0, h - 1)

    warped_image = cv2.remap(image, flow_map, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT, borderValue=(0, 0, 0)    )
    return warped_image