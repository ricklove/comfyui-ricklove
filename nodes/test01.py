import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import measure, feature, exposure, segmentation, color, graph, img_as_float
from skimage.util import random_noise
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.morphology import convex_hull_image
from skimage.util import invert
from skimage.feature import hog
from skimage.feature import peak_local_max
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)
from skimage.color import label2rgb
from skimage.filters import sobel
from skimage.measure import label
from skimage.segmentation import expand_labels, watershed
from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage import data, io
from skimage.filters import rank
from skimage.util import img_as_ubyte
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)

import cv2 as cv

from PIL import Image
import numpy as np

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    img_arr = np.array(img) #are all you need
    # print(img_arr.shape)
    # plt.imshow(img_arr)
    # plt.show()
    # data = np.asarray( img, dtype="int32" )
    return img_arr

def find_contour_ex():
    img = data.astronaut()
    img = rgb2gray(img)

    s = np.linspace(0, 2*np.pi, 400)
    r = 100 + 100*np.sin(s)
    c = 220 + 100*np.cos(s)
    init = np.array([r, c]).T

    snake = active_contour(gaussian(img, 3, preserve_range=False),
                        init, alpha=0.015, beta=10, gamma=0.001)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])

    plt.show()

def find_countour( img ):
    r = rgb2gray(img)

    # # Construct some test data
    # x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
    # r = np.sin(np.exp(np.sin(x)**3 + np.cos(y)**2))

    # Find contours at a constant value of 0.8
    contours = measure.find_contours(r, 0.34)

    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(r, cmap=plt.cm.gray)

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

def find_canny( image ):
    image = rgb2gray(image)

    # Compute the Canny filter for two values of sigma
    edges1 = feature.canny(image)
    edges2 = feature.canny(image, sigma=3)

    # display results
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('noisy image', fontsize=20)

    ax[1].imshow(edges1, cmap='gray')
    ax[1].set_title(r'Canny filter, $\sigma=1$', fontsize=20)

    ax[2].imshow(edges2, cmap='gray')
    ax[2].set_title(r'Canny filter, $\sigma=3$', fontsize=20)

    for a in ax:
        a.axis('off')

    fig.tight_layout()
    plt.show()

def find_active_contour(img, img_mask):
    img = rgb2gray(img)
    # img_mask = rgb2gray(img_mask)

    # chull = convex_hull_image(img_mask)
    img_mask_cv = img_mask.astype(np.uint8)
    img_mask_cv = cv.cvtColor(img_mask_cv, cv.COLOR_BGR2GRAY)
    # plt.imshow(img_mask_cv)
    # plt.show()

    contours, _ = cv.findContours(img_mask_cv, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    # img_contours = img.copy()
    # cv.drawContours(img_contours, contours, -1, (0,255,0), 1)
    # plt.imshow(img_contours)
    # plt.show()

    # print(contours)


    # chull = cv.convexHull(np.vstack(contours))
    # init = chull.reshape(chull.shape[0], 2)
    # print(np.shape(init))

    # # print(contours)
    # # print(np.shape(contours[0]))
    contours_arr = np.asarray(contours[0])
    print(contours_arr)
    print(contours_arr.shape)
    init = contours_arr.reshape(contours_arr.shape[0], 2)

    # print(init)
    # # print(np.shape(init))


    # plt.imshow(im2)
    # plt.show()

    # s = np.linspace(0, 2*np.pi, 400)
    # r = x + w*np.sin(s)
    # c = y + h*np.cos(s)
    # init = np.array([r, c]).T

    img_blur = img
    # img_blur = gaussian(img, 3, preserve_range=False)
    # plt.imshow(img_blur)
    # plt.show()

    snake = active_contour(img_blur, init, alpha=0.0015, beta=10, gamma=0.001)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])

    plt.show()

def find_convex_hull(img):
    # The original image is inverted as the object must be white.
    image = invert(img)

    chull = convex_hull_image(image)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax = axes.ravel()

    ax[0].set_title('Original picture')
    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_axis_off()

    ax[1].set_title('Transformed picture')
    ax[1].imshow(chull, cmap=plt.cm.gray)
    ax[1].set_axis_off()

    plt.tight_layout()
    plt.show()

def find_oriented_gradients(image):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, channel_axis=-1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()

def rag_threshold(img):

    labels1 = segmentation.slic(img, compactness=30, n_segments=400, start_label=1)
    out1 = color.label2rgb(labels1, img, kind='avg', bg_label=0)

    g = graph.rag_mean_color(img, labels1)
    labels2 = graph.cut_threshold(labels1, g, 29)
    out2 = color.label2rgb(labels2, img, kind='avg', bg_label=0)

    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True,
                        figsize=(6, 8))

    ax[0].imshow(out1)
    ax[1].imshow(out2)

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()

def find_threshold(image):
    image = rgb2gray(image)

    thresh = threshold_otsu(image)
    binary = image > thresh

    fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
    ax = axes.ravel()
    ax[0] = plt.subplot(1, 3, 1)
    ax[1] = plt.subplot(1, 3, 2)
    ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Original')
    ax[0].axis('off')

    ax[1].hist(image.ravel(), bins=256)
    ax[1].set_title('Histogram')
    ax[1].axvline(thresh, color='r')

    ax[2].imshow(binary, cmap=plt.cm.gray)
    ax[2].set_title('Thresholded')
    ax[2].axis('off')

    plt.show()

def find_local_max(img):
    img = rgb2gray(img)
    im = img_as_float(img)

    # image_max is the dilation of im with a 20*20 structuring element
    # It is used within peak_local_max function
    image_max = ndi.maximum_filter(im, size=20, mode='constant')

    # Comparison between image_max and im to find the coordinates of local maxima
    coordinates = peak_local_max(im, min_distance=20)

    # display results
    fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(im, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('Original')

    ax[1].imshow(image_max, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('Maximum filter')

    ax[2].imshow(im, cmap=plt.cm.gray)
    ax[2].autoscale(False)
    ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    ax[2].axis('off')
    ax[2].set_title('Peak local max')

    fig.tight_layout()

    plt.show()

def find_local_threshold(image):
    image = rgb2gray(image)

    matplotlib.rcParams['font.size'] = 9

    binary_global = image > threshold_otsu(image)

    window_size = 25
    thresh_niblack = threshold_niblack(image, window_size=window_size, k=0.8)
    thresh_sauvola = threshold_sauvola(image, window_size=window_size)

    binary_niblack = image > thresh_niblack
    binary_sauvola = image > thresh_sauvola

    plt.figure(figsize=(8, 7))
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title('Global Threshold')
    plt.imshow(binary_global, cmap=plt.cm.gray)
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(binary_niblack, cmap=plt.cm.gray)
    plt.title('Niblack Threshold')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(binary_sauvola, cmap=plt.cm.gray)
    plt.title('Sauvola Threshold')
    plt.axis('off')

    plt.show()

# def find_random_walker_seg(img):
#     rng = np.random.default_rng()

#     # Generate noisy synthetic data
#     data = skimage.img_as_float(binary_blobs(length=128, rng=1))
#     sigma = 0.35
#     data += rng.normal(loc=0, scale=sigma, size=data.shape)
#     data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
#                             out_range=(-1, 1))

#     # The range of the binary image spans over (-1, 1).
#     # We choose the hottest and the coldest pixels as markers.
#     markers = np.zeros(data.shape, dtype=np.uint)
#     markers[data < -0.95] = 1
#     markers[data > 0.95] = 2

#     # Run random walker algorithm
#     labels = random_walker(data, markers, beta=10, mode='bf')

#     # Plot results
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2),
#                                         sharex=True, sharey=True)
#     ax1.imshow(data, cmap='gray')
#     ax1.axis('off')
#     ax1.set_title('Noisy data')
#     ax2.imshow(markers, cmap='magma')
#     ax2.axis('off')
#     ax2.set_title('Markers')
#     ax3.imshow(labels, cmap='gray')
#     ax3.axis('off')
#     ax3.set_title('Segmentation')

#     fig.tight_layout()
#     plt.show()

def find_expanded_segmentation(img):
    img = rgb2gray(img)

    coins = img
    
    # Make segmentation using edge-detection and watershed.
    edges = sobel(coins)

    # Identify some background and foreground pixels from the intensity values.
    # These pixels are used as seeds for watershed.
    markers = np.zeros_like(coins)
    foreground, background = 1, 2
    markers[coins < 30.0] = background
    markers[coins > 150.0] = foreground

    ws = watershed(edges, markers)
    seg1 = label(ws == foreground)

    expanded = expand_labels(seg1, distance=10)

    # Show the segmentations.
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(9, 5),
        sharex=True,
        sharey=True,
    )

    axes[0].imshow(coins, cmap="Greys_r")
    axes[0].set_title("Original")

    color1 = label2rgb(seg1, image=coins, bg_label=0)
    axes[1].imshow(color1)
    axes[1].set_title("Sobel+Watershed")

    color2 = label2rgb(expanded, image=coins, bg_label=0)
    axes[2].imshow(color2)
    axes[2].set_title("Expanded labels")

    for a in axes:
        a.axis("off")
    fig.tight_layout()
    plt.show()

def find_gradient_segmentation(img):
    img = rgb2gray(img)
    
    image = img_as_ubyte(img)

    # denoise image
    denoised = rank.median(image, disk(2))

    # find continuous region (low gradient -
    # where less than 10 for this image) --> markers
    # disk(5) is used here to get a more smooth image
    markers = rank.gradient(denoised, disk(5)) < 10
    markers = ndi.label(markers)[0]

    # local gradient (disk(2) is used to keep edges thin)
    gradient = rank.gradient(denoised, disk(2))

    # process the watershed
    labels = watershed(gradient, markers)

    # display results
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8),
                            sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title("Original")

    ax[1].imshow(gradient, cmap=plt.cm.nipy_spectral)
    ax[1].set_title("Local Gradient")

    ax[2].imshow(markers, cmap=plt.cm.nipy_spectral)
    ax[2].set_title("Markers")

    ax[3].imshow(image, cmap=plt.cm.gray)
    ax[3].imshow(labels, cmap=plt.cm.nipy_spectral, alpha=.5)
    ax[3].set_title("Segmented")

    for a in ax:
        a.axis('off')

    fig.tight_layout()
    plt.show()

def morph_snakes_seg(img):
    img = rgb2gray(img)

    def store_evolution_in(lst):
        """Returns a callback function to store the evolution of the level sets in
        the given list.
        """

        def _store(x):
            lst.append(np.copy(x))

        return _store


    # Morphological ACWE
    image = img_as_float(img)

    # Initial level set
    init_ls = checkerboard_level_set(image.shape, 6)
    # List with intermediate results for plotting the evolution
    evolution = []
    callback = store_evolution_in(evolution)
    ls = morphological_chan_vese(image, num_iter=35, init_level_set=init_ls,
                                smoothing=3, iter_callback=callback)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax = axes.flatten()

    ax[0].imshow(image, cmap="gray")
    ax[0].set_axis_off()
    ax[0].contour(ls, [0.5], colors='r')
    ax[0].set_title("Morphological ACWE segmentation", fontsize=12)

    ax[1].imshow(ls, cmap="gray")
    ax[1].set_axis_off()
    contour = ax[1].contour(evolution[2], [0.5], colors='g')
    contour.collections[0].set_label("Iteration 2")
    contour = ax[1].contour(evolution[7], [0.5], colors='y')
    contour.collections[0].set_label("Iteration 7")
    contour = ax[1].contour(evolution[-1], [0.5], colors='r')
    contour.collections[0].set_label("Iteration 35")
    ax[1].legend(loc="upper right")
    title = "Morphological ACWE evolution"
    ax[1].set_title(title, fontsize=12)


    # Morphological GAC
    image = img_as_float(img)
    gimage = inverse_gaussian_gradient(image)

    # Initial level set
    init_ls = np.zeros(image.shape, dtype=np.int8)
    init_ls[10:-10, 10:-10] = 1
    # List with intermediate results for plotting the evolution
    evolution = []
    callback = store_evolution_in(evolution)
    ls = morphological_geodesic_active_contour(gimage, num_iter=230,
                                            init_level_set=init_ls,
                                            smoothing=1, balloon=-1,
                                            threshold=0.69,
                                            iter_callback=callback)

    ax[2].imshow(image, cmap="gray")
    ax[2].set_axis_off()
    ax[2].contour(ls, [0.5], colors='r')
    ax[2].set_title("Morphological GAC segmentation", fontsize=12)

    ax[3].imshow(ls, cmap="gray")
    ax[3].set_axis_off()
    contour = ax[3].contour(evolution[0], [0.5], colors='g')
    contour.collections[0].set_label("Iteration 0")
    contour = ax[3].contour(evolution[100], [0.5], colors='y')
    contour.collections[0].set_label("Iteration 100")
    contour = ax[3].contour(evolution[-1], [0.5], colors='r')
    contour.collections[0].set_label("Iteration 230")
    ax[3].legend(loc="upper right")
    title = "Morphological GAC evolution"
    ax[3].set_title(title, fontsize=12)

    fig.tight_layout()
    plt.show()

def find_merge_rag(img):
        
    def _weight_mean_color(graph, src, dst, n):
        """Callback to handle merging nodes by recomputing mean color.

        The method expects that the mean color of `dst` is already computed.

        Parameters
        ----------
        graph : RAG
            The graph under consideration.
        src, dst : int
            The vertices in `graph` to be merged.
        n : int
            A neighbor of `src` or `dst` or both.

        Returns
        -------
        data : dict
            A dictionary with the `"weight"` attribute set as the absolute
            difference of the mean color between node `dst` and `n`.
        """

        diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
        diff = np.linalg.norm(diff)
        return {'weight': diff}


    def merge_mean_color(graph, src, dst):
        """Callback called before merging two nodes of a mean color distance graph.

        This method computes the mean color of `dst`.

        Parameters
        ----------
        graph : RAG
            The graph under consideration.
        src, dst : int
            The vertices in `graph` to be merged.
        """
        graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
        graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
        graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                        graph.nodes[dst]['pixel count'])


    # img = data.coffee()
    labels = segmentation.slic(img, compactness=30, n_segments=400, start_label=1)
    g = graph.rag_mean_color(img, labels)

    labels2 = graph.merge_hierarchical(labels, g, thresh=35, rag_copy=False,
                                    in_place_merge=True,
                                    merge_func=merge_mean_color,
                                    weight_func=_weight_mean_color)

    out = color.label2rgb(labels2, img, kind='avg', bg_label=0)
    out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))
    io.imshow(out)
    io.show()

img = load_image("D:/Projects/ai/data/unclean/testing/lz000.png")
img_mask = load_image("D:/Projects/ai/data/unclean/testing/lz000-mask-dilated.png")
# find_active_contour(img, img_mask)
# find_canny(img)
# find_countour(img)
# find_active_contour(img, 600,300,700,700)
# find_convex_hull(img)
# rag_threshold(img)
# find_threshold(img)
# find_local_max(img)
# find_local_threshold(img)
# find_expanded_segmentation(img)
# find_gradient_segmentation(img)
# morph_snakes_seg(img)

# find_contour_ex()
find_merge_rag(img)
