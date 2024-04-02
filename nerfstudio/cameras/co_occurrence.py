import sys
import numpy as np
from skimage import feature
from skimage.io import imread, imsave
from skimage.color import rgb2gray, rgb2lab, rgb2hsv
from numba import jit, njit
import matplotlib.pyplot as plt
import PIL
from time import time, strftime
from cotm import pmi2dissimilarity
from collect_cooc_conv import collect_cooc
from cotm_quantization import cotm_quantization, recreate_image
from scipy.ndimage.filters import gaussian_filter
sys.path.extend(['/root/sharedfolder/CoTM'])

QX_DEF_CHAR_MAX = 255
FACTOR_INIT_VAL = 1.0


class RBFilter(object):
    """
    Recursive Bilateral Filter Implementation

    """
    def __init__(self, s_xy, s_r, similarity_mat=None):
        self.s_xy = s_xy
        self.s_r = s_r
        self.height = 0
        self.width = 0
        self.channels = 0
        self.use_ext_similarity = False
        self.similarity_mat = similarity_mat
        if similarity_mat is not None:
            self.use_ext_similarity = True

        self.range_lut = np.zeros(QX_DEF_CHAR_MAX+1)

        self.alpha_xy_f = 0.0
        self.inv_alpha_xy_f = 1.0

        self.left_pass_factor = []
        self.left_pass_color = []
        self.right_pass_factor = []
        self.right_pass_color = []
        self.down_pass_factor = []
        self.down_pass_color = []
        self.up_pass_factor = []
        self.up_pass_color = []
        self.img_out = []

    def _alloc_arrays(self, height, width, channels):
        self.left_pass_color = np.zeros([height, width, channels], dtype=np.float32)
        self.left_pass_factor = np.zeros([height, width], dtype=np.float32)
        self.right_pass_color = np.zeros([height, width, channels], dtype=np.float32)
        self.right_pass_factor = np.zeros([height, width], dtype=np.float32)
        self.down_pass_color = np.zeros([height, width, channels], dtype=np.float32)
        self.down_pass_factor = np.zeros([height, width], dtype=np.float32)
        self.up_pass_color = np.zeros([height, width, channels], dtype=np.float32)
        self.up_pass_factor = np.zeros([height, width], dtype=np.float32)

        self.img_out = np.zeros([height, width, channels], dtype=np.float32)

    def _get_alpha(self, px1, px2):
        if self.use_ext_similarity is True:
            return self.similarity_mat[px1[0], px2[0]]
        else:
            diff = self._get_diff_factor(px1, px2)
            return self.range_lut[diff]

    def _get_diff_factor(self, px1, px2):
        # Todo: apply for a whole row/col at once, and for a single value (gray, RGB px)
        px1 = px1.astype(np.int)
        px2 = px2.astype(np.int)
        if self.channels == 1:
            diff = max(px1, px2) - min(px1, px2)
            return diff

        diff = np.maximum(px1, px2) - np.minimum(px1, px2)
        diff = np.right_shift((diff[0] + diff[2]), 2) + np.right_shift(diff[1], 1)
        return diff.astype(np.uint8)

    def _left_to_right(self, img_ref, img_tgt, height, width):
        """
        Left to right filtering pass
        """
        # Handle first pixel separately
        self.left_pass_factor[:, 0] = FACTOR_INIT_VAL
        self.left_pass_color[:, 0] = img_tgt[:, 0]

        # Todo: Replace self.inv_alpha_xy_f with a new 1-alpha_xy_f calc??
        for x in range(1, width):
            alpha_xy_vec = self.similarity_mat[img_ref[:, x], img_ref[:, x-1]]
            self.left_pass_factor[:, x] = self.inv_alpha_xy_f + np.multiply(alpha_xy_vec, self.left_pass_factor[:, x-1])
            alpha_xy_vec_3d = np.dstack(tuple([alpha_xy_vec for _ in range(self.channels)]))
            self.left_pass_color[:, x] = self.inv_alpha_xy_f * img_tgt[:, x] + np.multiply(alpha_xy_vec_3d, self.left_pass_color[:, x-1])

    def _right_to_left(self, img_ref, img_tgt, height, width):
        #  Right to left scan
        self.right_pass_factor[:, width-1] = FACTOR_INIT_VAL
        self.right_pass_color[:, width-1] = img_tgt[:, width-1].astype(np.float32)

        for x in range(width-2, -1, -1):
            alpha_xy_vec = self.similarity_mat[img_ref[:, x], img_ref[:, x+1]]
            self.right_pass_factor[:, x] = self.inv_alpha_xy_f + np.multiply(alpha_xy_vec, self.right_pass_factor[:, x+1])
            alpha_xy_vec_3d = np.dstack(tuple([alpha_xy_vec for _ in range(self.channels)]))
            self.right_pass_color[:, x] = self.inv_alpha_xy_f * img_tgt[:, x] \
                                            + np.multiply(alpha_xy_vec_3d, self.right_pass_color[:, x+1])

    def _pre_vertical(self):
        """
        Set up values as preparation for the vertical passes
        """
        factor_sum = self.left_pass_factor + self.right_pass_factor
        if self.channels == 3:
            factor_sum_stack = np.dstack([factor_sum, factor_sum, factor_sum])
        else:
            factor_sum_stack = factor_sum
        color_sum = self.left_pass_color + self.right_pass_color
        self.img_out = np.divide(color_sum, factor_sum_stack)

    def _down(self, img_ref, height, width):
        """
        1st vertical pass - Down
        """
        # 1st line done separately, no previous line
        self.down_pass_factor[0] = FACTOR_INIT_VAL * np.ones([width], dtype=np.float32)
        self.down_pass_color[0] = self.img_out[0]

        # Rest of lines
        for y in range(1, height):
            alpha_xy_vec = self.similarity_mat[img_ref[y, :], img_ref[y-1, :]]
            self.down_pass_factor[y, :] = self.inv_alpha_xy_f + np.multiply(alpha_xy_vec, self.down_pass_factor[y-1, :])
            alpha_xy_vec_3d = np.dstack(tuple([alpha_xy_vec for _ in range(self.channels)]))
            self.down_pass_color[y, :] = self.inv_alpha_xy_f * self.img_out[y, :] \
                                            + np.multiply(alpha_xy_vec_3d, self.down_pass_color[y-1, :])

    def _up(self, img_ref, height, width):
        """
        2nd vertical pass - Up
        """
        # Last line done separately, no previous line
        self.up_pass_factor[height-1] = FACTOR_INIT_VAL * np.ones([width], dtype=np.float32)
        self.up_pass_color[height-1] = self.img_out[height-1]

        for y in range(height-2, -1, -1):
            alpha_xy_vec = self.similarity_mat[img_ref[y, :], img_ref[y+1, :]]
            self.up_pass_factor[y, :] = self.inv_alpha_xy_f + np.multiply(alpha_xy_vec, self.up_pass_factor[y+1, :])
            alpha_xy_vec_3d = np.dstack(tuple([alpha_xy_vec for _ in range(self.channels)]))
            self.up_pass_color[y, :] = self.inv_alpha_xy_f * self.img_out[y, :] \
                                            + np.multiply(alpha_xy_vec_3d, self.up_pass_color[y+1, :])

    def _post_vertical(self):
        """
        Average result of vertical pass is written to the output
        """
        factor_sum = self.down_pass_factor + self.up_pass_factor
        # Testing Edge detection using factors
        self.factor_sum = factor_sum
        color_sum = self.down_pass_color + self.up_pass_color
        if self.channels == 3:
            factor_sum_stack = np.dstack([factor_sum, factor_sum, factor_sum])
        else:
            factor_sum_stack = factor_sum
        self.img_out = np.divide(color_sum, factor_sum_stack)
        self.img_out = self.img_out.astype(np.uint8)

    def _calc_lut(self):
        self.alpha_xy_f = np.exp((-np.sqrt(2.0))/(self.s_xy * 255.0))
        self.inv_alpha_xy_f = 1.0 - self.alpha_xy_f
        inv_sigma_range = 1.0 / (self.s_r * QX_DEF_CHAR_MAX)
        ii = inv_sigma_range * np.linspace(1.0, QX_DEF_CHAR_MAX+1, 1.0)
        self.range_lut = self.alpha_xy_f * np.exp(ii)

    def filter(self, img_ref, img_tgt, iter=1):
        """
        Joint CoOc image filtering. Reference and Target images are constrained
        by Height and Width but not by the number of channels
        """
        # Todo: Force ref image to be ***SINGLE CHANNEL****
        if len(img_ref.shape) == 2:
            height, width = img_ref.shape
            img_ref_3d = np.dstack([img_ref, img_ref, img_ref])
            img_ref_1d = img_ref
        else:
            height, width, channels = img_ref.shape
            img_ref_3d = img_ref
            img_ref_1d = img_ref[..., 0]
        if len(img_tgt.shape) == 2:
            tgt_height, tgt_width = img_tgt.shape
            img_tgt_3d = np.dstack([img_tgt, img_tgt, img_tgt])
            channels = 3
        else:
            tgt_height, tgt_width, channels = img_tgt.shape
            img_tgt_3d = img_tgt

        assert (tgt_height == height and tgt_width == width), "Reference and Target Height and Width must be the same"

        self.height = height
        self.width = width
        self.channels = channels
        self._alloc_arrays(height, width, channels)
        self._calc_lut()
        # Todo: apply to greyscale too
        img_tgt_iter = self._filter(img_ref_1d, img_tgt_3d, height, width)
        for i in range(iter-1):
            img_tgt_iter = self._filter(img_ref_1d, img_tgt_iter, height, width)
        return self.img_out

    def _filter(self, img_ref_1d, img_tgt_3d, height, width):
        self._left_to_right(img_ref_1d, img_tgt_3d, height, width)
        self._right_to_left(img_ref_1d, img_tgt_3d, height, width)
        self._pre_vertical()
        self._down(img_ref_1d, height, width)
        self._up(img_ref_1d, height, width)
        self._post_vertical()
        return self.img_out


def cooc2similarity(cooc_mat, gamma=0.125):
    """ Transform CoOc mat to similarity mat. """
    cooc_diss = pmi2dissimilarity(cooc_mat)
    cooc_similarity = 1 - np.float_power(cooc_diss, gamma)
    cooc_similarity = np.nan_to_num(cooc_similarity)
    return cooc_similarity


def pil2numpy(pil_image):
    return np.array(pil_image)


def numpy2pil(np_image):
    return PIL.Image.fromarray(np_image)


def getpalette(input_img, quantized_img, n_colors=64):
    if len(input_img.shape) == 3:
        channels = input_img.shape[2]
    else:
        channels = 1

    palette = np.zeros([n_colors, channels], dtype=np.uint8)
    for level in range(n_colors):
        color_mask_1d = quantized_img == level
        color_mask_3d = np.dstack([color_mask_1d, color_mask_1d, color_mask_1d])
        level_avg = np.sum(np.multiply(input_img, color_mask_3d), axis=(0, 1))
        level_avg /= np.sum(color_mask_1d)
        palette[level] = np.around(level_avg).astype(np.uint8)

    return palette


def get_cooc_mat(input_img, tgt_filename=None, iters=1, quantization='mbkmeans', sigma=None, hws=None, gamma=None):
    """ Implementation of Recursive Co-Oc filtering
        Allows multiple Quantization methods and collects Co-Oc from
        quantized data by applying a mask (Usually Gaussian filter) to each
        level and aggregating the results according to the quantized input map

        Todo:
           + Further look into steerable filters. 1st and 2nd order gaussian derviatives for CoOc collection
             did not give any interesting results. May be worth further investigation
           + Comparison to Crisp Edges paper by Isola et al.
     """

    if tgt_filename is None:
        tgt_img = input_img
    else:
        tgt_img = imread(tgt_filename)

    #tgt_img = (tgt_img/256).astype(np.uint8)  # For depth superres example

    # input_img = imread('rbf_test.jpg')
    # input_img = imread('Leopard_bw.png')[..., 0]
    # input_img = imread('Leopard.png')
    # input_img = imread('Graffiti.png')
    # input_img = imread('Field.png')
    # input_img = imread('Forest.png')
    # input_img = imread('art.png')
    # input_img = imread('bw_test.png')
#   input_img_gray = input_img
#   input_img_gray = (rgb2gray(input_img)*255).astype(np.uint8)

    if len(input_img.shape) == 3:  # Remove A channel from png files RGBA 4 ch structure
        if input_img.shape[2] == 4:
            input_img = input_img[..., :3]

    """ Parameter Setting """
    s_xy = 0.03
    s_r = 0.1
    iterations = iters
    if hws is None:
        hws = 7
    if sigma is None:
        sigma = 2*np.sqrt(hws) + 1
    if gamma is None:
        gamma = 0.115
    quantization_factor = 4
    n_colors = 256 / quantization_factor

    """ Quantization or RGB to Greyscale """
    if len(input_img.shape) == 3:
        input_img_lab = rgb2lab(input_img)

    start_time = time()
    if quantization is 'kmeans':
        input_img_quantized = cotm_quantization(input_img_lab, n_colors)
        input_img_quantized = input_img_quantized[0].reshape(input_img.shape[:2])

    elif quantization is 'mbkmeans':
        input_img_quantized = cotm_quantization(input_img_lab, n_colors, mbkmeans=True)
        input_img_quantized = input_img_quantized[0].reshape(input_img.shape[:2])

    else:
        # For BW images
        input_img_quantized = (input_img/quantization_factor).astype(np.uint8)
        if len(input_img.shape) == 3:
            input_1ch = input_img_quantized[..., 0]
           #input_img_quantized = np.dstack([input_1ch, input_1ch, input_1ch])
            input_img_quantized = input_1ch

    quant_time = time() - start_time
    print("Quantization took {0:.3f} Seconds".format(quant_time))

    """ Quantized image generation from label map for evaluation (optional)"""
    recreate_quantized_image = False
    if recreate_quantized_image:
        palette = getpalette(input_img, input_img_quantized, n_colors)
        recreated_image = recreate_image(palette, input_img_quantized.flatten(), input_img.shape[0], input_img.shape[1])
        recreated_image = recreated_image.astype(np.uint8)

    """ CoOc Collection and Transformation to Similarity"""
    start_time = time()
    cooc_m = collect_cooc(input_img_quantized, n_colors=n_colors, sigma=sigma, hws=hws)
    # cooc_similarity = cooc2similarity(cooc_m, gamma=gamma)
    col_time = time() - start_time
    print("Collection took {0:.3f} Seconds".format(col_time))

    return cooc_m

    # """ Filtering """
    # start_time = time()
    # rb_filter = RBFilter(s_xy, s_r, similarity_mat=cooc_similarity)
    # output = rb_filter.filter(input_img_quantized, tgt_img, iter=iterations)
    # app_time = time()-start_time
    # print("Filtering took {0:.3f} Seconds".format(app_time))

   #  """ Plotting and saving to file """
   #  #plt.figure(figsize=(14, 12))
   #  plt.imshow(output)
   #  # plt.imshow(input_img[40:360, 150:520, :])
   #  # imsave('input_crop_zoom_4smear.png', input_img[195:310, 270:410, :])
   #  # imsave('output_s_crop_zoom_4smear.png', output[195:310, 270:410, :])
   #  single_title = "s_xy={0:.2f}, hws={1:2d}, s_cooc={2:.3f}, {3:2d} iter, gamma={4:.3f}".format(s_xy*255, int(hws), sigma, iterations, gamma)
   #  plt.title(single_title)
   #  output_filename = 'results/{}_{}_std{}_iter{}.png'.format(file_str, strftime("%b%d_%H%M%S"), int(sigma), iterations)
   # #plt.savefig(output_filename)
   # #plt.show()

    # h, w = input_img.shape[0], input_img.shape[1]
    # figsize = (2+2*int(w/100), 2+int(h/100))
    #
    # # Quantization Before and After
    # if recreate_quantized_image:
    #     fig_quant, axes_quant = plt.subplots(ncols=2, figsize=figsize)
    #     axes_quant[0].imshow(input_img)
    #     axes_quant[0].set_title("Input Target Image Before Quantization")
    #     axes_quant[1].imshow(recreated_image)
    #     axes_quant[1].set_title("Image Quantized to {0} colors using {1} in {2:.3f} sec".format(n_colors, quantization, quant_time))
    #     plt.show()
    #
    # fig, axes = plt.subplots(ncols=2, figsize=figsize)
    # axes[0].imshow(input_img)
    # axes[0].set_title("Input ({0:4d}x{1:4d}), n={2:3d}, {3} quant {4:.3f}s, collect {5:.3f}s, apply "
    #                   "{6:.3f}s".format(h, w, n_colors, quantization, quant_time, col_time, app_time))
    #
    # if np.allclose(output[..., 0], output[..., 1]):   # Handle duplicated grayscale images
    #     axes[1].imshow(output[..., 0], cmap='CMRmap', vmin=0, vmax=255)
    # else:
    #     axes[1].imshow(output)
    # axes[1].set_title("Filtered Image, s_xy={0:.2f}, hws={1:2d}, s_cooc={2:.3f}, {3:2d}"
    #                   " iter, gamma={4:.3f}".format(s_xy*255, int(hws), sigma, iterations, gamma))
    #
    # plt.show()
    # print("Done\n")
    # timestr = strftime("%b%d_%H%M")
   #fig.savefig('results/{}_{}_{}iter_ang{}.png'.format(file_str, timestr, iterations, int(angle)), dpi=100)
   #fig_quant.savefig('results/{}_{}_{}{}.png'.format(file_str, timestr, quantization, n_colors), dpi=100)


# if __name__ == "__main__":
#     main(ref_filename="bw_test.png", quantization='gray')

    #for file_name in ['Leopard.png', 'Graffiti.png', 'Field.png', 'Forest.png']:   #  , 'art.png', 'house.png']:
    #    for sigma in [7.5]:
    #        for iiters in [3]:
    #            for gamma in [0.125]:
    #                main(file_name, iiters, sigma=sigma, gamma=gamma)