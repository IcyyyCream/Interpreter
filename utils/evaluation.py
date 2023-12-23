import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from copy import deepcopy

from argument import imagenet_path, ground_path
from model.detect import detect_via_image
from utils.auxiliary import parse_xml


def patch_mask(original_image, heat, patch_size=3):
    """
    By masking the maximum position of the heat-map obtaining the decrease of masked image prediction.

    :param original_image: (3D ndarray) Original input image
    :param heat: (2D ndarray) Raw heat-map (before normalizing and composition, belonging to [0, 1))
    :param patch_size: (int) Size of the mask block
    :return: (double) Difference between original prediction score and mask prediction score
    """

    prop = detect_via_image(original_image)[0]
    c_index = np.argmax(prop)
    # print('category index: ', c_index)
    original_p = prop[c_index]

    mask_image = deepcopy(original_image)
    index = np.where(heat == np.max(heat))

    if index[0].shape[0] == 0 or index[1].shape[0] == 0:     # 防止传入异常值
        return 0
    else:
        # print(mask_image.shape)
        for ind_x in range(max(index[0][0] - patch_size, 0), min(index[0][0] + patch_size + 1, mask_image.shape[0])):
            for ind_y in range(max(index[1][0] - patch_size, 0), min(index[1][0] + patch_size + 1, mask_image.shape[1])):
                mask_image[ind_x, ind_y, :] = 128
                # mask_image[ind_x, ind_y, :] = 0
        # plt.imshow(mask_image)
        # plt.show()
        # img_save = Image.fromarray(mask_image)
        # img_save.save('mask_image_' + str(patch_size) + '.png')
        new_prop = detect_via_image(mask_image)[0]
        new_p = new_prop[c_index]

        # plt.imshow(original_image)
        # plt.show()
        # plt.imshow(mask_image)
        # plt.show()

        return original_p - new_p


def point_game(heat, boundary_box):
    """
    Measuring whether the max activation point of the heat-map falls within the boundary box
    and calculate the pointing game score.

    :param heat: (2D ndarray) Raw heat-map (before normalizing and composition, belonging to [0, 1))
    :param boundary_box: (1D list) Four values to define a boundary box (x_left, x_right, y_left, y_right)
    :return: (int) Represents whether the maximum point hit
    """

    index = np.where(heat == np.max(heat))
    if index[0].shape[0] == 0 or index[1].shape[0] == 0:     # 防止传入异常值
        return 0
    else:
        if boundary_box[0] < index[0][0] < boundary_box[1] and boundary_box[2] < index[1][0] < boundary_box[3]:
            return 1
        else:
            return 0


def iou(heat, boundary_box, threshold=0.5):
    """
    Intersection over union between heat map area and boundary box area.
    :param heat: (2D ndarray) Raw heat-map (before normalizing and composition, belonging to [0, 1))
    :param boundary_box: (1D list) Four values to define a boundary box (x_left, x_right, y_left, y_right)
    :param threshold: (double) Pixels greater than this threshold will participate in the calculation
    :return: (double) IoU value
    """

    inter = 0
    union = (boundary_box[1] - boundary_box[0]) * (boundary_box[3] - boundary_box[2])

    index = np.where(heat > threshold)
    if index[0].shape[0] == 0 or index[1].shape[0] == 0:     # 防止传入异常值
        return 0
    else:
        for ind_x, ind_y in zip(index[0], index[1]):
            if boundary_box[0] < ind_x < boundary_box[1] and boundary_box[2] < ind_y < boundary_box[3]:
                inter += 1
            else:
                union += 1

        return inter / union
