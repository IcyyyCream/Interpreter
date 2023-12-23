import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision import transforms as T
import matplotlib.pyplot as plt
import xml.dom.minidom as xmldom
import cv2
import xlwt
import pandas as pd


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as a heatmap.
    By default, the heatmap is in BGR format.
    param img: The base image in RGB or BGR format.
    param mask: The cam mask.
    param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    param colormap: The OpenCV colormap to be used.
    returns: The default image with the cam overlay.
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)

    return np.uint8(255 * cam)


def get_input_tensors(image):
    """
    Only Normalize.

    :param image: (torch.Tensor, numpy.ndarray) Image data
    :return: None
    """

    transform = T.Compose([T.ToTensor(),
                           T.Normalize([0.485, 0.456, 0.406],
                                       [0.229, 0.224, 0.225])])
    return transform(image).unsqueeze(0)


def show_img_tb(images):
    """
    Show images on Tensorboard.

    :param images: (torch.Tensor or its List) Image data
    :return: None
    """

    writer = SummaryWriter("tf-logs")
    if isinstance(images, list):
        step = 0
        for image in images:
            writer.add_image(tag="explanations", img_tensor=image, global_step=step)
            step += 1
    else:
        writer.add_image(tag="explanations", img_tensor=images)
    writer.close()


def show_img_loc(images):
    """
    Show images locally.

    :param images: (torch.Tensor, numpy.ndarray or their List) Image data
    :return: None
    """

    if isinstance(images, list):
        for image in images:
            plt.imshow(image)
            plt.show()
    else:
        plt.imshow(images)
        plt.show()


def parse_xml(xml_path):
    """
    Parse the xml file to obtain the boundary box and ground truth (as class name).

    :param xml_path: (str) xml file path
    :return: (str, 1D list [4 * int]) class name, bbox info [x min, x max, y min, y max]
    """
    xml_file = xmldom.parse(xml_path)
    xml_obj = xml_file.documentElement

    class_name = xml_obj.getElementsByTagName('name')[0].firstChild.data
    x_min = xml_obj.getElementsByTagName('xmin')[0].firstChild.data
    x_max = xml_obj.getElementsByTagName('xmax')[0].firstChild.data
    y_min = xml_obj.getElementsByTagName('ymin')[0].firstChild.data
    y_max = xml_obj.getElementsByTagName('ymax')[0].firstChild.data

    return class_name, [int(x_min), int(x_max), int(y_min), int(y_max)]


def draw_bbox(image, boundary_box):
    """
    Draw the boundary box.

    :param image: (3D ndarray) Original image
    :param boundary_box: Boundary box information (x_min, x_max, y_min, y_max)
    :return: None
    """

    cv2.rectangle(image, (boundary_box[0], boundary_box[2]), (boundary_box[1], boundary_box[3]), (255, 0, 0), 2)


def data_write2excel(file_path, datas):
    """
    Write the data into Excel, starting from the start line.

    :param file_path: (str) Excel file path
    :param datas: (2D list) Evaluation data, each row presents an image
    :return: None
    """

    df = pd.DataFrame(datas)
    df.to_excel(file_path)
