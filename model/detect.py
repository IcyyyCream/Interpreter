import numpy as np
import torch
import torch.nn as nn
import heapq
from PIL import Image
import torchvision
from torchvision.transforms import transforms as T
from tqdm import tqdm

from argument import model_path, imagenet_path, ground_path, label_path, res_save_path
from utils.auxiliary import parse_xml, data_write2excel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# change here
# 导入vgg16模型
model = torchvision.models.vgg16(weights=None).to(device)

# 导入alexnet模型
# model = torchvision.models.alexnet(weights=None).to(device)

# 导入resnet50模型
# model = torchvision.models.resnet50(weights=None).to(device)

model_arg = torch.load(model_path, map_location=device)


# randomize model
def randomize_vgg16():
    # features.28.weight
    arg = model_arg['features.28.weight'].data
    random = torch.rand(arg.size()[1], 1)
    for i in range(arg.size()[1]):
        #     arg[:][i][0][0] = random[i]   # 30%
        #     arg[:][i][0][2] = random[i]   # 55%
        #     arg[:][i][0][1] = random[i]   # 75%
        arg[:][i][1][1] = random[i]  # 10%
    #     arg[:][i][2][2] = random[i]   # 30%
    #     arg[:][i][2][0] = random[i]   # 55%
    #     arg[:][i][2][1] = random[i]   # 75%
    model_arg['features.28.weight'].data = arg


def randomize_alexnet():
    # features.28.weight
    arg = model_arg['features.10.weight'].data
    # print(arg.shape)
    random = torch.rand(arg.size()[1], 1)
    for i in range(arg.size()[1]):
        arg[:][i][0][0] = random[i]   # 30%
        arg[:][i][0][2] = random[i]   # 55%
        arg[:][i][0][1] = random[i]   # 75%
        arg[:][i][1][1] = random[i]  # 10%
        arg[:][i][2][2] = random[i]   # 30%
        arg[:][i][2][0] = random[i]   # 55%
        arg[:][i][2][1] = random[i]   # 75%
    model_arg['features.10.weight'].data = arg


def randomize_resnet50():
    # features.28.weight
    arg = model_arg['layer4.2.conv2.weight'].data
    # print(arg.shape)
    random = torch.rand(arg.size()[1], 1)
    for i in range(arg.size()[1]):
        arg[:][i][0][0] = random[i]   # 30%
        arg[:][i][0][2] = random[i]   # 55%
        arg[:][i][0][1] = random[i]   # 75%
        arg[:][i][1][1] = random[i]  # 10%
        arg[:][i][2][2] = random[i]   # 30%
        arg[:][i][2][0] = random[i]   # 55%
        arg[:][i][2][1] = random[i]   # 75%
    model_arg['layer4.2.conv2.weight'].data = arg


# randomize_vgg16()
# randomize_alexnet()
# randomize_resnet50()

# load model
model.load_state_dict(model_arg)
model.eval()

transform = T.Compose([T.ToTensor(),
                       T.Resize((224, 224)),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def detect_via_path(image_path):
    """
    Generate model prediction through image path. Including softmax.

    :param image_path: (str) The path of input image
    :return: (2D array) Model prediction output
    """

    image = Image.open(image_path).convert('RGB')
    # 修改图片维度为[1, 3, 224, 224]
    image = transform(image).unsqueeze(0).to(device)
    # print(img.shape)
    # print(vgg16)

    with torch.no_grad():
        output = model(image)
        y = nn.Softmax(dim=1)(output).cpu()

    return np.array(y)


def detect_via_image(image):
    """
    Generate model prediction through an image. Without softmax.

    :param image: (ndarray or tensor or PIL.Image) Input image
    :return: (2D array) Model prediction output
    """

    # 修改图片维度为[1, 3, 224, 224]
    image = transform(image).unsqueeze(0).to(device)
    # print(img.shape)
    # print(vgg16)

    with torch.no_grad():
        output = model(image).cpu()
        # y = nn.Softmax(dim=1)(output).cpu()

    return np.array(output)


def read_labels():
    """
    Read ImageNet1k labels from txt file

    :return: (dictionary) label index: label name
    """

    label_dict = {}
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            if line != '':
                label_dict[line.split(' ', 2)[0]] = line.split(' ', 2)[1]
        # split()函数用法: 逗号前面是以什么来分割, 后是分割成 n+1 个部分, 且以数组形式从 0 开始

    return label_dict


def test_acc(image_num=9999):
    """
    Test CNN model prediction accuracy.

    :param image_num: (int) Number of image tested
    :return: (double) Model prediction accuracy
    """

    hits = 0
    label_dict = read_labels()
    print('----------Starting model accuracy test----------')
    for k in tqdm(range(image_num)):
        img_path = imagenet_path + 'ILSVRC2012_val_0000' + str(k + 1).zfill(4) + '.JPEG'
        xml_path = ground_path + 'ILSVRC2012_val_0000' + str(k + 1).zfill(4) + '.xml'
        pre = detect_via_path(img_path)
        pre_name = label_dict[str(np.argmax(pre[0]))]
        c_name, _ = parse_xml(xml_path)
        if pre_name == c_name:
            hits += 1

    print('Model accuracy: ' + str(hits / image_num))

    return hits / image_num


def save_pre_val(image_num=300):
    """

    """
    results = []
    for k in tqdm(range(image_num)):
        img_path = imagenet_path + 'ILSVRC2012_val_0000' + str(k + 1).zfill(4) + '.JPEG'
        image = Image.open(img_path).convert('RGB')
        pre = detect_via_image(image)
        pre_val = max(pre[0])
        # print(pre_val)
        results.append(pre_val)

    data_write2excel(res_save_path + 'prediction.xlsx', results)


if __name__ == "__main__":
    test_acc()

    # prop = detect_via_path(imagenet_path + 'ILSVRC2012_val_00000112.JPEG')
    # top3_prop = heapq.nlargest(3, prop[0])
    # top3_index = heapq.nlargest(3, range(len(prop[0])), prop[0].take)
    # print(top3_prop)
    # print(top3_index)
    # save_pre_val()
