import numpy as np
import torch
import torch.nn as nn
import cv2
import torchvision
from PIL import Image
from torchvision.transforms import transforms as T
import copy
from pytorch_grad_cam import ActivationsAndGradients
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model.detect import detect_via_image
from argument import model_path

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

# 导入vgg16模型
target_layer = model.features[-1]
remode = nn.Sequential(model.avgpool, model.classifier).to(device)

# 导入alexnet模型
# target_layer = model.features[-1]
# remode = nn.Sequential(model.avgpool, model.classifier).to(device)

# 导入resnet50模型
# target_layer = model.layer4[-1]
# remode = nn.Sequential(model.avgpool, model.fc).to(device)

model.eval()
remode.eval()


def get_features(image, targets=None, transform=T.Compose([T.ToTensor(),
                                                           T.Resize((224, 224)),
                                                           T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])):
    """
    Do a forward propagation and obtain the feature maps of the target layer.

    :param image: (3D PIL, ndarray, tensor) Input image
    :param targets: (int) Category to be explained. If None, then maximum predicted class
    :param transform: (torchvision.transforms) Transformation on image
    :return: (4D tensor with b*c*w*h) The feature maps,
            (1D float list) Channel gradient of each feature map (normalized)
    """

    # features = []
    # image = transform(image).unsqueeze(0).to(device)
    #
    # def activation(model, input, output):
    #     """
    #     Automatic hook function.
    #     """
    #     activation = output
    #     features.append(activation.cpu().detach())
    #
    # handle = target_layer.register_forward_hook(activation)
    # model(image)
    # handle.remove()
    #
    # features = torch.tensor(features[0].numpy())
    #
    # return features

    image = transform(image).unsqueeze(0).to(device)
    activations_and_grads = ActivationsAndGradients(model, [target_layer], None)
    outputs = activations_and_grads(image)
    if targets is None:
        target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
        targets = [ClassifierOutputTarget(
            category) for category in target_categories]
    else:
        targets = [ClassifierOutputTarget(targets)]

    model.zero_grad()
    loss = sum([target(output) for target, output in zip(targets, outputs)])
    loss.backward()

    feature = activations_and_grads.activations[0]

    grad = []
    grad_tensor = activations_and_grads.gradients[0].squeeze(0)
    for i in range(grad_tensor.size()[0]):
        grad_val = torch.mean(grad_tensor[i]).cpu().item()
        if grad_val < 0:
            grad.append(0)
        else:
            grad.append(grad_val)
    grad = np.array(grad)
    maxV = np.max(grad)
    grad[:] = grad[:] / maxV

    return feature, grad


def mask_features(features, mask_vector):
    """
    Select feature map via mark vector.

    :param features: (4D tensor with b*c*w*h) The feature maps
    :param mask_vector: (1D array) Mask vector representing whether and how the feature map exists
    :return: (4D tensor with b*c*w*h) The feature maps after mask
    """

    new_features = copy.deepcopy(features)
    for i in range(len(mask_vector)):
        new_features[0][i] *= mask_vector[i]

    return new_features


def get_resized(features):
    """
    Expand the feature maps for full connection layer input.

    :param features: (4D tensor with b*c*w*h) The feature maps
    :return: (2D tensor with b*c) 1-dimensional tensor input after resize
    """
    n_num = features.size()[1] * features.size()[2] * features.size()[3]

    return features.view(1, n_num)


def get_prediction(features):
    """
    Send the feature maps back to the model to obtain the prediction probability value.

    :param features: (4D tensor with b*c*w*h) The feature maps
    :return: (2D array) Model prediction output
    """

    with torch.no_grad():
        stage1 = remode.get_submodule('0')
        stage2 = remode.get_submodule('1')
        temp_feature = stage1(features)
        prop = stage2(get_resized(temp_feature).to(device)).cpu()

    return np.array(prop)


def get_heatmap(features, mask_vector, target_size=None):
    """
    Input feature maps and mask vector generating interpretable heat map.

    :param features: (4D tensor with b*c*w*h) The original feature maps
    :param mask_vector: (1D array) Mask vector representing whether the feature map exists
    :param target_size: (tuple with w, h) Original input image size
    :return: (2D array with w*h) The initial heat map
    """

    cam = np.array(mask_features(features, mask_vector)[0])
    cam = cam.sum(axis=0)
    cam[cam < 0] = 0  # ReLU
    heatmap = cam - np.min(cam)
    heatmap = heatmap / (1e-7 + np.max(heatmap))
    if target_size is not None:
        heatmap = cv2.resize(heatmap, target_size)
    heatmap = np.float32(heatmap)

    return heatmap


def test():
    image = Image.open("test_images/wan.jpg")
    prop1 = detect_via_image(image)

    features = get_features(image, model)
    prop2 = get_prediction(features)

    print(prop1)
    print('-----------------------------------')
    print(prop2)


if __name__ == "__main__":
    test()
    # print(vgg16)
    # print(new_vgg16)
