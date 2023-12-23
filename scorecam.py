import numpy as np
import torch, gc
import torchvision
from PIL import Image
import os

from argument import model_path, imagenet_path, ground_path, img_save_path, res_save_path
from utils.auxiliary import get_input_tensors, show_img_loc
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from utils.evaluation import patch_mask, point_game, iou
from utils.auxiliary import parse_xml, data_write2excel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# change here
# 导入vgg16模型
model = torchvision.models.vgg16(weights=None).to(device)
target_layers = [model.features[-1]]

# 导入alexnet模型
# model = torchvision.models.alexnet(weights=None).to(device)
# target_layers = [model.features[-1]]

# 导入resnet50模型
# model = torchvision.models.resnet50(weights=None).to(device)
# target_layers = [model.layer4[-1]]
# load model
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


def gradcam_explain(image_path):
    """
    Using Grad-CAM to explain vgg16 model.

    :param image_path: (str) Image path to be explained
    :return: visualization: (3D ndarray) Interpretable heat-map
        grayscale_cam: (2D ndarray [0, 1)) Raw heat-map activation value for quantitative evaluation
    """

    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
    # target_layers可以传入多个层结构，获取哪一个网络层结构的输出
    # 可视化vgg16最后一层特征图

    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    img_tensor = get_input_tensors(img_array).to(device)

    cam = ScoreCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    # 目标类序号
    # target_label = 0
    # None表示输出值最大类
    grayscale_cam = cam(input_tensor=img_tensor)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img_array.astype(dtype=np.float32) / 255., grayscale_cam, use_rgb=True)

    return visualization, grayscale_cam


def explain_demo():
    img_path = imagenet_path + 'ILSVRC2012_val_00000001.JPEG'
    vis, _ = gradcam_explain(img_path)
    show_img_loc(vis)


def main(image_num=50):
    """
    Explain and evaluate.
    """

    gradcam_save_path = img_save_path + 'score_cam/'
    result_save_path = res_save_path + '/score_cam/'

    if not os.path.exists(gradcam_save_path):
        os.makedirs(gradcam_save_path)

    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    results = []    # each row is: patch_mask(1, 3, 5, 7, 9), iou(0.5, 0.7, 0.9), point_game
    for i in range(1, image_num + 1):
        img_name = 'ILSVRC2012_val_0000' + str(i).zfill(4) + '.JPEG'
        print('Starting ' + img_name)
        xml_name = 'ILSVRC2012_val_0000' + str(i).zfill(4) + '.xml'
        img_path = imagenet_path + img_name

        c_name, bbox = parse_xml(ground_path + xml_name)
        img = np.array(Image.open(img_path).convert('RGB'))

        vis, cam = gradcam_explain(img_path)
        vis = Image.fromarray(vis)
        vis.save(gradcam_save_path + img_name)

        res = []
        for p in [1, 3, 5, 7, 9]:
            res.append(patch_mask(img, cam, p))
        for t in [0.5, 0.7, 0.9]:
            res.append(iou(cam, bbox, t))
        res.append(point_game(cam, bbox))
        results.append(res)

    # 计算平均值
    mean = np.array(results).mean(axis=0).tolist()
    results.append(mean)

    data_write2excel(result_save_path + 'result.xlsx', results)
    print('All Down!')


if __name__ == "__main__":
    # dir_path = 'D:/PyCharmLibrary/Interpreter/model/test_images/ImageNet-mini/image'
    # img_path_list = os.listdir(dir_path)
    # images = []
    # for i in range(len(img_path_list)):
    #     if not img_path_list[i].startswith("."):
    #         img_name = os.path.join(dir_path, img_path_list[i])
    #         print(f"Explaining {img_name}")
    #         explain = gradcam_explain(img_name)
    #         images.append(explain)
    # show_img_tb(images)
    main(300)
