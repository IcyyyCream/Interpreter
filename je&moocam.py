import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm
import os

from argument import imagenet_path, ground_path, img_save_path, res_save_path
from utils.evaluation import patch_mask, point_game, iou
from model.getfeatures import get_heatmap
from nsga2.nsga2_feature import PopulationFeature
from pytorch_grad_cam.utils.image import show_cam_on_image
from utils.auxiliary import show_img_loc, parse_xml, data_write2excel


def nsga2_features_explain(image_path, iteration=50, p_size=50, label=None):
    """
    Using JE&MOO-CAM to explain CNN model.

    :param image_path: (str) Image path to be explained
    :param iteration: (int) Population Iterations
    :param p_size: (int) Population Size
    :param label: (int) Category to be explained. If None, then maximum predicted class
    :return: visualization (3D ndarray) Interpretable heat-map
        heatmap: (2D ndarray [0, 1)) Raw heat-map activation value for quantitative evaluation
    """

    img = Image.open(image_path).convert('RGB')
    img_size = img.size
    img_array = np.array(img)
    pop1 = PopulationFeature(img, p_size, label)

    for _ in tqdm(range(iteration)):
        offs = pop1.scm()
        pop1.regeneration(offs)

    # 取目标函数1最小的解作为最终解释
    res_priority = sorted(pop1.pareto_list[0], key=lambda _: (pop1.f1_value[_], pop1.f2_value[_]))
    res_index = res_priority[0]
    res = pop1.population[res_index]

    heatmap = get_heatmap(pop1.original_features, res, target_size=img_size)
    visualization = show_cam_on_image(img_array.astype(dtype=np.float32) / 255., heatmap, use_rgb=True)

    # feature1 = deepcopy(res)
    # for i in range(int(len(feature1) / 5 * 1), int(len(feature1) / 5 * 4)):
    #     feature1[i] = 0
    # heatmap = get_heatmap(pop1.original_features, feature1, target_size=img_size)
    # visualization = show_cam_on_image(np.zeros(img_array.shape), heatmap, use_rgb=True)

    return visualization, heatmap

    # 展示所有pareto等级1的解
    # results = []
    # for i in pop1.pareto_list[0]:
    #     results.append(pop1.population[i])
    # results.append(np.ones(pop1.individual_size))
    #
    # for res in results:
    #     heatmap = get_heatmap(pop1.original_features, res, target_size=img_size)
    #     vis = show_cam_on_image(img_array.astype(dtype=np.float32) / 255., heatmap, use_rgb=True)
    #     plt.imshow(vis)
    #     plt.show()


def explain_demo():
    """
    Explain single images and show its heat map.
    """

    img_path = imagenet_path + 'ILSVRC2012_val_00000112.JPEG'
    vis, _ = nsga2_features_explain(img_path, 100, 50, label=720)
    show_img_loc(vis)
    vis = Image.fromarray(vis)
    vis.save('hm1.png')


def arg_selection():
    """
    Parameter selection experiment.
    """

    result_save_path = res_save_path + 'op_cam_arg/'
    file_name = 'result_' + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.xlsx'
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    args = [(50, 50), (50, 100), (50, 150), (100, 50), (100, 100), (100, 150), (150, 50), (150, 100), (150, 150)]
    # args = [(100, 50), (100, 100), (100, 150)]
    # args = [(150, 50), (150, 100), (150, 150)]

    results = []
    for arg in args:
        ave_res = []  # each row is: patch_mask(1, 3, 5, 7, 9), iou(0.5, 0.7, 0.9), point_game
        for i in range(150, 200 + 1):
            img_name = 'ILSVRC2012_val_0000' + str(i).zfill(4) + '.JPEG'
            print('Starting argument: ' + str(arg[0]) + ', ' + str(arg[1]) + ', image: ' + img_name)
            xml_name = 'ILSVRC2012_val_0000' + str(i).zfill(4) + '.xml'
            img_path = imagenet_path + img_name

            c_name, bbox = parse_xml(ground_path + xml_name)
            img = np.array(Image.open(img_path).convert('RGB'))

            vis, cam = nsga2_features_explain(img_path, arg[0], arg[1])

            res = []
            for p in [1, 3, 5, 7, 9]:
                res.append(patch_mask(img, cam, p))
            for t in [0.5, 0.7, 0.9]:
                res.append(iou(cam, bbox, t))
            res.append(point_game(cam, bbox))
            ave_res.append(res)
        results.append(np.array(ave_res).mean(axis=0).tolist())

    data_write2excel(result_save_path + file_name, results)
    print('All Down!')


def ablation(it=100, img_mun=100):
    """
    Disturbing CNN parameters and conducting model sensitivity experiments.
    """

    gradcam_save_path = img_save_path + 'op_cam_abla/'
    result_save_path = res_save_path + 'op_cam_abla/'
    file_name = 'result_' + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.xlsx'

    if not os.path.exists(gradcam_save_path):
        os.makedirs(gradcam_save_path)

    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    results = []  # each row is: patch_mask(1, 3, 5, 7, 9), iou(0.5, 0.7, 0.9), point_game
    for i in range(51, img_mun + 1):
        img_name = 'ILSVRC2012_val_0000' + str(i).zfill(4) + '.JPEG'
        print('Starting ' + img_name)
        xml_name = 'ILSVRC2012_val_0000' + str(i).zfill(4) + '.xml'
        img_path = imagenet_path + img_name

        c_name, bbox = parse_xml(ground_path + xml_name)
        img = np.array(Image.open(img_path).convert('RGB'))

        vis, cam = nsga2_features_explain(img_path, it, label=0)
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

    data_write2excel(result_save_path + file_name, results)
    print('All Down!')


def main(it=100, image_num=50):
    """
    Explain and evaluate.
    """

    gradcam_save_path = img_save_path + 'op_cam_' + str(it) + '/'
    result_save_path = res_save_path + 'op_cam_' + str(it) + '/'
    file_name = 'result_' + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.xlsx'

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

        vis, cam = nsga2_features_explain(img_path, it)
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

    data_write2excel(result_save_path + file_name, results)
    print('All Down!')


if __name__ == "__main__":
    # explain_demo()
    main(100, 300)
    # arg_selection()
    # ablation(img_mun=100)
