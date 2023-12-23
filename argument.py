import torch

"""
Unify the writing of paths in files to ensure normal operation across devices.
model_path: (file path) Model weight file
image_path: (dir path) ImageNet-mini image library
ground_path: (dir path) ImageNet-mini ground truth library
label_path: (file path) ImageNet labels
img_save_path: (dir path) Heatmaps save path
res_save_path: (dir path) Quantitative evaluation save path
"""

"""
When you change model, these files should be changed
argument.py, xxxcam.py (for other CAMs), model.getfeatures.py (for JE&MOO-CAM), model.detect.py
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# change here
model_name = 'vgg16'
# model_name = 'alexnet'
# model_name = 'resnet50'

model_path = 'D:/PyCharmLibrary/Interpreter/model/' + model_name + '.pth'
imagenet_path = 'E:/ImageNet1k_val/ILSVRC2012_img_val/'
ground_path = 'E:/ImageNet1k_val/ILSVRC2012_bbox_val/'
label_path = 'E:/ImageNet1k_val/ImageNet1K_labels.txt'
img_save_path = 'E:/ImageNet1k_Interpreter/' + model_name + '/'
res_save_path = 'D:/PyCharmLibrary/Interpreter/results/' + model_name + '/'
