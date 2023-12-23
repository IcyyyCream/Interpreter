### Experimental environment configuration

see `requirements.txt`. 



### Before you run these CAM methods, you should: 

1. Make sure you have downloaded the official pre-trained weights of `VGG16`, `AlexNet`, and `ResNet50`;
2. Make sure you have downloaded the complete `ImageNet1k` validation set, including images, ground truth bounding boxes, and labels;
3. Make sure that the correct file path has been modified in `argument.py`;
4. Make sure that the model to be explained is consistent across different python files (see `argument.py` for specific files);
5. Now you can run `xxxcam.py` in sequence, pay attention to the `image_num` parameter. 