import numpy as np
import torch
import torch.nn as nn
import ttach as tta
from typing import Callable, List, Tuple
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import warnings


warnings.filterwarnings('ignore')


class BaseCAM:
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 use_cuda: bool = False,
                 reshape_transform: Callable = None,
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True) -> None:
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(self,
                        input_tensor: torch.Tensor,
                        target_layers: List[torch.nn.Module],
                        targets: List[torch.nn.Module],
                        activations: torch.Tensor,
                        grads: torch.Tensor) -> np.ndarray:
        raise Exception("Not Implemented")

    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:

        weights = self.get_cam_weights(input_tensor,
                                       target_layer,
                                       targets,
                                       activations,
                                       grads)
        weighted_activations = weights[:, :, None, None] * activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> np.ndarray:

        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        outputs = self.activations_and_grads(input_tensor)
        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(
                category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([output[target]
                       for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

    def get_target_width_height(self,
                                input_tensor: torch.Tensor) -> Tuple[int, int]:
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(
            self,
            input_tensor: torch.Tensor,
            targets: List[torch.nn.Module],
            eigen_smooth: bool) -> np.ndarray:
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     targets,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
            cam = np.maximum(cam, 0)
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(
            self,
            cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return scale_cam_image(result)

    def forward_augmentation_smoothing(self,
                                       input_tensor: torch.Tensor,
                                       targets: List[torch.nn.Module],
                                       eigen_smooth: bool = False) -> np.ndarray:
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                               targets,
                               eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(self,
                 input_tensor: torch.Tensor,
                 targets: List[torch.nn.Module] = None,
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False) -> np.ndarray:

        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(
                input_tensor, targets, eigen_smooth)

        return self.forward(input_tensor,
                            targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


class FDCAM(BaseCAM):
    def __init__(self,
                 model,
                 target_layers,
                 threshold,
                 use_cuda=False,
                 reshape_transform=None):

        super(FDCAM, self).__init__(model, target_layers, use_cuda=use_cuda,
                                    reshape_transform=reshape_transform)
        self.model = model
        self.threshold = threshold
        self.target_layers = target_layers

    def minMax(self, tensor):
        maxs = tensor.max(dim=1)[0]
        mins = tensor.min(dim=1)[0]
        return (tensor - mins) / (maxs - mins)

    def scaled(self, tensor):
        maxs = tensor.max(dim=1)[0]
        return (tensor) / (maxs)

    def get_cos_similar_matrix(self, v1, v2):
        num = np.dot(v1, np.array(v2).T)
        denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)
        res = num / denom
        res[np.isnan(res)] = 0
        return res

    def combination(self, scores, grads_tensor):

        grads = self.minMax(grads_tensor)
        scores = self.minMax(scores)

        weights = torch.exp(scores) * grads - 0.5

        return weights

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        with torch.no_grad():
            grads = np.mean(grads, axis=(2, 3))
            BATCH_SIZE = 32

            activation = activations.reshape(activations.shape[1], -1)
            consine = self.get_cos_similar_matrix(activation, activation)
            activation_tensor = torch.from_numpy(activations)

            consine = torch.from_numpy(consine)
            activation = torch.from_numpy(activation)

            record0 = torch.ones(consine.shape).cuda()
            record1 = torch.zeros(consine.shape).cuda()

            for i in range(consine.shape[0]):
                threshold0 = torch.quantile(consine[i, :], self.threshold)
                record1[i, :] = consine[i, :] > threshold0

            record2 = record0 - record1

            if self.cuda:
                activation_tensor = activation_tensor.cuda()
                grad_tensor = torch.from_numpy(grads).cuda()
                record1 = record1.cuda()
                record2 = record2.cuda()

            scores = []
            orig_result = np.float32(self.model(input_tensor)[:, target_category].cpu()).reshape(1, )
            number_of_channels = activation_tensor.shape[1]
            for tensor, category in zip(activation_tensor, target_category):
                batch_tensor = tensor.repeat(BATCH_SIZE, 1, 1, 1)
                for i in range(0, number_of_channels, BATCH_SIZE):
                    batch = batch_tensor * record1[i:i + BATCH_SIZE, :, None, None]  ## on
                    # vgg16, alexnet
                    score = self.model.classifier(torch.flatten(self.model.avgpool(batch), 1))[:, category].cpu().numpy()
                    # resnet50
                    # score = self.model.fc(torch.flatten(self.model.avgpool(batch), 1))[:, category].cpu().numpy()
                    batch = batch_tensor * record2[i:i + BATCH_SIZE, :, None, None]  ## off
                    # vgg16, alexnet
                    score += orig_result - self.model.classifier(torch.flatten(self.model.avgpool(batch), 1))[:, category].cpu().numpy().reshape(BATCH_SIZE, )
                    # resnet50
                    # score += orig_result - self.model.fc(torch.flatten(self.model.avgpool(batch), 1))[:, category].cpu().numpy().reshape(BATCH_SIZE, )
                    scores.extend(score)

            scores = np.float32(scores).reshape(activations.shape[0], activations.shape[1])
            # scores = scores/(2*orig_result)              ## off+on relative 

            scores = torch.tensor(scores).cuda()
            scores = self.combination(scores, grad_tensor).cpu().numpy()

            return scores