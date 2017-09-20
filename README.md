# pspnet-pytorch
PyTorch implementation of PSPNet segmentation network


### Original paper

 [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)
 
### Details

This is a slightly different version - instead of direct 8x upsampling at the end I use three consequitive upsamplings for stability. 

### Feature extraction

Using pretrained weights for extractors - improved quality and convergence dramatically.

Currently supported:

* ResNet-18
* ResNet-34
* ResNet-50
* ResNet-101
* ResNet-152

Planned:

* DenseNet-121
* DenseNet-169
* DenseNet-201
