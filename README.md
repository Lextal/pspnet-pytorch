# pspnet-pytorch
PyTorch implementation of PSPNet segmentation network


### Original paper

 [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)
 
### Details

This is a slightly different version - instead of direct 8x upsampling at the end I use three consequitive upsamplings for stability. 

### Feature extraction

Using pretrained weights for extractors - improved quality and convergence dramatically.

Currently supported:

* SqueezeNet
* DenseNet-121
* ResNet-18
* ResNet-34
* ResNet-50
* ResNet-101
* ResNet-152

Planned:

* DenseNet-169
* DenseNet-201

### Usage 

To follow the training routine in train.py you need a DataLoader that yields the tuples of the following format:
(Bx3xHxW FloatTensor x, BxHxW LongTensor y, BxN LongTensor y\_cls) where
x - batch of input images,
y - batch of groung truth seg maps,
y\_cls - batch of 1D tensors of dimensionality N: N total number of classes, 
y\_cls[i, T] = 1 if class T is present in image i, 0 otherwise