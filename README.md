# pspnet-pytorch
PyTorch implementation of PSPNet segmentation network


### Original paper

 [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)
 
### Details

This is a slightly different version - instead of direct 8x upsampling at the end I use three consequitive upsamplings for stability. 
Augmentations described in the paper are not included, but probably will be added later.
