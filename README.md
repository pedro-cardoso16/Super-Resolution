# Super Resolution

This code implements the ESPCN based on the article: 
and the pytorch example:
- [PyTorch github example](https://github.com/pytorch/examples/tree/main/super_resolution)
- [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158) 

## Getting Started
In order to train the model you can run the following command in the project directory
```shell
python3 super-resolve train dataset/BSDS300/images/train --batch-size 32 --epoch 30 -S .models --use-cuda 
```
If no training set is specified, [BSDS300](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) will be downloaded.

### Applying the model
Once the model is trained you can excute it using the following command:
```shell
python3 super-resolve <image> -m <model> -o <output-image>
```




<!-- 
https://data.vision.ee.ethz.ch/cvl/DIV2K/ 
-->