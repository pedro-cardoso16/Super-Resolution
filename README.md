# Super Resolution

This code implements the ESPCN based on the article: https://arxiv.org/abs/1609.05158 
and the pytorch example: 

## Getting Started
In order to train the model you can run the following command in the project directory
```shell
python3 super-resolve train ./dataset/BSDS300/images/train --batch-size 32 --epoch 30 -S .models --use-cuda 
```
If no training set is specified, **BSDS300** will be downloaded.

### Applying the model
Once the model is trained you can excute it using the following command:
```shell
python3 super-resolve <image> -m <model> -o <output-image>
```

![butterfly](source/images/image.png)



<!-- 
https://data.vision.ee.ethz.ch/cvl/DIV2K/ 
-->