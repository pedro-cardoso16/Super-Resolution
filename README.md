# Super Resolution
This code implements the based on the article

$x^2$

https://data.vision.ee.ethz.ch/cvl/DIV2K/

## Getting Started
In order to train the model you can run the following command in the project directory
```shell
python3 super-resolve train ./dataset/BSDS300/images/train --batch-size 32 --epoch 30 -S .models --use-cuda 
```
Once the model is trained you can excute it using
```shell
python3 super-resolve <image> -m <model> -o <output-image>
```

## 
