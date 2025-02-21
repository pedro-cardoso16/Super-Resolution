# Super Resolution

This repository implements Super-Resolution using the ESPCN (Efficient Sub-Pixel Convolutional Neural Network) model, based on the following references:

- [PyTorch GitHub Example](https://github.com/pytorch/examples/tree/main/super_resolution)
- [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network (Paper)](https://arxiv.org/abs/1609.05158)

## Getting Started

### Prerequisites
Ensure you have Python and PyTorch installed. You can install dependencies with:
```sh
pip install -r requirements.txt
```

### Training the Model
To train the model, run the following command from the project directory:
```sh
python3 super-resolve.py train dataset/BSDS300/images/train --batch-size 32 --epoch 30 -S .models --use-cuda 
```
- If no training dataset is provided, the [BSDS300 dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) will be downloaded automatically.
- Adjust batch size, epoch count, and other parameters as needed.

### Applying the Model
Once the model is trained, apply it to an image using:
```sh
python3 super-resolve.py <image> -m <model> -o <output-image>
```
- `<image>`: Path to the input image.
- `-m <model>`: Path to the trained model file.
- `-o <output-image>`: Path to save the super-resolved output image.

## Notes
- Ensure CUDA is enabled (`--use-cuda`) for faster training and inference if running on a GPU.
- Experiment with different hyperparameters for better results.
- Contributions and improvements are welcome!
