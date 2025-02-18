import torch, torchvision
import os
from torcheval.metrics.functional import peak_signal_noise_ratio

from .data import *
from .dataset import *


def eval_model(filepath: str, upscale_factor, model: any):
    crop_size = calculate_valid_crop_size(256, upscale_factor)
    it = input_transform(crop_size, 3)
    tt = target_transform(crop_size)

    input = it(load_img(filepath))
    target = tt(load_img(filepath))

    # target = torchvision.transforms.functional.to_tensor(load_img(filepath))
    # target = torchvision.transforms.functional.center_crop(
    #     target, (crop_size, crop_size)
    # )

    img_original = PIL.Image.open(img).copy()
    img_in = downsample_image(img, out="tmp/my_downsampled_image.jpg")
    fp_in = "tmp/my_downsampled_image.jpg"

    # img_in = PIL.Image.open("my_downsampled_image.jpg")
    # img_in = img_in.convert("YCbCr").copy()

    input, cb, cr = image_to_input(fp_in)
    out = model(input)
    img_out = output_to_image(out, cb, cr)

    psnr = peak_signal_noise_ratio(
        target.view(crop_size, crop_size), out.view(crop_size, crop_size)
    ).item()

    img_in_upscaled = img_in.resize(img_out.size, PIL.Image.BICUBIC)

    img_out.save("tmp/asd.jpg")
