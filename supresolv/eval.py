import torch
import torchvision.transforms.functional as F

from torcheval.metrics.functional import peak_signal_noise_ratio, mean_squared_error
from .data import *
from .dataset import *


def eval_model(*imgs: str, upscale_factor: int, model: str | nn.Module) -> dict:

    if type(model) == str:
        # load model from filepath
        model: nn.Module = torch.load(
            model,
            weights_only=False,
        )

    model.eval()

    for img in imgs:
        img_original = PIL.Image.open(img).copy()
        x = load_img(img)

        crop_size_h, crop_size_v = list(
            map(lambda size: calculate_valid_crop_size(size, upscale_factor), x.size)
        )

        target = F.to_tensor(load_img(img))
        target = F.center_crop(target, (crop_size_h, crop_size_v))

        img_in = downsample_image(img)

        input, cb, cr = image_to_input(fp_in)
        output = model(input)
        img_out = output_to_image(output, cb, cr)

        psnr = peak_signal_noise_ratio(
            target.view(crop_size_h, crop_size_v), output.view(crop_size_h, crop_size_v)
        ).item()
        mse = mean_squared_error(
            target.view(crop_size_h, crop_size_v), output.view(crop_size_h, crop_size_v)
        ).item()

        img_bicubic = img_in.resize(img_out.size, PIL.Image.Resampling.BICUBIC)
        img_nearest = img_in.resize(img_out.size, PIL.Image.Resampling.NEAREST)

        performance: dict = {
            "psnr": psnr,
            "mse": mse,
        }
