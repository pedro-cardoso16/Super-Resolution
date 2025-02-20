import torch
import torchvision.transforms.functional as F
from torcheval.metrics.functional import peak_signal_noise_ratio, mean_squared_error

from skimage.metrics import structural_similarity

from .data import *
from .dataset import *


def eval_model(*imgs: str, upscale_factor: int, model: str | nn.Module) -> list:
    """
    Evaluate model with images

    :param imgs: path/to/images
    :param upscale_factor: how much the model will upscale the image
    :param model: path/to/model.pth or model itself
    """

    if isinstance(model, str):  # load model from filepath
        model: nn.Module = torch.load(
            model,
            weights_only=False,
        )
    model.to("cpu")
    model.eval()

    result = []

    for img in imgs:
        result_img = {}

        img_original = PIL.Image.open(img).copy()

        crop_size_h, crop_size_v = list(
            map(
                lambda size: calculate_valid_crop_size(size, upscale_factor),
                img_original.size,
            )
        )

        target = F.to_tensor(load_img(img))
        target = F.center_crop(target, (crop_size_v, crop_size_h))

        img_in = downsample_image(img,None, factor=upscale_factor)

        input, cb, cr = image_to_input(img_in.convert("YCbCr"))

        # input.to('cpu')
        output: torch.Tensor = model(input)
        img_out = output_to_image(output, cb, cr)

        img_bicubic = img_in.resize(img_out.size, PIL.Image.Resampling.BICUBIC)
        img_nearest = img_in.resize(img_out.size, PIL.Image.Resampling.NEAREST)

        target_image = output_to_image(
            target.reshape(1, -1, target.shape[-1], target.shape[-2]), cb, cr
        )

        # print(target_image.size, img_out.size)
        result_img.update({"input": {"img": img_in}})  # add target image
        result_img.update({"target": {"img": img_original}})  # add target image

        for i, name in zip(
            (img_bicubic, img_nearest, img_out), ("bicubic", "nearest", "espcn")
        ):
            img = i.convert("YCbCr")
            y, _, _ = img.split()
            y = F.to_tensor(y)
            y *= 255
            y = y.clip(0, 255)
            y = y.view(1, -1, target.shape[-2], target.shape[-1])
            y = y[0]

            # print(
            #     y.shape, target.shape
            # )
            psnr = peak_signal_noise_ratio(
                y.view(crop_size_h, crop_size_v),
                target.reshape(crop_size_h, crop_size_v),
            ).item()

            # mse = mse_loss(
            #     y.view(crop_size_h, crop_size_v),
            #     target.view(crop_size_h, crop_size_v),
            # )
            mse = mean_squared_error(
                y.view(crop_size_h, crop_size_v),
                target.reshape(crop_size_h, crop_size_v),
            ).item()

            # print(i.size, target_image.size)
            a = np.array(i.convert("RGB")).swapaxes(0, -1)
            b = np.array(target_image.convert("RGB")).swapaxes(0, -1).swapaxes(-1,-2)

            print(a.shape, b.shape)

            ssim = 0
            for index in range(3):
                ssim += structural_similarity(a[index], b[index]) / 3
            # print(a.shape,b.shape)

            performance = {
                "img": i,
                "psnr": psnr,
                "mse": mse,
                "ssim": ssim,
            }

            result_img.update({name: performance})

        result.append(result_img)

    return result
