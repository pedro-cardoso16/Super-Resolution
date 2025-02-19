import torch
import torchvision.transforms.functional as F

from torcheval.metrics.functional import peak_signal_noise_ratio, mean_squared_error
from .data import *
from .dataset import *
from skimage.metrics import structural_similarity


def eval_model(*imgs: str, upscale_factor: int, model: str | nn.Module) -> list:
    """
    Evaluate model with images

    :param imgs: path/to/images
    :param upscale_factor: how much the model will upscale the image
    :param model: path/to/model.pth or model itself
    """

    if type(model) == str:  # load model from filepath
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
        target = F.center_crop(target, (crop_size_h, crop_size_v))

        img_in = downsample_image(img)

        input, cb, cr = image_to_input(img_in.convert('YCbCr'))

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

            psnr = peak_signal_noise_ratio(
                y.view(crop_size_h, crop_size_v),
                target.view(crop_size_h, crop_size_v),
            ).item()

            mse = mean_squared_error(
                y.view(crop_size_h, crop_size_v),
                target.view(crop_size_h, crop_size_v),
            ).item()

            # print(i.size, target_image.size)
            a = np.array(i.convert("RGB")).swapaxes(0, -1)
            b = np.array(target_image.convert("RGB")).swapaxes(0, -1)

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
