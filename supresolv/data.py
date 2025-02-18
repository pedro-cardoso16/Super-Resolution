from os.path import exists, join, basename
from os import makedirs, remove
import PIL.Image
from six.moves import urllib
import tarfile
import torchvision
import PIL
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import torchvision.transforms.functional
import numpy as np
from .dataset import ImageSet


def download_bsd300(dest="dataset"):
    """
    Returns:
        Path to folder with dataset
    """
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        makedirs(dest)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, "wb") as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose(
        [
            CenterCrop(crop_size),  # crop the image to fit size
            Resize(crop_size // upscale_factor),  # reduces the resolution
            ToTensor(),
        ]
    )


def target_transform(crop_size):
    return Compose(
        [
            CenterCrop(crop_size),  # crop image to fit size
            ToTensor(),
        ]
    )


def get_training_set(upscale_factor) -> ImageSet:
    """
    Returns:
        The training `ImageSet`
    """
    root_dir = download_bsd300()
    train_dir = join(root_dir, "train")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return ImageSet(
        train_dir,
        input_transform=input_transform(crop_size, upscale_factor),
        target_transform=target_transform(crop_size),
    )


def get_test_set(upscale_factor):
    root_dir = download_bsd300()
    test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return ImageSet(
        test_dir,
        input_transform=input_transform(crop_size, upscale_factor),
        target_transform=target_transform(crop_size),
    )


def image_to_input(filepath: str) -> tuple:
    """
    Transforms the image from filepath to input image the model input should be 
    the channel \'y\'
    """
    img = PIL.Image.open(filepath).convert("YCbCr")
    y, cb, cr = img.split()
    y = torchvision.transforms.functional.to_tensor(y)

    return y, cb, cr


def output_to_image(out, cb, cr) -> PIL.Image.Image:
    """
    Convert model output to an image

        :param out: direct output of the model 
        :param cb: cb channels
        :param cr: cr channels
    """
    out_img_y = out[0].detach().numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y[0]), mode="L")

    out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img = PIL.Image.merge("YCbCr", [out_img_y, out_img_cb, out_img_cr]).convert(
        "RGB"
    )

    return out_img

def downsample_image(
    filepath: str, out: str = None, factor: int = 3
) -> PIL.Image.Image:
    """
    Pixelizes an image by downsampling it to a specified factor and then re-scaling
    it to its original size.

        :param filepath: Path to the input image file.
        :type filepath: str
        :param out: Path where the output pixelized image will be saved.
        :type out: str
        :param factor: Factor by which to downscale the image before re-scaling. Defaults to 3.
        :type factor: int
    """
    image = PIL.Image.open(filepath).copy()
    w, h = image.size

    # convolve by gaussian filter
    image = image.filter(PIL.ImageFilter.GaussianBlur)

    # downsample image
    image = image.resize((w // factor, h // factor), PIL.Image.Resampling.BICUBIC)

    if out:
        image.save(out)

    return image


def image_to_input(filepath) -> tuple:
    """
    Transform the image from filepath to input image the model input should be
    the channel \'y\'
    """
    img = PIL.Image.open(filepath).convert("YCbCr")
    y, cb, cr = img.split()

    img_to_tensor = torchvision.transforms.ToTensor()
    input_y = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])

    return input_y, cb, cr


def output_to_image(out, cb, cr) -> PIL.Image.Image:
    """Convert model output to an image"""
    out_img_y = out[0].detach().numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y[0]), mode="L")

    out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img = PIL.Image.merge("YCbCr", [out_img_y, out_img_cb, out_img_cr]).convert(
        "RGB"
    )

    return out_img
