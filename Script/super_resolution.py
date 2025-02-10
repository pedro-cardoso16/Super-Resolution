import PIL.Image
import torch, torchvision
import PIL
import numpy as np


def upscale_image(filepath: str, output_filepath: str, model: str) -> None:
    """
    Upscale the image

    :param filepath: path/to/image
    :type filepath: str
    :param output_filepath: path/to/output/image
    :param model: path/to/model.pth
    """
    # image loading
    img = PIL.Image.open(filepath).convert("YCbCr")
    y, cb, cr = img.split()

    model = torch.load(model, weights_only=False)
    img_to_tensor = torchvision.transforms.ToTensor()
    input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])

    # if opt.cuda:
    #     model = model.cuda()
    #     input = input.cuda()

    out = model(input)
    out = out.cpu()
    out_img_y = out[0].detach().numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y[0]), mode="L")

    out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img = PIL.Image.merge("YCbCr", [out_img_y, out_img_cb, out_img_cr]).convert(
        "RGB"
    )

    out_img.save(output_filepath)
