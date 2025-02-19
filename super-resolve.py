import argparse
import numpy as np
import matplotlib.pyplot as plt
import os, sys, re

from supresolv.data import train_model, process_image

# -- Main parser -------------------------------------------------------------

parser = argparse.ArgumentParser(
    prog="super-resolve",
    description="CNN model for super resolution of images",
)
parser.add_argument("image", help="IMAGE file to super resolve")
parser.add_argument("-m", "--model", help="load MODEL for super resolution")
parser.add_argument("-o", "--output", help="OUTPUT image")

subparsers = parser.add_subparsers(help="Sub-type", dest="command")

# -- Train subparser ---------------------------------------------------------

train_parser = subparsers.add_parser("train")

train_parser.add_argument("train_folder")
train_parser.add_argument("-u", "--upscale-factor", default=3)
train_parser.add_argument(
    "-S",
    "--save-folder",
    default="./",
    type=str,
    help="save models to destination folder",
)
train_parser.add_argument(
    "-e", "--epochs", default=30, type=int, help="How many epochs to train the model"
)
train_parser.add_argument("-b", "--batch-size", default=256, type=int)
train_parser.add_argument("--lr", "--learning-rate", default=0.001, type=float)

train_parser.add_argument("--use-cuda", action="store_true")


# -- Evaluation subparser ----------------------------------------------------

# eval_parser = subparsers.add_parser("eval")

if __name__ == "__main__":

    match sys.argv[1]:
        case "train":
            args, unknown = train_parser.parse_known_args()

            train_model(
                args.epochs,
                args.upscale_factor,
                args.train_folder,
                args.save_folder,
                device=("cuda" if args.use_cuda else "cpu"),
                batch_size=args.batch_size,
            )

        case _: # default case
            args, unknown = parser.parse_known_args()

            process_image(args.image, args.model, args.output)


    # if args.command == 'train':
    #     print('train called')

    # args, unknown = train_parser.parse_known_args()

# import argparse
# import logging
# import os
# import sys
# import torch
# from supresolv.data import train_model, process_image

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
# )

# # -- Main parser -------------------------------------------------------------
# parser = argparse.ArgumentParser(
#     prog="super-resolve",
#     description="CNN model for super resolution of images",
# )
# parser.add_argument("image", nargs="?", help="IMAGE file to super resolve")
# parser.add_argument(
#     "-m", "--model", required=False, help="Load MODEL for super resolution"
# )
# parser.add_argument("-o", "--output", help="OUTPUT image")

# subparsers = parser.add_subparsers(help="Sub-type", dest="command")

# # -- Train subparser ---------------------------------------------------------
# train_parser = subparsers.add_parser("train", help="Train a super-resolution model")
# train_parser.add_argument("train_folder", type=str, help="Path to the training dataset")
# train_parser.add_argument(
#     "-u", "--upscale-factor", type=int, default=3, help="Upscale factor"
# )
# train_parser.add_argument(
#     "-S",
#     "--save-folder",
#     default="./",
#     type=str,
#     help="Save models to destination folder",
# )
# train_parser.add_argument(
#     "-e", "--epochs", type=int, default=30, help="Number of epochs for training"
# )
# train_parser.add_argument(
#     "-b", "--batch-size", type=int, default=256, help="Batch size"
# )
# train_parser.add_argument(
#     "--lr", "--learning-rate", type=float, default=0.001, help="Learning rate"
# )
# train_parser.add_argument(
#     "--use-cuda", action="store_true", help="Use CUDA if available"
# )

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         parser.print_help()
#         sys.exit(1)

#     args = parser.parse_args()
#     device = torch.device(
#         "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
#     )

#     if args.command == "train":
#         logging.info("Starting training...")
#         train_model(
#             epochs=args.epochs,
#             upscale_factor=args.upscale_factor,
#             train_folder=args.train_folder,
#             save_folder=args.save_folder,
#             device=device,
#             batch_size=args.batch_size,
#         )
#     else:
#         if not args.image:
#             logging.error(
#                 "Error: No image file provided. Please specify an image file."
#             )
#             sys.exit(1)
#         if not args.model:
#             logging.error(
#                 "Error: No model file provided. Use -m or --model to specify the model path."
#             )
#             sys.exit(1)

#         logging.info("Processing image...")
#         process_image(args.image, args.model, args.output)
#         logging.info("Image processing complete.")
