import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision, torcheval
import os, sys, re

from supresolv.data import train_model


from tqdm import tqdm
from torch.utils.data import DataLoader

# -- Main parser -------------------------------------------------------------

parser = argparse.ArgumentParser(
    prog="super-resolve",
    description="CNN model for super resolution of images",
)

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

        case _:
            args, unknown = parser.parse_known_args()

    # if args.command == 'train':
    #     print('train called')

    # args, unknown = train_parser.parse_known_args()
