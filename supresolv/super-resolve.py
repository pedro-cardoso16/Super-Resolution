import argparse

parser = argparse.ArgumentParser(
    prog="super-resolve",
    description="CNN model for super resolution of images",
)

parser.add_argument("file")
parser.add_argument("-u", "--upscale_factor", default=3)
parser.add_argument("-m", "--model", help="load MODEL for super resolution")

subparsers = parser.add_subparsers(help="Sub-type")

# -- train subparser ----------------------------

train_parser = subparsers.add_parser("train")

train_parser.add_argument("folder")
train_parser.add_argument("-u", "--upscale_factor", default=3)
train_parser.add_argument(
    "--save_folder", default="./", help="save models to destination folder"
)
train_parser.add_argument(
    "-e", "--epochs", default=30, help="How many epochs to train the model"
)
train_parser.add_argument("-b", "--batch_size", default=32)


if __name__ == "__main__":
    pass


# def train_model(epochs, folder):
#     epochs: int = 10

#     for epoch in tqdm(range(epochs)):

#         # epoch training
#         for input, target in train_loader:
#             # zeros all gradients
#             optimizer.zero_grad()

#             # propagate input in nn
#             output = model(input)

#             # loss calculation
#             loss: torch.Tensor = mse_loss(output, target)
#             loss.backward()

#             # parameters update
#             optimizer.step()

#         # save the model for each epoch
#         torch.save(
#             model,
#             os.path.join(
#                 folder,
#                 f"model-{epoch+1:03d}.pth",
#             ),
#         )