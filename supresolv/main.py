from tqdm import tqdm





# for epoch in tqdm(range(epochs)):

#     # epoch training
#     for input, target in train_loader:
#         # zeros all gradients
#         optimizer.zero_grad()

#         # propagate input in nn
#         output = model(input)

#         # loss calculation
#         loss: torch.Tensor = mse_loss(output, target)
#         loss.backward()

#         # parameters update
#         optimizer.step()

#     # save state for each epoch
#     torch.save(
#         model,
#         os.path.join(
#             ".model_parameters/",
#             f"model-epoch_{epoch+1:03d}-upscale_factor_{upscale_factor:03d}.pth",
#         ),
#     )