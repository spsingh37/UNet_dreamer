# Imports
import pathlib
from transformations import (
    ComposeDouble,
    normalize_01,
    AlbuSeg2d,
    FunctionWrapperDouble,
    create_dense_target,
)
from sklearn.model_selection import train_test_split
from customdatasets import TrackDataset
import torch
import numpy as np
from unet import UNet
from trainer import Trainer
from torch.utils.data import DataLoader
from skimage.transform import resize
import albumentations




def get_filenames_of_path(path: pathlib.Path, ext: str = "*"):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames




# pre-transformations
pre_transforms = ComposeDouble(
    [
        FunctionWrapperDouble(
            resize, input=True, target=False, output_shape=(64, 64, 3)
        ),
        FunctionWrapperDouble(
            resize,
            input=False,
            target=True,
            output_shape=(64, 64),
            order=0,
            anti_aliasing=False,
            preserve_range=True,
        ),
    ]
)

# training transformations and augmentations
transforms_training = ComposeDouble(
    [
        AlbuSeg2d(albumentations.HorizontalFlip(p=0.5)),
        FunctionWrapperDouble(create_dense_target, input=False, target=True),
        FunctionWrapperDouble(
            np.moveaxis, input=True, target=False, source=-1, destination=0
        ),
        FunctionWrapperDouble(normalize_01),
    ]
)

# validation transformations
transforms_validation = ComposeDouble(
    [
        FunctionWrapperDouble(
            resize, input=True, target=False, output_shape=(64, 64, 3)
        ),
        FunctionWrapperDouble(
            resize,
            input=False,
            target=True,
            output_shape=(64, 64),
            order=0,
            anti_aliasing=False,
            preserve_range=True,
        ),
        FunctionWrapperDouble(create_dense_target, input=False, target=True),
        FunctionWrapperDouble(
            np.moveaxis, input=True, target=False, source=-1, destination=0
        ),
        FunctionWrapperDouble(normalize_01),
    ]
)

# random seed
# random_seed = 42

# # split dataset into training set and validation set
# train_size = 0.8  # 80:20 split


# root directory
train_root = pathlib.Path.cwd() / "dataset/train"
val_root = pathlib.Path.cwd() / "dataset/val"
# input and target files
inputs_train = get_filenames_of_path(train_root / "images")
targets_train = get_filenames_of_path(train_root / "targets")

inputs_valid = get_filenames_of_path(val_root / "images")
targets_valid = get_filenames_of_path(val_root / "targets")
# inputs_train, inputs_valid = train_test_split(
#     inputs, random_state=random_seed, train_size=train_size, shuffle=True
# )

# targets_train, targets_valid = train_test_split(
#     targets, random_state=random_seed, train_size=train_size, shuffle=True
# )

# inputs_train, inputs_valid = inputs[:80], inputs[80:]
# targets_train, targets_valid = targets[:80], targets[:80]

# dataset training
dataset_train = TrackDataset(
    inputs=inputs_train,
    targets=targets_train,
    transform=transforms_training,
    use_cache=True,
    pre_transform=pre_transforms,
)

# dataset validation
dataset_valid = TrackDataset(
    inputs=inputs_valid,
    targets=targets_valid,
    transform=transforms_validation,
    use_cache=True,
    pre_transform=pre_transforms,
)

# dataloader training
dataloader_training = DataLoader(dataset=dataset_train, batch_size=2, shuffle=True)

# dataloader validation
dataloader_validation = DataLoader(dataset=dataset_valid, batch_size=2, shuffle=True)

# device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# model
model = UNet(
    in_channels=3,
    out_channels=3,
    n_blocks=4,
    start_filters=32,
    activation="relu",
    normalization="batch",
    conv_mode="same",
    dim=2,
).to(device)

# criterion
criterion = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# trainer
trainer = Trainer(
    model=model,
    device=device,
    criterion=criterion,
    optimizer=optimizer,
    training_dataloader=dataloader_training,
    validation_dataloader=dataloader_validation,
    lr_scheduler=None,
    epochs=5,
    epoch=0,
    notebook=False,
    checkpoint_dir= 'experiment',
    save_frequency=1
)

# start training
training_losses, validation_losses, lr_rates = trainer.run_trainer()

from visual import plot_training

fig = plot_training(
    training_losses,
    validation_losses,
    lr_rates,
    gaussian=True,
    sigma=1,
    figsize=(10, 4),
)

