import os
import torch

from argparse import ArgumentParser

from utils import Model
from dataloader import build_dataloader


# *------------------------------------------------------------------------------------------------------------------------------------------------------*

parser = ArgumentParser()

# Input data parameters
parser.add_argument(
    "--dataset_name",
    type=str,
    default="kits23",
    help="Name of the dataset which should be equal to the folder name in the folder <data>.",
)
# 2D: opg_binary, opg_color_binary, opg_multiclass, frs_binary, cityscapes
# 3D: crs_sinus_maxillaris, crs_multiclass, ct_82 (binary), word, word_pancreas

parser.add_argument(
    "--dataset_path",
    type=str,
    default="data/",
    help="Path for the images and masks for which the U-Net is used.",
)
# rest of the path is automatically filled with dataset_name, train/test/validation, images/masks, etc.

parser.add_argument(
    "--dim",
    type=str,
    default="2d",
    help="Number of dimensions of the data and the U-Net. Possible values: 2d or 3d",
)
parser.add_argument(
    "--color",
    type=str,
    default="grey",
    help="Color of the input images. Possible values: grey (greyscale) and rgb (colored).",
)

parser.add_argument(
    "--amount_classes",
    type=int,
    default=4,
    help="Number of classes + Background (e.g. 4 classes + Background = 5). For binary: 1.",
)
# 33 for opg, 11 for crs, 17 for word, 34 for cityscapes

parser.add_argument(
    "--resize",
    type=int,
    default=(512, 512),
    help="Size for the input images. Depending on the trained network parameters. Format: (height, width)",
)
# opg_binary/multiclass: 560, 992
# opg_color_binary: 734, 1100
# ct_82 :  100, 100
# crs: 100, 100
# word: 70, 102
# cityscapes: 1024 x 2048 -> 512, 1024
# ham (450, 600),
# kits (512, 512)

parser.add_argument(
    "--aug_prob", type=int, default=0.0, help="Augmentation probability."
)
# We did not use any augmentation for the paper

# Model hyperparameters
parser.add_argument(
    "--model",
    type=str,
    default="vanilla_unet",
    help="Choose the model. Possible values: vanilla_unet",
)

parser.add_argument(
    "--learning_rate", type=float, default=3e-4, help="Choose the learning rate."
)
# 3e-3 for 2D

parser.add_argument(
    "--num_epochs",
    type=int,
    default=10000,
    help="Choose the number of epochs for the training.",
)
# max. 100 for 2D
# min. 300 for 3D
parser.add_argument("--batch_size", type=int, default=1, help="Choose the batch size.")
# currently only batch size of 1 is supported and implemented due to GPU memory limitations

# Output data parameters
parser.add_argument(
    "--result_folder", type=str, default="results/", help="Choose the result folder."
)

# Operating Systems
parser.add_argument(
    "--gpu", type=str, default="0", help="Choose the GPU for the training."
)
parser.add_argument(
    "--num_workers", type=int, default=4, help="Choose the number of workers."
)

args = parser.parse_args()

# *------------------------------------------------------------------------------------------------------------------------------------------------------*

# Set the visible and used GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
print(f"DEVICE CHOICE: {device}")

# Set the result folder in more detail
path_save_folder = f"{args.result_folder}{args.model}_{args.dim}/"
path_save_model = f"{args.result_folder}{args.model}_{args.dim}/models/"

# Set resize argument
transform_resize = {"resize": args.resize}

base_path = f"{args.dataset_path}{args.dataset_name}/"
train_path, validation_path, test_path = (
    f"{base_path}train/",
    f"{base_path}validation/",
    f"{base_path}test/",
)
classes = "multiclass" if args.amount_classes != 1 else "binary"
# Set Data Loaders for train, val and test
train_loader = build_dataloader(
    dir_root=train_path,
    transform={**transform_resize},
    num_workers=args.num_workers,
    batch_size=args.batch_size,
    return_filenames=True,
    shuffle=True,
    dataset_color=args.color,
    classes=classes,
)
validation_loader = build_dataloader(
    dir_root=validation_path,
    transform={**transform_resize},
    num_workers=args.num_workers,
    batch_size=1,
    return_filenames=True,
    shuffle=False,
    dataset_color=args.color,
    classes=classes,
)
test_loader = build_dataloader(
    dir_root=test_path,
    transform={**transform_resize},
    num_workers=args.num_workers,
    batch_size=1,
    return_filenames=True,
    shuffle=False,
    dataset_color=args.color,
    classes=classes,
)

# Create, Build and Train model
model = Model(
    train_loader=train_loader,
    validation_loader=validation_loader,
    test_loader=test_loader,
    modelname=f"{args.model}_{args.dim}",
    path_model=path_save_model,
    device=device,
    learning_rate=args.learning_rate,
    nr_epochs=args.num_epochs,
    dataset_name=args.dataset_name,
    dataset_color=args.color,
    nr_classes=classes,
    classes=args.amount_classes,
    batch_size=args.batch_size,
)

model.train_val_test()
