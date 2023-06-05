import torch
import os

from argparse import ArgumentParser

# local imports
from dataloader import build_dataloader
from utils import (
    set_encoder_decoder,
    create_model,
    create_save_folders,
    Model_Inference,
)


# *------------------------------------------------------------------ Arguments --------------------------------------------------------------------*
parser = ArgumentParser()

parser.add_argument("--dataset_name", type=str, default="cityscapes")
parser.add_argument("--test_path", type=str, default="data/cityscapes/test/")

parser.add_argument(
    "--color",
    type=str,
    default="rgb",
    help="possible values: grey (greyscale) and color (rgb).",
)
parser.add_argument(
    "--amount_classes",
    type=int,
    default=34,  # 33 for opg, 17 for word, 2 for binary, 34 for cityscapes
    help="Only needed for mutliple class segmentation. Number of classes + Background (e.g. 4 classes + Background = 5).",
)

parser.add_argument("--resize", type=int, default=(512, 1024))  # height, width
# opg multiclass: 560, 992
# word multiclass: 70, 102
# cityscapes: 512, 1024

# Model choice
parser.add_argument(
    "--model", type=str, default="vanilla_unet", help="possible values: vanilla_unet"
)
parser.add_argument(
    "--encoder_depth",
    type=int,
    default=5,
    help="possible values up to 5. Depth 1 = [1, 64] for the encoder. Depth 2 = [1, 64, 128]. And so on.",
)

# Seg-Grad CAM or Seg-HiRes-Grad CAM
parser.add_argument(
    "--cam",
    type=str,
    default="gradcam",
    help="Decide whether to use Seg-Grad CAM (gradcam) or Seg-HiRes-Grad CAM (hirescam).",
)
parser.add_argument(
    "--px_set",
    type=str,
    default="class",
    help="Decide whether to use image, class, point or zero. Best results with class.",
)
parser.add_argument(
    "--px_set_point",
    type=str,
    default=(300, 300),
    help="X, y coordinates if px_set is chosen to be point.",
)

# Output data parameters
parser.add_argument("--result_folder", type=str, default="results/")
parser.add_argument("--extension", type=str, default=".pdf")  # .png

# Operating System
parser.add_argument("--gpu", type=str, default="0")

args = parser.parse_args()


# *--------------------------------------------------------------- Argument Parsing ----------------------------------------------------------------*

# Set resize argument
transform_resize = {"resize": args.resize}

model_name = f"{args.model}"

# Set the result folder in more detail
if args.cam == "gradcam":
    path_save_folder_grad_cam = f"{args.result_folder}{model_name}/visualizations/{args.dataset_name}/seg_grad_cam/gradcam_{args.px_set}/"
if args.cam == "hirescam":
    path_save_folder_grad_cam = f"{args.result_folder}{model_name}/visualizations/{args.dataset_name}/seg_grad_cam/hirescam_{args.px_set}/"

path_load_model = f"{args.result_folder}{model_name}/models/{args.dataset_name}/model"

# Set the visible and used GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE CHOICE: {device}")

classes = "multiclass" if args.amount_classes != 1 else "binary"

# Set encoder and decoder values
encoder_channels, decoder_channels = set_encoder_decoder(
    color=args.color, amount_classes=args.amount_classes, depth=5
)

# Create the model
model, path_save_folder_grad_cam, path_load_model = create_model(
    modelname=model_name,
    encoder_channels=encoder_channels,
    decoder_channels=decoder_channels,
    path_save_folder_grad_cam=path_save_folder_grad_cam,
    path_load_model=path_load_model,
)

# validate that save folders are existing
create_save_folders(path_save_folder_grad_cam=path_save_folder_grad_cam)

# *------------------------------------------------------------------ Main -----------------------------------------------------------------------*
if __name__ == "__main__":
    # Create the dataloader
    dataloader = build_dataloader(
        dir_root=args.test_path,
        transform={**transform_resize},
        num_workers=1,
        batch_size=1,
        return_filenames=True,
        shuffle=False,
        dataset_color=args.color,
        classes=classes,
    )
    # Load the respective model
    Model_Class = Model_Inference(
        model=model,
        model_name=model_name,
        dataset_name=args.dataset_name,
        path_model_state_dict=path_load_model,
        device=device,
        dataloader=dataloader,
        amount_classes=args.amount_classes,
        cam_type=args.cam,
        pixel_set=args.px_set,
        pixel_set_point=args.px_set_point,
        path_save_folder_grad_cam=path_save_folder_grad_cam,
    )

    Model_Class.inference(cam=True)