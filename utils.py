import cv2
import os
import torch
import csv

import numpy as np
import torch.nn as nn

# local 2d imports
from networks import Vanilla_UNet_2d

# Metric imports
from metrics import f1_score, iou_score
from monai.losses.dice import DiceLoss as mDiceLoss

import torch.nn.functional as F
from PIL import Image


# SciencePlots for Matplot Lib
# plt.style.use("science")
# larger plt fontsize
# plt.rcParams.update({'font.size': 10})


def set_encoder_decoder(color, amount_classes, depth=5):
    start_val = 1 if color == "grey" else 3

    available_depths_encoder = [start_val, 64, 128, 256, 512, 1024]
    available_depths_decoder = [1024, 512, 256, 128, 64, amount_classes]

    # Set encoder and decoder values for the network
    encoder_channels = available_depths_encoder[: depth + 1]
    decoder_channels = available_depths_decoder[5 - depth :]

    return encoder_channels, decoder_channels


def create_save_folders(path_save_folder_grad_cam):
    if not os.path.exists(path_save_folder_grad_cam):
        os.makedirs(path_save_folder_grad_cam)


def create_model(
    modelname,
    encoder_channels,
    decoder_channels,
    path_save_folder_grad_cam,
    path_load_model,
):
    if modelname == "vanilla_unet":
        model = Vanilla_UNet_2d(
            encoder_channels=encoder_channels, decoder_channels=decoder_channels
        )
    else:
        print("Err: Model not defined")

    return model, path_save_folder_grad_cam, path_load_model


class Model_Inference:
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        dataset_name: str,
        path_model_state_dict: str,
        device: torch.device,
        dataloader: torch.utils.data.DataLoader,
        amount_classes: int,
        cam_type=None,
        pixel_set=None,
        pixel_set_point=None,
        path_save_folder_grad_cam=None,
        path_save_folder_figures=None,
        show_progress=True,
    ):
        """Inference class for doing CAM and/or metric calculations.

        Args:
            model (nn.Module): Model for doing inference
            model_name (str): modelname
            dataset_name (str): dataset name
            path_model_state_dict (str): Path for the model's state dict
            device (torch.device): Device for doing predictions on
            dataloader (torch.utils.data.DataLoader): DataLoader
            amount_classes (int): Number of classes to predict (x classes + 1 background)
            cam_type (_type_, optional): Cam Type to choose (gradcam or proposed hirescam (for segmentation and not classification)).
                                        Defaults to None.
            path_save_folder_grad_cam (_type_, optional): Save folder for grad cam images.
                                                        Defaults to None.
            path_save_folder_figures (_type_, optional): Save folder for figures of metrics.
                                                        Defaults to None.
            show_progress (bool, optional): Show the current file name and how many others are left.
                                            Defaults to True.
        """

        self.model = model
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.model_state_dict = path_model_state_dict
        self.device = device
        self.dataloader = dataloader
        self.amount_classes = amount_classes
        self.cam_type = cam_type
        self.pixel_set = pixel_set
        self.pixel_set_point = pixel_set_point
        self.show_progress = show_progress

        if path_save_folder_grad_cam is not None:
            self.path_save_folder_grad_cam = path_save_folder_grad_cam

        if path_save_folder_figures is not None:
            self.path_save_folder_figures = path_save_folder_figures

        if self.cam_type is not None:
            self.cam = CAM(
                cam_type=cam_type,
                pixel_set=self.pixel_set,
                pixel_set_point=self.pixel_set_point,
                model=self.model,
                path_model_state_dict=self.model_state_dict,
                device=self.device,
                dataloader=self.dataloader,
                path_save_folder_grad_cam=self.path_save_folder_grad_cam,
                amount_classes=self.amount_classes,
            )

    def prepare_model(self):
        self.model.load_state_dict(torch.load(self.model_state_dict))
        self.model.grad_cam = True
        self.model.to(self.device)

    def predict(self, image: torch.tensor):
        self.model.zero_grad()
        self.model.eval()
        # get the image from the loader
        image = image.to(self.device)
        # make a prediction with the model
        prediction = self.model(image)

        return prediction

    def inference(self, cam=True):
        self.prepare_model()

        for i, (image, ground_truth, filenames) in enumerate(self.dataloader):
            torch.cuda.empty_cache()
            if self.show_progress:
                print(
                    f"Currently at {filenames[0]} (number {i+1} out of {len(self.dataloader)+1})"
                )

            # do inference prediction with the model
            prediction = self.predict(image)

            # CAM
            if cam:
                # iterate through all classes
                for cl in range(self.amount_classes):
                    self.cam.inference_cam(
                        prediction=prediction,
                        image=image.to(self.device),
                        filenames=filenames,
                        backward_class=cl,
                    )


class CAM:
    """Script for re-implementation of Seg-Grad-Cam originally proposed by Vinogradova et al..
        Further, re-implementation of HiResCAM originally proposed by Draelos et al.

        Finally, proposed combination of HiResCAM with Seg-Grad-Cam which results in HiRes-Seg-Grad-CAM
        for more explainable results compared to Seg-Grad-Cam.

    Citations:
        Kira Vinogradova, Alexandr Dibrov, & Eugene W. Myers (2020). Towards Interpretable Semantic Segmentation via
        Gradient-weighted Class Activation Mapping. In Proceedings of the AAAI Conference on Artificial Intelligence.

        Draelos, R., & Carin, L.. (2020). Use HiResCAM instead of Grad-CAM for faithful explanations of convolutional
        neural networks.

    """

    def __init__(
        self,
        cam_type: str,
        pixel_set: str,
        pixel_set_point: None,
        model: nn.Module,
        path_model_state_dict: str,
        device: torch.device,
        dataloader: torch.utils.data.DataLoader,
        path_save_folder_grad_cam: str,
        amount_classes: int,
    ):
        """Initialize CAM.

        Args:
            cam_type (str): "gradcam" for segmentation CAM method proposed by Vinogradova et al.
                            "hirescam" for combination of gradcam (segmentation) and hirescam (classification) for
                            segmentation purposes (HiRes-Seg-Grad-CAM).
            pixel_set (str): Define the pixel set as desribed by Vinogradova et al.
                            "image" for the whole image
                            "class" for the set of pixels which represent the respective class (e.g. class 3 -> pixels for a dog)
                            "point" for a specific x,y coordinate in the picture. Variable "pixel_set_point" needed then
                            "zero" for an empty pixel set. This can be used as a sanity check.
            pixel_set_point (tuple): x,y coordinate for a specific point in the image. E.g. (100, 150)
            model (nn.Module): Model which should be used for CAM. The model has to be prepared for CAM and needs some additional
                                functions.
            path_model_state_dict (str): Path to the model's state dict.
            device (torch.device): Device to use
            dataloader (torch.utils.data.DataLoader): DataLoader to use
            path_save_folder_grad_cam (str): Path for saving grad cam images
            amount_classes (int): Amount of classes (needed for backwarding every class separately)
        """

        self.model = model
        self.model_state_dict = path_model_state_dict
        self.device = device
        self.dataloader = dataloader
        self.cam_type = cam_type
        self.pixel_set = pixel_set
        self.pixel_set_point = pixel_set_point
        self.amount_classes = amount_classes
        self.path_save_folder_grad_cam = path_save_folder_grad_cam

    def backward_cam(self, prediction: torch.tensor, backward_class: int):
        """Do the backward pass for a specific class for a specific model prediction.

        Args:
            prediction (torch.tensor): The model's prediction.
            backward_class (int): Number of the class to do the backward pass.
        """
        if prediction.shape[1] > 1:
            # get the correct segmentation result via argmax because we have multiple classes in dim 1
            segmentation = torch.argmax(prediction, dim=1)

        # get the pixel set for a specific region/point/... (explained in SegGradCam Paper)
        if self.pixel_set == "image":
            px_set_m = torch.ones_like(segmentation).squeeze().to(self.device)
        elif self.pixel_set == "class":
            px_set_m = (
                torch.where(segmentation == backward_class, 1, 0)
                .squeeze()
                .to(self.device)
            )
            # alternative way for indexing via [] in the prediction tensor:
            # px_set_m = segmentation == backward_class
            # px_set_m = px_set_m.squeeze().nonzero().to(dtype=torch.long, device=self.device)
        elif self.pixel_set == "zero":
            px_set_m = torch.zeros_like(segmentation).squeeze().to(self.device)
        elif self.pixel_set == "point":
            px_set_m = torch.zeros_like(segmentation).squeeze().to(self.device)
            px_set_m[self.pixel_set_point[0], self.pixel_set_point[1]] = 1

        # do the backward pass for a specific class
        # Note: multiplaction with the pixel set M is the extension from Grad CAM to Seg-Grad CAM
        # print((prediction[:, backward_class, :, :].squeeze() * px_set_m).shape) # -> shape of the outcome segmentation map, e.g. 560, 992
        self.model.zero_grad()
        torch.sum(prediction[:, backward_class, :, :].squeeze() * px_set_m).backward(
            retain_graph=True
        )

    def get_activations(self, image: torch.tensor):
        """Pass the model until the activation gradients are reached. The model has to be specialized for this task.
            Calculations are different depending on the cam type (seg-grad-cam or hi-res-seg-grad-cam)

        Args:
            image (torch.tensor): Image to get the activations from
        """
        if self.cam_type == "gradcam":
            # Get the gradients and pool them via mean
            gradients = self.model.get_act_grad()

            pooled_gradients = torch.mean(
                gradients, dim=[0, 2, 3]
            ).detach()  # this equals alpha_c^k

            # Get the activations
            self.activations = self.model.get_act(image).detach()

            # Multiply activations with pooled gradients
            for act in range(self.activations.shape[1]):
                self.activations[:, act, :, :] *= pooled_gradients[act]

        if self.cam_type == "hirescam":
            # Get the gradients and do not pool them (in contrast to gradcam)
            gradients = self.model.get_act_grad()
            not_pooled_gradients = gradients.detach()  # HiResCAM

            # Get the activations
            self.activations = self.model.get_act(image).detach()

            # Multiply activations with pooled gradients
            print(self.activations.shape, not_pooled_gradients.shape)
            self.activations[:, :, :, :] *= not_pooled_gradients

    def get_heatmap(self, image: torch.tensor):
        """Create heatmaps for the CAM visualization.

        Args:
            image (torch.tensor): Image to put the heatmap on.
        """
        if image.ndim == 4:
            # Create a heatmap out of the new activations
            heatmap = torch.sum(
                self.activations, dim=1
            ).squeeze()  # sum_k in L^c = ReLU(...)
            heatmap = np.maximum(heatmap.cpu(), 0)  # ReLU in L^c = ReLU(...)
            heatmap /= torch.max(
                heatmap
            ).squeeze()  # 35, 62, normalization for the heatmap visualization

            # Get the original image and "convert" it to RGB
            img = image.cpu().detach().squeeze()

            if img.shape[0] == 3:
                # rgb
                img = img.permute(1, 2, 0).numpy() / 3  # (512, 1024, 3)
            else:
                # grey
                img = img.unsqueeze(2).repeat(1, 1, 3).numpy() / 3  # (512, 1024, 3)

            # Resize the heatmap to the size of the image
            heatmap = cv2.resize(np.asarray(heatmap), (img.shape[1], img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            self.final_heatmap = heatmap * 0.4 + (img * 255)
        else:
            print("Err: Wrong image dimensions")

    def save_heatmap(self, filenames: list, backward_class: int):
        """Save function for the heatmap images

        Args:
            filenames (list): Filename for saving the image, should be only one in a list.
            backward_class (int): Class on which the backward pass was done. This is used as additional saving parameter name.
        """
        new_filename = filenames[0].replace(".png", f"_cam_{backward_class}.png")
        # create save path
        save_path = f"{self.path_save_folder_grad_cam}{new_filename}"
        cv2.imwrite(save_path, self.final_heatmap)

    def inference_cam(
        self,
        prediction: torch.tensor,
        image: torch.tensor,
        filenames: list,
        backward_class: int,
    ):
        """Do the whole inference for the CAM-method.

        Args:
            prediction (torch.tensor): Prediction of the model.
            image (torch.tensor): Image which was sent into the model for the prediction.
            filenames (list): List of filenames, should contain only 1 item as batch size has to be 1.
            backward_class (int): Class on which the backward pass is performed.
        """

        self.backward_cam(prediction=prediction, backward_class=backward_class)

        self.get_activations(image=image)

        self.get_heatmap(image=image)

        self.save_heatmap(filenames=filenames, backward_class=backward_class)


#################################################################################################################################
################################################ Architecture Training & Testing ################################################
#################################################################################################################################
class Model(object):
    def __init__(
        self,
        train_loader: torch.utils.data.DataLoader,
        validation_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        path_model: str,
        weight_decay: float = 0.005,
        modelname: str = "vanilla_unet_2d",
        learning_rate: float = 3e-3,
        device: str = "cuda",
        nr_epochs: int = 10,
        dataset_name: str = "opg_binary",
        dataset_color: str = "grey",
        nr_classes: str = "binary",
        classes: int = 33,
        batch_size: int = 1,
    ):
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.dataset_name = dataset_name
        self.dataset_color = dataset_color
        self.nr_classes = nr_classes
        self.classes = classes
        self.batch_size = batch_size

        self.path_model = path_model

        self.modelname = modelname
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.dim = self.modelname.split("_")[-1]  # 2d or 3d

        if self.dataset_color == "grey":
            input_color_channels = 1
        elif self.dataset_color == "rgb":
            input_color_channels = 3

        # Set encoder channels
        if self.dim == "2d":
            self.encoder_channels = [input_color_channels, 64, 128, 256, 512, 1024]
        if self.dim == "3d":
            self.encoder_channels = [input_color_channels, 64, 128, 256, 512, 1024]

        # Set decoder channels
        if self.dim == "2d":
            self.decoder_channels = [1024, 512, 256, 128, 64, self.classes]
        if self.dim == "3d":
            self.decoder_channels = [1024, 512, 256, 128, 64, self.classes]

        self.device = torch.device(device)
        self.nr_epochs = nr_epochs

        if self.nr_classes == "binary":
            # loss function depending on model
            if self.modelname == "vanilla_unet_2d":
                self.criterion = mDiceLoss(reduction="mean")  # torch.nn.BCELoss()
                self.loss_function = "dice"  # "ce"

        elif self.nr_classes == "multiclass":
            if self.dim == "2d":
                self.criterion = torch.nn.CrossEntropyLoss()
                self.loss_function = "ce"
                if self.modelname == "attention_unet_2d":
                    self.criterion = mDiceLoss(reduction="mean", softmax=True)
                    self.loss_function = "dice"

        self.model = None
        self.build_model()

    def build_model(self):
        print(f"Currently selected model ({self.modelname}) is built.")
        # Select the model and build it

        # *------------------------------------ 2D Models --------------------------------------*
        if self.modelname == "vanilla_unet_2d":
            self.model = Vanilla_UNet_2d(
                encoder_channels=self.encoder_channels,
                decoder_channels=self.decoder_channels,
            )

        # *-------------------------------- Devices, Optimizer ---------------------------------*
        # Select the optimizer
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()), self.learning_rate
        )

        # Send the model to the device
        # if self.dim == "3d":
        #    self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

    def save_metrics_to_csv(self, f1: torch.tensor, iou: torch.tensor):
        path_save_csv = f"{self.path_model}{self.dataset_name}/metrics.csv"

        with open(path_save_csv, mode="w") as metric_csv:
            fieldnames = ["f1", "iou"]
            writer = csv.DictWriter(metric_csv, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerow({"f1": str(f1.item()), "iou": str(iou.item())})

    def save_image_metrics_to_csv(
        self, filename_list: list, f1_list: list, iou_list: list
    ):
        path_save_csv = f"{self.path_model}{self.dataset_name}/metrics_per_img.csv"
        nr_files = len(filename_list)

        with open(path_save_csv, mode="w") as metric_csv:
            fieldnames = ["filename", "f1", "iou"]
            writer = csv.DictWriter(metric_csv, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(nr_files):
                writer.writerow(
                    {
                        "filename": filename_list[i][0],
                        "f1": str(f1_list[i].item()),
                        "iou": str(iou_list[i].item()),
                    }
                )

    def train_val_test(self):
        # Init a score to determine whether to save a new model from a new epoch or not
        self.model_summary_score = 0

        for epoch in range(self.nr_epochs):
            # __________________________ Training ___________________________
            self.model.train()
            print(f"Epoch: {epoch+1}/{self.nr_epochs}")

            epoch_loss = 0
            # Set further metrics to 0 here
            f1 = 0.0
            iou = 0.0

            # make a list of filenames to create visualizations after an epoch
            list_filenames = []

            # Train Loop
            for i, (image, ground_truth, filenames) in enumerate(self.train_loader):
                self.model.train()
                self.optimizer.zero_grad()

                # Move image and GT to device
                image = image.to(self.device)
                ground_truth = ground_truth.to(self.device)

                # get model prediction
                segmentation = self.model(image)

                if self.nr_classes == "multiclass":
                    ground_truth = ground_truth.squeeze(dim=1)
                    ground_truth = ground_truth.long()

                if (
                    self.loss_function == "dice"
                    and self.dim == "2d"
                    and self.nr_classes == "multiclass"
                ):
                    ground_truth = F.one_hot(
                        ground_truth, num_classes=self.classes
                    ).permute(0, 3, 1, 2)

                # get loss
                loss = self.criterion(segmentation, ground_truth)
                # calculate the whole epoch loss (for the print statement later)
                epoch_loss = epoch_loss + loss.item()

                if (
                    self.loss_function == "ce"
                    and self.dim == "2d"
                    and self.nr_classes == "multiclass"
                ):
                    ground_truth = F.one_hot(
                        ground_truth, num_classes=self.classes
                    ).permute(0, 3, 1, 2)
                    ground_truth = ground_truth.float()

                # calculate and sum metrics
                f1 += f1_score(ground_truth, segmentation, nr_classes=self.classes)
                iou += iou_score(ground_truth, segmentation, nr_classes=self.classes)

                # backward pass
                loss.backward()

                # Step for the optimizer
                self.optimizer.step()

                list_filenames.append(filenames)

            f1 = f1 / len(self.train_loader)
            iou = iou / len(self.train_loader)
            epoch_loss = epoch_loss / len(self.train_loader)

            # Print training information such as metrics, etc.
            print(
                "[Train] \t Loss: {:.4f} \t \t F1-Score: {:.4f} \t IoU: {:.4f}".format(
                    epoch_loss, f1, iou
                )
            )

            # __________________________ Validation ___________________________

            # reset metrics here
            f1 = 0.0
            iou = 0.0

            epoch_loss_val = 0

            self.model.eval()
            for i, (image, ground_truth) in enumerate(self.validation_loader):
                # Move image and GT to device
                image = image.to(self.device)
                ground_truth = ground_truth.to(self.device)

                with torch.no_grad():
                    segmentation = self.model(image)

                # calculate loss
                with torch.no_grad():
                    if self.nr_classes == "multiclass":  # and (self.dim == "2d")
                        ground_truth = ground_truth.squeeze(dim=1)
                        ground_truth = ground_truth.long()

                    if (
                        self.loss_function == "dice"
                        and self.dim == "2d"
                        and self.nr_classes == "multiclass"
                    ):
                        ground_truth = F.one_hot(
                            ground_truth, num_classes=self.classes
                        ).permute(0, 3, 1, 2)

                    loss = self.criterion(segmentation, ground_truth)

                    epoch_loss_val += loss.item()

                    if (
                        self.loss_function == "ce"
                        and self.dim == "2d"
                        and self.nr_classes == "multiclass"
                    ):
                        ground_truth = F.one_hot(
                            ground_truth, num_classes=self.classes
                        ).permute(0, 3, 1, 2)
                        ground_truth = ground_truth.float()

                # calculate and sum metrics
                f1 += f1_score(ground_truth, segmentation, nr_classes=self.classes)
                iou += iou_score(ground_truth, segmentation, nr_classes=self.classes)

            f1 = f1 / len(self.validation_loader)
            iou = iou / len(self.validation_loader)
            epoch_loss_val = epoch_loss_val / len(self.validation_loader)

            # Print validation information such as metrics, etc.
            print(
                "[Validation] \t Loss: {:.4f} \t \t F1-Score: {:.4f} \t IoU: {:.4f}".format(
                    epoch_loss_val, f1, iou
                )
            )
            new_model_summary_score = f1 + iou
            # If the new score is better than the last score, save the model.
            if new_model_summary_score > self.model_summary_score:
                self.model_summary_score = new_model_summary_score
                if not os.path.exists(self.path_model):
                    os.makedirs(self.path_model)
                if not os.path.exists(f"{self.path_model}{self.dataset_name}/"):
                    os.makedirs(f"{self.path_model}{self.dataset_name}/")
                torch.save(
                    self.model.state_dict(),
                    f"{self.path_model}{self.dataset_name}/model",
                )
            else:
                print(f"")
                continue  # do the test-path only if there's a better model. This saves time.

            # __________________________ Test ___________________________
            # By doing the following, the best model is always selected for the testing afterwards.
            self.model.load_state_dict(
                torch.load(f"{self.path_model}{self.dataset_name}/model")
            )

            path_save_images = f"{self.path_model}".replace(
                "models", f"images/{self.dataset_name}"
            )

            if not os.path.exists(path_save_images):
                os.makedirs(path_save_images)

            # reset metrics here
            f1 = 0.0
            iou = 0.0

            filename_list = []

            f1_list = []
            iou_list = []

            self.model.eval()
            for i, (image, ground_truth, filenames) in enumerate(self.test_loader):
                # Move image and GT to device
                image = image.to(self.device)
                ground_truth = ground_truth.to(self.device)

                with torch.no_grad():
                    segmentation = self.model(image)
                    if self.nr_classes == "multiclass":  # and (self.dim == "2d")
                        ground_truth = ground_truth.squeeze(dim=1)
                        ground_truth = ground_truth.long()

                    if (
                        self.loss_function == "dice"
                        and self.dim == "2d"
                        and self.nr_classes == "multiclass"
                    ):
                        ground_truth = F.one_hot(
                            ground_truth, num_classes=self.classes
                        ).permute(0, 3, 1, 2)

                    loss = self.criterion(segmentation, ground_truth)

                    if (
                        self.loss_function == "ce"
                        and self.dim == "2d"
                        and self.nr_classes == "multiclass"
                    ):
                        ground_truth = F.one_hot(
                            ground_truth, num_classes=self.classes
                        ).permute(0, 3, 1, 2)
                        ground_truth = ground_truth.float()

                # calculate and sum metrics
                current_f1 = f1_score(
                    ground_truth, segmentation, nr_classes=self.classes
                )
                current_iou = iou_score(
                    ground_truth, segmentation, nr_classes=self.classes
                )

                if self.dim == "2d":
                    save_test_predictions(
                        segmentation.detach(),
                        path_save_images,
                        filenames,
                        nr_classes=self.nr_classes,
                    )

                f1 += current_f1
                iou += current_iou
                f1_list.append(current_f1)
                iou_list.append(current_iou)
                filename_list.append(filenames)

            f1 = f1 / len(self.test_loader)
            iou = iou / len(self.test_loader)

            self.save_metrics_to_csv(f1, iou)
            self.save_image_metrics_to_csv(filename_list, f1_list, iou_list)

            # Print validation information such as metrics, etc.
            print(
                "[Test] \t \t \t \t \t F1-Score: {:.4f} \t IoU: {:.4f}".format(f1, iou)
            )
            print(f"")


def save_test_predictions(
    predictions: torch.tensor,
    save_root: str,
    filenames: list,
    nr_classes: str = "binary",
):
    # predictions e.g. shape [1, 1, 128, 128] - B, C, H, W
    batchsize = predictions.shape[0]

    if nr_classes == "multiclass":
        for i in range(batchsize):
            # get current prediction mask from the whole batch
            prediction_mask = predictions[i, :, :, :]

            # argmax the prediction mask to be able to visualize multi class segmentation results
            prediction_mask_softmax = nn.Softmax(dim=0)(prediction_mask)
            prediction_mask_argmax = torch.argmax(prediction_mask_softmax, dim=0)
            # range the final segmentation result to the original segmentation classes
            segmentation_result = prediction_mask_argmax / 255
            # send prediction to cpu and convert to numpy
            segmentation_result = segmentation_result.to("cpu").numpy() * 255
            # convert to uint8 as it is binary
            segmentation_result = segmentation_result.astype(np.uint8)
            # convert to grayscale PIL image
            if np.max(segmentation_result) == 1:
                segmentation_result = Image.fromarray(
                    segmentation_result * 255
                ).convert("L")
            else:
                segmentation_result = Image.fromarray(segmentation_result).convert("L")
            # save the image
            if ".pt" in filenames[i]:
                filenames[i] = filenames[i].replace(".pt", ".png")
            segmentation_result.save(f"{save_root}{filenames[i]}")

    elif nr_classes == "binary":
        for i in range(batchsize):
            # get current prediction mask from the whole batch
            prediction_mask = predictions[i, :, :, :]
            # send prediction to cpu and convert to numpy
            prediction_mask = prediction_mask.to("cpu").numpy()
            # treshold the binary mask
            prediction_mask = np.where(prediction_mask > 0.5, 1, 0) * 255
            # convert to uint8 as it is binary
            prediction_mask = prediction_mask.astype(np.uint8)
            # remove unneccesary dimension
            prediction_mask = np.squeeze(prediction_mask)
            # print(np.unique(prediction_mask))
            # convert to grayscale PIL image
            prediction_mask = Image.fromarray(prediction_mask).convert("L")
            # save the image
            if ".pt" in filenames[i]:
                filenames[i] = filenames[i].replace(".pt", "")
            prediction_mask.save(f"{save_root}{filenames[i]}")
