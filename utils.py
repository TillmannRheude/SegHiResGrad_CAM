import cv2 
import os 
import torch 

import numpy as np 
import torch.nn as nn 

# local 2d imports
from networks import Vanilla_UNet_2d




def set_encoder_decoder(color, amount_classes, depth=5):
    start_val = 1 if color=="grey" else 3

    available_depths_encoder = [start_val, 64, 128, 256, 512, 1024]
    available_depths_decoder = [1024, 512, 256, 128, 64, amount_classes]    
    
    # Set encoder and decoder values for the network
    encoder_channels = available_depths_encoder[:depth+1]
    decoder_channels = available_depths_decoder[5-depth:]

    return encoder_channels, decoder_channels 

def create_save_folders(path_save_folder_grad_cam):
    if not os.path.exists(path_save_folder_grad_cam):
        os.makedirs(path_save_folder_grad_cam)

def create_model(modelname, encoder_channels, decoder_channels, path_save_folder_grad_cam, path_load_model): 
    if modelname == "vanilla_unet":
        model = Vanilla_UNet_2d(encoder_channels=encoder_channels, decoder_channels=decoder_channels)
    else:
        print("Err: Model not defined")
    
    return model, path_save_folder_grad_cam, path_load_model


class Model_Inference():
    def __init__(self, 
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
                show_progress=True): 
        """ Inference class for doing CAM and/or metric calculations.

        Args:
            model (nn.Module): Model for doing inference
            model_name (str): modelname
            shift_type (str): shift type (for Shift UNet)
            dataset_name (str): dataset name
            path_model_state_dict (str): Path for the model's state dict
            device (torch.device): Device for doing predictions on 
            dataloader (torch.utils.data.DataLoader): DataLoader
            amount_classes (int): Number of classes to predict (x classes + 1 background)
            cam_type (_type_, optional): Cam Type to choose (gradcam or proposed hirescam (for segmentation and not classification)). 
                                        Defaults to None.
            metrics (bool, optional): Choose if metrics should be calculated or not. 
                                    Defaults to True.
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
            self.cam = CAM(cam_type=cam_type, 
                            pixel_set=self.pixel_set,
                            pixel_set_point=self.pixel_set_point,
                            model=self.model, 
                            path_model_state_dict=self.model_state_dict,
                            device=self.device,
                            dataloader=self.dataloader,
                            path_save_folder_grad_cam=self.path_save_folder_grad_cam,
                            amount_classes=self.amount_classes)

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
                print(f"Currently at {filenames[0]} (number {i+1} out of {len(self.dataloader)+1})")

            # do inference prediction with the model
            prediction = self.predict(image)

            # CAM  
            if cam: 
                # iterate through all classes
                for cl in range(self.amount_classes):
                    self.cam.inference_cam(prediction=prediction, image=image.to(self.device), filenames=filenames, backward_class=cl)


class CAM():
    """ Script for re-implementation of Seg-Grad-Cam originally proposed by Vinogradova et al..
        Further, re-implementation of HiResCAM originally proposed by Draelos et al.

        Finally, proposed combination of HiResCAM with Seg-Grad-Cam which results in HiRes-Seg-Grad-CAM 
        for more explainable results compared to Seg-Grad-Cam.

    Citations: 
        Kira Vinogradova, Alexandr Dibrov, & Eugene W. Myers (2020). Towards Interpretable Semantic Segmentation via 
        Gradient-weighted Class Activation Mapping. In Proceedings of the AAAI Conference on Artificial Intelligence.

        Draelos, R., & Carin, L.. (2020). Use HiResCAM instead of Grad-CAM for faithful explanations of convolutional 
        neural networks.

    """
    def __init__(self, 
                cam_type: str,
                pixel_set: str,
                pixel_set_point: None,
                model: nn.Module, 
                path_model_state_dict: str, 
                device: torch.device,
                dataloader: torch.utils.data.DataLoader,
                path_save_folder_grad_cam: str,
                amount_classes: int): 
        """ Initialize CAM.

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
        """ Do the backward pass for a specific class for a specific model prediction.

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
            px_set_m = torch.where(segmentation == backward_class, 1, 0).squeeze().to(self.device) 
            # alternative way for indexing via [] in the prediction tensor:
            #px_set_m = segmentation == backward_class 
            #px_set_m = px_set_m.squeeze().nonzero().to(dtype=torch.long, device=self.device)
        elif self.pixel_set == "zero":
            px_set_m = torch.zeros_like(segmentation).squeeze().to(self.device)
        elif self.pixel_set == "point":
            px_set_m = torch.zeros_like(segmentation).squeeze().to(self.device)
            px_set_m[self.pixel_set_point[0], self.pixel_set_point[1]] = 1

        # do the backward pass for a specific class
        # Note: multiplaction with the pixel set M is the extension from Grad CAM to Seg-Grad CAM
        # print((prediction[:, backward_class, :, :].squeeze() * px_set_m).shape) # -> shape of the outcome segmentation map, e.g. 560, 992
        self.model.zero_grad()
        torch.sum( prediction[:, backward_class, :, :].squeeze() * px_set_m ).backward(retain_graph=True)
    
    def get_activations(self, image: torch.tensor):
        """ Pass the model until the activation gradients are reached. The model has to be specialized for this task. 
            Calculations are different depending on the cam type (seg-grad-cam or hi-res-seg-grad-cam)

        Args:
            image (torch.tensor): Image to get the activations from
        """
        if self.cam_type == "gradcam":
            # Get the gradients and pool them via mean
            gradients = self.model.get_act_grad()
            
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3]).detach() # this equals alpha_c^k 

            # Get the activations
            self.activations = self.model.get_act(image).detach()

            # Multiply activations with pooled gradients
            for act in range(self.activations.shape[1]):
                self.activations[:, act, :, :] *= pooled_gradients[act]

        
        if self.cam_type == "hirescam":
            # Get the gradients and do not pool them (in contrast to gradcam)
            gradients = self.model.get_act_grad()
            not_pooled_gradients = gradients.detach() # HiResCAM

            # Get the activations
            self.activations = self.model.get_act(image).detach()

            # Multiply activations with pooled gradients
            self.activations[:, :, :, :] *= not_pooled_gradients
            
    
    def get_heatmap(self, image: torch.tensor):
        """ Create heatmaps for the CAM visualization.

        Args:
            image (torch.tensor): Image to put the heatmap on.
        """
        if image.ndim == 4: 
            # Create a heatmap out of the new activations
            heatmap = torch.sum(self.activations, dim=1).squeeze() # sum_k in L^c = ReLU(...)
            heatmap = np.maximum(heatmap.cpu(), 0) # ReLU in L^c = ReLU(...)
            heatmap /= torch.max(heatmap).squeeze() # 35, 62, normalization for the heatmap visualization 

            # Get the original image and "convert" it to RGB
            img = image.cpu().detach().squeeze()
            
            if img.shape[0] == 3: 
                # rgb 
                img = img.permute(1, 2, 0).numpy() / 3 # (512, 1024, 3)
            else: 
                # grey
                img = img.unsqueeze(2).repeat(1, 1, 3).numpy() / 3 # (512, 1024, 3)

            # Resize the heatmap to the size of the image 
            heatmap = cv2.resize(np.asarray(heatmap), (img.shape[1], img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            self.final_heatmap = heatmap * 0.4 + (img * 255)
        else:
            print("Err: Wrong image dimensions")
        
    def save_heatmap(self, filenames: list, backward_class: int):
        """ Save function for the heatmap images

        Args:
            filenames (list): Filename for saving the image, should be only one in a list. 
            backward_class (int): Class on which the backward pass was done. This is used as additional saving parameter name. 
        """
        new_filename = filenames[0].replace(".png", f"_cam_{backward_class}.png")
        # create save path
        save_path = f"{self.path_save_folder_grad_cam}{new_filename}"
        cv2.imwrite(save_path, self.final_heatmap)

    def inference_cam(self, prediction: torch.tensor, image: torch.tensor, filenames: list, backward_class: int):
        """ Do the whole inference for the CAM-method. 

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

