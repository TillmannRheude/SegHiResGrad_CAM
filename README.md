<br />
<div align="center">
  

  <h3 align="center">Seg-HiRes-Grad CAM</h3>

  <p align="justify">
    CNNs achieve prevailing results in segmentation tasks nowadays and represent the state-of-the-art for image-based analysis. However, the understanding of their 
    accurate decision-making process is rather unknown. Besides only one method for segmentation-based interpretation of CNNs with the visualization of CAMs, no
    further and more modern methods exist in contrast to classification-based interpretation methods. Our method improves the previously-mentioned existing 
    segmentation-based method by adjusting it to recently published classification-based methods. We propose a transfer between existing classification- and 
    segmentation-based methods for more detailed, explainable, and consistent results. The resulting Seg-HiRes-Grad CAM is an extension of the segmentation-based 
    Seg-Grad CAM with the transfer of the classification-based HiRes CAM. This produces explainable heatmaps which show salient pixels in semantic segmentation tasks. 
    Especially for medical image segmentation, this transfer solves explainability disadvantages of Seg-Grad CAM.
  </p>
  
  <p align="center">
    <br />
    <a href="https://github.com/">Paper</a>
  </p>
  
  <a href="https://github.com">
    <img src="readme_images/tuda_igd.png" alt="Logo" width="376" height="77">
  </a>
  <br />
  <a href="https://github.com">
    <img src="readme_images/demo_pic.PNG" alt="Demo" width="872" height="364">
  </a>
</div>


# Requirements
Python 3.9.12, PyTorch 1.11.0, NumPy 1.21.5, Image PIL, torchvision 0.12.0


# Data and Model Preparation
Please structure the data and model as the folders in this repository, e.g. as follows:
```bash
├──data 
    ├──dataset_1 (e.g. cityscapes)
          ├──test
              ├──images
                  ├──files.png
              ├──masks
                  ├──files.png
    ├──dataset_2
          ├──...
├──results 
    ├──modelname (e.g. vanilla_unert)
          ├──models
              ├──dataset_1 (e.g. cityscapes)
                  ├──modelfile
    ├──modelname
          ├──...
``` 

The convolutional neural network model has to be adjusted (if it is not the U-Net which is in this repository) such that it can be used out of the box with the ```main.py``` in this repository, e.g. as the following minimal example:
```bash
class Vanilla_UNet_2d(nn.Module):
    def __init__(self, grad_cam=False):
        super(Vanilla_UNet_2d, self).__init__()
        
        self.grad_cam = grad_cam # activate this for .eval() the model 
        
    def activations_hook(self, grad): 
        # for GradCam
        self.gradients = grad
    
    def forward(self, x):
        # do something
        
        # encoder
        
        # i want the explanation of this layer
        # for GradCam
        if self.grad_cam: self.h = x.register_hook(self.activations_hook)
        
        # decoder
    
        return x
    
    def get_act_grad(self):
        # for GradCam
        return self.gradients
    
    def get_act(self, x):
        # for GradCam
        for block in self.encoder:
            # send the tensor to the encoder block and get the encoded tensor before and after the max pooling operation
            x_pre_pool, x_post_pool = block(x)
            # save the encoded tensor before the max pooling operation for the skip connection part later
            x = x_post_pool
        x = x_pre_pool

        return x 
``` 

To run Seg(-HiRes)-Grad CAM, the ```main.py``` file provides the relevant arguments. 

# Citation
Please cite the work with the following information (bibtex format):
...
