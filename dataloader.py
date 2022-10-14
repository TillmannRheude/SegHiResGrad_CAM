import torch
import glob
import cv2

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils import data

from torchvision import transforms as T

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



class Dataset_2d(Dataset):
    """ Data loader for the U-Net architectures. 

        Folder should be structured like the following:

        /architectures
        ├── ...
        ├── unet
        │   ├── dataloader.py
        │   ├── ...
        ├── data
        │   ├── data_1
        │   │   ├── test
        │   │   │   ├── images
        │   │   │   │   ├── image_1.jpg
        │   │   │   │   ├── ...
        │   │   │   ├── masks
        │   │   │   │   ├── mask_1.png
        │   │   │   │   ├── ...
        │   │   ├── train
        │   │   │   ├── images
        │   │   │   ├── masks
        │   │   ├── validation
        │   │   │   ├── images
        │   │   │   ├── masks
        │   ├── data_2
        │   ├── ...
    
    Args:
        -
    """
    
    def __init__(self, dir_root, dataset_color="grey", transform={}, return_filenames=False, classes="binary"):
        # sanity checks
        if not dir_root.endswith("/"): 
            dir_root += "/"
        
        # direct assignments
        self.dir_root = dir_root # "../data/data_1/train/"
        self.transform = transform
        self.return_filenames = return_filenames
        self.dataset_color = dataset_color
        self.classes = classes

        # define image file and mask file extensions
        self.img_extension = "jpg"
        if "cityscapes" in self.dir_root: self.img_extension = "png"
        self.mask_extension = "png"

        # further assignments
        self.paths_images = sorted(glob.glob(f"{dir_root}images/*.{self.img_extension}"))
        self.paths_masks = sorted(glob.glob(f"{dir_root}masks/*.{self.mask_extension}"))

        self.array_transformations = [] # transformations for image and mask

        # transformation dictionary
        if("resize" in self.transform):
            self.resize = self.transform.get("resize")
            if self.resize is not None:
                self.array_transformations.append(T.Resize(size=self.resize, interpolation=T.InterpolationMode.NEAREST)) # (250,358)

    def __len__(self):
        return len(self.paths_images)

    def get_labels(self):
        if "opg_multiclass" in self.dir_root:
            class_labels = [0.0000, 0.0039, 0.0078, 0.0118, 0.0157, 0.0196, 0.0235, 0.0275, 0.0314,
                            0.0353, 0.0392, 0.0431, 0.0471, 0.0510, 0.0549, 0.0588, 0.0627, 0.0667,
                            0.0706, 0.0745, 0.0784, 0.0824, 0.0863, 0.0902, 0.0941, 0.0980, 0.1020,
                            0.1059, 0.1098, 0.1137, 0.1176, 0.1216, 0.1255]
        elif "cityscapes" in self.dir_root:
            class_labels = [0, 0.0039, 0.0078, 0.0118,0.0157, 0.0196, 0.0235, 0.0275, 0.0314,     
                            0.0353, 0.0392, 0.0431, 0.0471, 0.0510, 0.0549, 0.0588, 0.0627,      
                            0.0667, 0.0706, 0.0745, 0.0784, 0.0824, 0.0863, 0.0902, 0.0941,     
                            0.0980, 0.1020, 0.1059, 0.1098, 0.1137, 0.1176, 0.1216, 0.1255,    
                            0.1294]
        else: 
            class_labels = [0, 0, 0]
            print("Warning: Class labels not defined.") 
        
        return class_labels

    def __getitem__(self, idx):
        filename = Path(self.paths_masks[idx]).name

        # Load grey jpg image 
        if self.dataset_color == "grey":
            image = Image.open(self.paths_images[idx]).convert("L")
        # Load rgb jpg image 
        elif self.dataset_color == "rgb":
            image = cv2.imread(self.paths_images[idx]) # the workaround with cv2 for solving a warn-message
            image = Image.fromarray(image).convert("RGB")

        # Load respective masks
        mask = Image.open(self.paths_masks[idx]).convert("L")

        # do basic transformations (such as resizing)
        Transform = T.Compose(self.array_transformations)
        image, mask = Transform(image), Transform(mask)

        # convert mask and image to tensor
        image = T.ToTensor()(image)
        mask = T.ToTensor()(mask)

        # sanity checks
        if image.ndim == 4:
            # check image dimensions
            image = torch.unsqueeze(image, dim=0)
        if mask.ndim == 4:
            # check mask dimensions
            mask = torch.unsqueeze(mask, dim=0)
        if self.classes == "binary":
            if torch.max(image) != 1:
                min_val = torch.min(image)
                max_val = torch.max(image)
                # check mask normalization range, they should already be normalized to 0-1 but multiplied with 255
                image = (image - min_val)/(max_val - min_val)
        elif self.classes == "multiclass":
            class_labels = self.get_labels()
            class_labels = torch.round(torch.Tensor(class_labels), decimals=4)

            for class_label_int, class_label in enumerate(class_labels):
                # change float labels to 0, 1, ... nClasses-1 (integer) labels
                mask[torch.round(mask, decimals=4) == class_label] = class_label_int

            if torch.max(image) != 1:
                min_val = torch.min(image)
                max_val = torch.max(image)
                # check mask normalization range, they should already be normalized to 0-1 but multiplied with 255
                image = (image - min_val)/(max_val - min_val)

        if self.return_filenames:
            return image, mask, filename
        else:
            return image, mask


def build_dataloader(dir_root, transform={}, batch_size=1, num_workers=1, return_filenames=False, shuffle=True, dataset_color="grey", classes="binary"):
    dataset = Dataset_2d(dir_root=dir_root, transform=transform, return_filenames=return_filenames, dataset_color=dataset_color, classes=classes)
   
    dataloader = data.DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers)
    return dataloader