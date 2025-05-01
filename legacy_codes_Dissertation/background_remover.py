import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from model import U2NET 
from model import U2NETP 
from IPython.display import display
from PIL import Image as Img
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import os
import random
import matplotlib.pyplot as plt
from PIL import Image as Img

from sklearn.model_selection import train_test_split



!git clone https://github.com/shreyas-bk/U-2-Net

import sys
sys.path.append('./U-2-Net')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


THRESHOLD = 0.1

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn

def remove_background2(image):
    # Convert PIL image to numpy array
    image_np = np.array(image)
    # Create a temporary image file for processing
    print("Removing")
    temp_img_path = 'temp_img.png'
    Img.fromarray(image_np).save(temp_img_path)

    # Prepare the dataset and dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=[temp_img_path], lbl_name_list=[], 
                                        transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

    for i_test, data_test in enumerate(test_salobj_dataloader):
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)
        predict = d5[:, 0, :, :]
        predict = normPRED(predict)

        del d1, d2, d3, d4, d5, d6, d7

        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()

        predict_np[predict_np > THRESHOLD] = 1
        predict_np[predict_np <= THRESHOLD] = 0
        mask = Img.fromarray(predict_np * 255).convert('RGB')
        imask = mask.resize((image_np.shape[1], image_np.shape[0]), resample=Img.BILINEAR)
        back = Img.new("RGB", (image_np.shape[1], image_np.shape[0]), (255, 255, 255))
        mask = imask.convert('L')
        im_out = Img.composite(Img.fromarray(image_np), back, mask)

    return im_out


# Load the U2NET model
model_dir = "./U-2-Net/u2netp.pth"
net = U2NETP(3, 1)

if torch.cuda.is_available():
    net.load_state_dict(torch.load(model_dir))
    net.cuda()
else:
    net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))

net.eval()


# Custom dataset to load images and labels from directory structure
class FullDataset(Dataset):
    def __init__(self, root_dir, transform=None, device=None, backremove=True, save_dir=None):
        self.root_dir = root_dir
        self.transform = transform
        self.device = device
        self.backremove = backremove
        self.save_dir = save_dir

        self.image_paths = []
        self.labels = []

        # Scan directory structure to get image paths and labels
        for class_idx, class_dir in enumerate(sorted(os.listdir(root_dir))):
            class_dir_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_dir_path):
                for img_name in os.listdir(class_dir_path):
                    self.image_paths.append(os.path.join(class_dir_path, img_name))
                    self.labels.append(class_idx)  # Label inferred from directory structure

        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error: Image {img_path} could not be loaded.")
            return None, None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.backremove:
            image = remove_background2(image)
            image = np.array(image)

        if self.save_dir:
            # Ensure the directory structure is replicated
            class_dir = os.path.dirname(img_path).replace(self.root_dir, '').lstrip(os.sep)
            save_class_dir = os.path.join(self.save_dir, class_dir)
            if not os.path.exists(save_class_dir):
                os.makedirs(save_class_dir)
                
            new_img_name = os.path.join(save_class_dir, os.path.basename(img_path))
            cv2.imwrite(new_img_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
#             print(f"Saved processed image to {new_img_name}")

        image = Img.fromarray(image)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx]).float()

        return image, label


# Data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Path to image directory
image_dir = "/plain/"
save_dir = "background/"

# Load the dataset with background removal and save the processed images
print("Creating dataset...")
full_dataset = FullDataset(root_dir=image_dir, transform=data_transforms['train'], device=device, backremove=True, save_dir=save_dir)


print(f"Number of samples in the dataset: {len(full_dataset)}")



# Process all images to ensure they are saved with background removed
print("Processing all images...")
for idx in range(len(full_dataset)):
    img, lbl = full_dataset[idx]
    if img is None:
        print(f"Skipping image at index {idx} due to loading error.")
    else:
        if idx%10 ==0 :
            print(f"Processed image at index {idx}.")

# Split the dataset into training and validation sets
train_indices, val_indices = train_test_split(list(range(len(full_dataset))), test_size=0.2, random_state=42)
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

# Apply the respective transforms to each subset
train_dataset.dataset.transform = data_transforms['train']
val_dataset.dataset.transform = data_transforms['val']

print("Creating data loaders...")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)