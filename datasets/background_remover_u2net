# datasets/background_remover_u2net.py

import os
import random
import numpy as np
import torch
import cv2
from PIL import Image as Img
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

# Import U-2-Net
from data_loader import RescaleT, ToTensor, ToTensorLab, SalObjDataset
from model import U2NETP

# Define global threshold
THRESHOLD = 0.1

def normPRED(d):
    """Normalize prediction maps to [0,1]"""
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn

def remove_background_u2net(image, net):
    """
    Remove background from a given image using U-2-Net model.

    Args:
        image: PIL Image or numpy array.
        net: Pre-loaded U-2-Net model.

    Returns:
        PIL Image with background removed.
    """
    if isinstance(image, Img.Image):
        image_np = np.array(image)
    else:
        image_np = image

    temp_img_path = 'temp_img.png'
    Img.fromarray(image_np).save(temp_img_path)

    test_dataset = SalObjDataset(img_name_list=[temp_img_path], lbl_name_list=[], 
                                 transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    for data in test_loader:
        inputs = data['image'].type(torch.FloatTensor)
        inputs = Variable(inputs.cuda() if torch.cuda.is_available() else inputs)
        
        d1, _, _, _, d5, _, _ = net(inputs)
        predict = normPRED(d5[:, 0, :, :])
        
        predict = predict.squeeze().cpu().data.numpy()
        predict[predict > THRESHOLD] = 1
        predict[predict <= THRESHOLD] = 0

        mask = Img.fromarray((predict * 255).astype(np.uint8)).convert('RGB')
        mask = mask.resize((image_np.shape[1], image_np.shape[0]), resample=Img.BILINEAR)

        back = Img.new("RGB", (image_np.shape[1], image_np.shape[0]), (255, 255, 255))
        mask = mask.convert('L')
        im_out = Img.composite(Img.fromarray(image_np), back, mask)

        return im_out

class FullDataset(Dataset):
    """
    Full dataset class with optional background removal using U-2-Net.
    """

    def __init__(self, root_dir, transform=None, device=None, backremove=True, save_dir=None, net=None):
        self.root_dir = root_dir
        self.transform = transform
        self.device = device
        self.backremove = backremove
        self.save_dir = save_dir
        self.net = net

        self.image_paths = []
        self.labels = []

        for class_idx, class_dir in enumerate(sorted(os.listdir(root_dir))):
            class_dir_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_dir_path):
                for img_name in os.listdir(class_dir_path):
                    self.image_paths.append(os.path.join(class_dir_path, img_name))
                    self.labels.append(class_idx)

        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Error loading image {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.backremove and self.net is not None:
            image = remove_background_u2net(image, self.net)
            image = np.array(image)

        if self.save_dir:
            class_dir = os.path.dirname(img_path).replace(self.root_dir, '').lstrip(os.sep)
            save_class_dir = os.path.join(self.save_dir, class_dir)
            os.makedirs(save_class_dir, exist_ok=True)
            new_img_path = os.path.join(save_class_dir, os.path.basename(img_path))
            cv2.imwrite(new_img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        image = Img.fromarray(image)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx]).float()

        return image, label

def load_u2net_model(model_path="./U-2-Net/u2netp.pth"):
    """
    Loads U-2-NetP model.

    Args:
        model_path: Path to the u2netp.pth file.

    Returns:
        Loaded U-2-Net model.
    """
    net = U2NETP(3, 1)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()
    return net
