import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image as Img

class CustomDataset(Dataset):
    """
    Dataset that loads images from directory and applies:
    - Optional background removal (via a callable)
    - Optional image transform
    - Optional saving to disk after background removal
    """
    def __init__(self, root_dir, transform=None, background_remover=None,
                 save_dir=None, class_map=None):
        self.root_dir = root_dir
        self.transform = transform
        self.background_remover = background_remover
        self.save_dir = save_dir
        self.image_paths = []
        self.labels = []

        class_folders = sorted(os.listdir(root_dir))
        self.class_map = class_map or {cls: i for i, cls in enumerate(class_folders)}

        for class_name in class_folders:
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for img_name in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, img_name))
                self.labels.append(self.class_map[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.background_remover is not None:
            image = self.background_remover(image)
            image = np.array(image)

        # Save processed image (optional)
        if self.save_dir:
            rel_path = os.path.relpath(img_path, self.root_dir)
            save_path = os.path.join(self.save_dir, rel_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        image = Img.fromarray(image)
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label).long()
