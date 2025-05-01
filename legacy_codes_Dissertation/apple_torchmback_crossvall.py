import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, Subset
import timm
import argparse
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Argument parser for model selection
parser = argparse.ArgumentParser(description="Train different CNN architectures on the dataset")
parser.add_argument('--model', type=str, required=True, help='Model architecture to use: inception, xception, resnet50, resnet18, resnet152, vgg19')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train the model')
parser.add_argument('--lr', type=float, default=0.005, help='Learning rate for the optimizer')
args = parser.parse_args()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}

        for class_name in self.class_names:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

root_dir = "/cs/home/psxjs24/data/apple_binary/processed_images"  # Update this path to your dataset directory

# Load dataset with appropriate transforms
full_dataset = CustomDataset(root_dir=root_dir, transform=data_transforms['val'])


k_folds = 5

# Initialize the KFold object
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)


for fold, (train_indices, val_indices) in enumerate(kf.split(full_dataset)):
    print(f"\nFold {fold + 1}/{k_folds}")
    
    # Prepare training and validation datasets for the current fold
    train_dataset = Subset(CustomDataset(root_dir=root_dir, transform=data_transforms['train']), train_indices)
    val_dataset = Subset(CustomDataset(root_dir=root_dir, transform=data_transforms['val']), val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=True)


    def get_model(model_name, num_classes):
        if model_name == 'inception':
            model = models.inception_v3(pretrained=True)
            model.aux_logits = False
        elif model_name == 'xception':
            model = timm.create_model('xception', pretrained=True)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
        elif model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
        elif model_name == 'resnet152':
            model = models.resnet152(pretrained=True)
        elif model_name == 'vgg19':
            model = models.vgg19(pretrained=True)
        else:
            raise ValueError(f"Unknown model architecture: {model_name}")

        # Freeze pre-trained layers
        for param in model.parameters():
            param.requires_grad = False

        # Modify the final fully connected layer for classification
        if model_name in ['inception', 'resnet50', 'resnet18', 'resnet152']:
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_features, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.5),
                nn.Linear(64, num_classes),  # Classification
                nn.Softmax(dim=1)  # Use Softmax for multi-class classification
            )
        elif model_name == 'vgg19':
            num_features = model.classifier[6].in_features
            model.classifier[6] = nn.Sequential(
                nn.Linear(num_features, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.5),
                nn.Linear(64, num_classes),  # Classification
                nn.Softmax(dim=1)  # Use Softmax for multi-class classification
            )
        elif model_name == 'xception':
            num_features = model.get_classifier().in_features
            model.fc = nn.Sequential(
                nn.Linear(num_features, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.5),
                nn.Linear(64, num_classes),  # Classification
                nn.Softmax(dim=1)  # Use Softmax for multi-class classification
            )

        return model

    # Determine the number of classes
    num_classes = len(full_dataset.class_names)

    # Load the selected model
    model = get_model(args.model, num_classes)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    num_epochs = args.epochs
    best_val_accuracy = 0.0
    best_model_path = f"best_model_{args.model}.pth"

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_accuracy = correct / total

        print(f"Epoch {epoch+1}/{num_epochs}, Model: {args.model}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Save the best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)

    # Load the best model
    model.load_state_dict(torch.load(best_model_path))

    # Evaluation
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())

    # Convert lists to numpy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Print classification report and confusion matrix
    print(f"\nClassification Report for Model: {args.model}")
    print(classification_report(true_labels, predicted_labels, target_names=full_dataset.class_names))

    print(f"\nConfusion Matrix for Model: {args.model}")
    print(confusion_matrix(true_labels, predicted_labels))

    if os.path.exists(best_model_path):
        os.remove(best_model_path)
        print(f"Deleted the model file: {best_model_path}")
