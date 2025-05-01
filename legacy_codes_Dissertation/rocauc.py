import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, Subset
import timm
import argparse
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


import random
import time
import pandas as pd


# Argument parser for model selection
parser = argparse.ArgumentParser(description="Train different CNN architectures on the dataset")
parser.add_argument('--model', type=str, required=True, help='Model architecture to use: inception, xception, resnet50, resnet18, resnet152, vgg19')
args = parser.parse_args()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("background")
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

root_dir = "/cs/home/psxjs24/data/tomato_binary/data/background"  # Update this path to your dataset directory

full_dataset = CustomDataset(root_dir=root_dir, transform=data_transforms['train'])

# Split the dataset into training and validation sets
train_indices, val_indices = train_test_split(list(range(len(full_dataset))), test_size=0.2, random_state=42)

# Create subsets with appropriate transforms
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(CustomDataset(root_dir=root_dir, transform=data_transforms['val']), val_indices)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, drop_last=True)



print(f"Total dataset size: {len(full_dataset)}")
print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

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
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),  # Classification
            nn.Softmax(dim=1)  # Use Softmax for multi-class classification
        )
    elif model_name == 'vgg19':
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),  # Classification
            nn.Softmax(dim=1)  # Use Softmax for multi-class classification
        )
    elif model_name == 'xception':
        num_features = model.get_classifier().in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes),  # Classification
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
optimizer = optim.Adam(model.parameters(), lr=0.00005)

# Training loop
num_epochs = 30
best_val_accuracy = 0.0
best_model_path = f"best_model_{args.model}.pth"



train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []


for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        _, preds = torch.max(outputs, 1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_accuracy = correct_train / total_train

    # Store the training loss and accuracy
    train_losses.append(epoch_loss)
    train_accuracies.append(train_accuracy)

    # Validation
    model.eval()
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

    val_loss = val_running_loss / len(val_loader.dataset)
    val_accuracy = correct_val / total_val

    # Store the validation loss and accuracy
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Save the best model based on validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), best_model_path)

# Load the best model
model.load_state_dict(torch.load(best_model_path))


# Generate a unique identifier using a combination of timestamp and random number
unique_id = f"{int(time.time())}_{random.randint(1000, 9999)}"

# Filename with unique identifier
metrics_file = f"lossauc/metrics_{args.model}_{args.plant}_{unique_id}.csv"

# Ensure that the lengths are consistent before saving
assert len(train_losses) == len(val_losses), "Mismatch in train and val losses"
assert len(train_accuracies) == len(val_accuracies), "Mismatch in train and val accuracies"

# Create a dictionary for the DataFrame
metrics = {
    'epoch': list(range(1, len(train_losses) + 1)),
    'train_loss': train_losses,
    'val_loss': val_losses,
    'train_accuracy': train_accuracies,
    'val_accuracy': val_accuracies
}

# Save the metrics DataFrame to a CSV file
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(metrics_file, index=False)
print(f"Metrics saved to {metrics_file}")





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
