import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import argparse
import random
import timm

print("mean_plantvill_torchm")

# Argument parser for model selection
parser = argparse.ArgumentParser(description="Train different ResNet architectures on the PlantVillage dataset")
parser.add_argument('--model', type=str, required=True, help='Model architecture to use: resnet50, resnet18, resnet152')
args = parser.parse_args()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Directories
plant_village_dir = "/cs/home/psxjs24/data/PlantVillage/train"
plant_doc_dir = "/kaggle/input/plantdoc-dataset/train"
dataset_dir = "/kaggle/working/dataset/"

# Data transformations with augmentation for training
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

# Load datasets
full_dataset = datasets.ImageFolder(plant_village_dir, transform=data_transforms['train'])
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

def get_model(model_name):
    if model_name == 'resnet50':
        base_model = models.resnet50(pretrained=True)
    elif model_name == 'resnet18':
        base_model = models.resnet18(pretrained=True)
    elif model_name == 'resnet152':
        base_model = models.resnet152(pretrained=True)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")

    # Freeze pre-trained layers
    for param in base_model.parameters():
        param.requires_grad = False

    # Modify the final fully connected layer
    num_features = base_model.fc.in_features
    base_model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.5),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(0.5),
        nn.Linear(64, 38),  # Assuming 38 classes in the dataset
        nn.LogSoftmax(dim=1)
    )

    return base_model

class ResNetWithSkipConnections(nn.Module):
    def __init__(self, base_model):
        super(ResNetWithSkipConnections, self).__init__()
        self.base_model = base_model

    def forward(self, x):
        x = torch.relu(self.base_model.bn1(self.base_model.conv1(x)))
        x = self.base_model.maxpool(x)
        out1 = self.base_model.layer1(x)
        out2 = self.base_model.layer2(out1)
        out3 = self.base_model.layer3(out2)
        out4 = self.base_model.layer4(out3)
        x = self.base_model.avgpool(out4)
        x = torch.flatten(x, 1)
        x = self.base_model.fc(x)
        return x, out1, out2, out3, out4

# Custom loss function
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets, skip_connections):
        ce_loss = self.cross_entropy_loss(outputs, targets)
        # Randomly select one of the skip connections
        skip_output = skip_connections[np.random.randint(0, len(skip_connections))]
        skip_loss_component = torch.mean(skip_output)
        # Additional loss component: L2 regularization on the final output
        l2_loss_component = torch.mean(outputs**2)
        total_loss = ce_loss + 0.005 * skip_loss_component + 0.001 * l2_loss_component  # Weighting the components
        return total_loss

# Load the selected model
base_model = get_model(args.model)
model = ResNetWithSkipConnections(base_model).to(device)

# Define loss function and optimizer
criterion = CustomLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Training loop
num_epochs = 60
best_val_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, out1, out2, out3, out4 = model(inputs)
        loss = criterion(outputs, labels, [out1, out2, out3, out4])
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / train_size

    # Validation
    model.eval()
    val_running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, out1, out2, out3, out4 = model(inputs)
            loss = criterion(outputs, labels, [out1, out2, out3, out4])

            val_running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_running_loss / val_size
    val_accuracy = correct / total

    print(f"Epoch {epoch+1}/{num_epochs}, Model: {args.model}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Save the best model based on validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_path = f"best_model_{args.model}.pth"
        torch.save(model.state_dict(), best_model_path)

# Load the best model
model.load_state_dict(torch.load(best_model_path))

# Evaluation on validation set
model.eval()
true_labels = []
predicted_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, _, _, _, _ = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

# Print classification report and confusion matrix
print(f"\nClassification Report for Model: {args.model}")
print(classification_report(true_labels, predicted_labels))

print(f"\nConfusion Matrix for Model: {args.model}")
print(confusion_matrix(true_labels, predicted_labels))


if os.path.exists(best_model_path):
    os.remove(best_model_path)
    print(f"Deleted the model file: {best_model_path}")
