import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import timm
import argparse

import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split, DistributedSampler
import timm
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.utils import make_grid



# Argument parser for model selection
parser = argparse.ArgumentParser(description="Train different CNN architectures on the PlantVillage dataset")
parser.add_argument('--model', type=str, required=True, help='Model architecture to use: inception, xception, resnet50, resnet18, resnet152, vgg19')
args = parser.parse_args()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Non Augmentation")




def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# Define base directory for saving classification results
base_dir = "classification_results"
create_directory(base_dir)

# Function to save images
def save_sample(sample, label, prediction, count_dict):
    true_label_name = train_dataset.dataset.classes[label]
    predicted_label_name = train_dataset.dataset.classes[prediction]
    subdir_name = f"true-{true_label_name}_classified-{predicted_label_name}"
    subdir_path = os.path.join(base_dir, subdir_name)
    create_directory(subdir_path)

    # Limit to 50 images per folder
    if count_dict[subdir_name] < 50:
        img_name = f"sample_{count_dict[subdir_name]}.jpg"
        img_path = os.path.join(subdir_path, img_name)
        save_image(sample, img_path)
        count_dict[subdir_name] += 1

# Helper to convert tensor to image and save
def save_image(tensor, path):
    from torchvision.utils import save_image
    save_image(tensor, path)





# Directories
plant_village_dir = "/cs/home/psxjs24/data/PlantVillage/color"
plant_doc_dir = "/kaggle/input/plantdoc-dataset/train"
dataset_dir = "/kaggle/working/dataset/"

# Data transformations with augmentation for training
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
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

classification_counts = {f"true-{cls}_classified-{cls2}": 0 for cls in train_dataset.dataset.classes for cls2 in train_dataset.dataset.classes}

# Function to select model
def get_model(model_name):
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

    # Modify the final fully connected layer
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
            nn.Linear(64, 38),  # Assuming 38 classes in the dataset
            nn.LogSoftmax(dim=1)
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
            nn.Linear(64, 38),  # Assuming 38 classes in the dataset
            nn.LogSoftmax(dim=1)
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
            nn.Linear(64, 38),  # Assuming 38 classes in the dataset
            nn.LogSoftmax(dim=1)
        )

    return model

# Load the selected model
model = get_model(args.model)


model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 5
best_val_accuracy = 0.0


optimizer = optim.Adam(model.parameters(), lr=0.005)


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

    epoch_loss = running_loss / train_size

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
true_labels = []
predicted_labels = []

model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        for i in range(inputs.size(0)):
            true_label = labels[i].item()
            predicted_label = predicted[i].item()
            if true_label != predicted_label or (true_label == predicted_label and classification_counts[f"true-{train_dataset.dataset.classes[true_label]}_classified-{train_dataset.dataset.classes[predicted_label]}"] < 50):
                save_sample(inputs[i].cpu(), true_label, predicted_label, classification_counts)

# Print classification report and confusion matrix
print(f"\nClassification Report for Model: {args.model}")
print(classification_report(true_labels, predicted_labels))

print(f"\nConfusion Matrix for Model: {args.model}")
print(confusion_matrix(true_labels, predicted_labels))