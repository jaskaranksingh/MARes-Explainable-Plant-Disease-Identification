import os
import numpy as np
import argparse
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Import the ReduceLROnPlateau scheduler


parser = argparse.ArgumentParser(description="Train a ResNet model with channel attention")
parser.add_argument('--model', type=str, default='resnet50', choices=['resnet18', 'resnet50', 'resnet152'],
                    help='Select the ResNet model architecture (resnet18, resnet50, or resnet152)')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
parser.add_argument('--lr', type=float, default=0.005, help='Learning rate for the optimizer')
args = parser.parse_args()



# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

root_dir = "/cs/home/psxjs24/data/tomato_binary/data/background"

# Dataset definition
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
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

full_dataset = CustomDataset(root_dir=root_dir, transform=data_transforms['train'])

# Split the dataset into training and validation sets
indices = list(range(len(full_dataset)))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

train_dataset = Subset(CustomDataset(root_dir=root_dir, transform=data_transforms['train']), train_indices)
val_dataset = Subset(CustomDataset(root_dir=root_dir, transform=data_transforms['val']), val_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=True)

print(len(train_dataset))
print(len(val_dataset))

# Channel Attention block definition
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return torch.sigmoid(out) * x

# Modify ResNet models to include two Channel Attention blocks
def get_model_with_channel_attention(model_name, num_classes):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=True)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")

    # Freeze all the pre-trained layers
    for param in model.parameters():
        param.requires_grad = False

    # Insert Channel Attention after layer3 and layer4
    model.layer3 = nn.Sequential(
        model.layer3,  # Keep the pre-trained layer3
        ChannelAttention(1024)  # Add Channel Attention block after layer3
    )
    model.layer4 = nn.Sequential(
        model.layer4,  # Keep the pre-trained layer4
        ChannelAttention(2048)  # Add Channel Attention block after layer4
    )

    # Replace the final fully connected layer with a custom classification head
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
        nn.Linear(64, num_classes),
        nn.Softmax(dim=1)
    )

    return model

# Determine the number of classes
num_classes = len(full_dataset.class_names)

# Model selection (choose ResNet18, ResNet50, or ResNet152)
model_name = args.model  # Model name selected via command-line argument
model = get_model_with_channel_attention(model_name, num_classes)
model = model.to(device)

# Define loss function and optimizer (only train attention and final layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

# Learning rate scheduler using ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True, threshold=0.0001, cooldown=1, min_lr=1e-5)

# Training loop with ReduceLROnPlateau
num_epochs = args.epochs  # Use the number of epochs from command-line arguments
best_val_accuracy = 0.0
best_model_path = f"best_model_{model_name}.pth"

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

    print(f"Epoch {epoch+1}/{num_epochs}, Model: {model_name}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Update the scheduler based on validation loss
    scheduler.step(val_loss)  # Scheduler steps based on validation loss

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
print(f"\nClassification Report for Model: {model_name}")
print(classification_report(true_labels, predicted_labels, target_names=full_dataset.class_names))

print(f"\nConfusion Matrix for Model: {model_name}")
print(confusion_matrix(true_labels, predicted_labels))

# Cleanup
if os.path.exists(best_model_path):
    os.remove(best_model_path)
    print(f"Deleted the best model file: {best_model_path}")
