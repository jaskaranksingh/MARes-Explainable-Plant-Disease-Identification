
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
import timm
import argparse

# Argument parser for model selection, plant species, and debug mode
parser = argparse.ArgumentParser(description="Train CNN architectures on specific plant species for healthy vs. not healthy classification")
parser.add_argument('--model', type=str, required=True, help='Model architecture to use: inception, xception, resnet50, resnet18, resnet152, vgg19')
parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train the model')
parser.add_argument('--lr', type=float, default=0.005, help='Learning rate for the optimizer')
parser.add_argument('--debug', action='store_true', help='If set, do not train the model, just print the crop, model name, and dataset size')
args = parser.parse_args()


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Nested dictionary of all 38 folders categorized by plant species with healthy and diseased classes
plant_categories = {
    'apple': {
        'healthy': ['Apple___healthy'],
        'diseased': ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust']
    },
    'blueberry': {
        'healthy': ['Blueberry___healthy'],
        'diseased': []
    },
    'cherry': {
        'healthy': ['Cherry_(including_sour)___healthy'],
        'diseased': ['Cherry_(including_sour)___Powdery_mildew']
    },
    'corn': {
        'healthy': ['Corn_(maize)___healthy'],
        'diseased': ['Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight']
    },
    'grape': {
        'healthy': ['Grape___healthy'],
        'diseased': ['Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)']
    },
    'orange': {
        'healthy': [],
        'diseased': ['Orange___Haunglongbing_(Citrus_greening)']
    },
    'peach': {
        'healthy': ['Peach___healthy'],
        'diseased': ['Peach___Bacterial_spot']
    },
    'pepper': {
        'healthy': ['Pepper,_bell___healthy'],
        'diseased': ['Pepper,_bell___Bacterial_spot']
    },
    'potato': {
        'healthy': ['Potato___healthy'],
        'diseased': ['Potato___Early_blight', 'Potato___Late_blight']
    },
    'raspberry': {
        'healthy': ['Raspberry___healthy'],
        'diseased': []
    },
    'soybean': {
        'healthy': ['Soybean___healthy'],
        'diseased': []
    },
    'squash': {
        'healthy': [],
        'diseased': ['Squash___Powdery_mildew']
    },
    'strawberry': {
        'healthy': ['Strawberry___healthy'],
        'diseased': ['Strawberry___Leaf_scorch']
    },
    'tomato': {
        'healthy': ['Tomato___healthy'],
        'diseased': ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']
    }
}

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

# Custom dataset to filter and relabel images
class PlantVillageDataset(datasets.ImageFolder):
    def __init__(self, root, class_mapping, transform=None):
        super().__init__(root, transform=transform)
        self.samples = [(path, class_mapping[os.path.basename(os.path.dirname(path))]) for path, _ in self.samples if os.path.basename(os.path.dirname(path)) in class_mapping]

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

# Paths to train and validation datasets
train_dir = f"/cs/home/psxjs24/data/PlantVillage/color"

class_mapping = {}

for plant in plant_categories.keys():
    if plant_categories[plant]['healthy']:  # Check if healthy list is not empty
        class_mapping[plant_categories[plant]['healthy'][0]] = 0  # Healthy class
    
    for diseased_class in plant_categories[plant]['diseased']:
        class_mapping[diseased_class] = 1  # Diseased class





full_dataset = PlantVillageDataset(train_dir, class_mapping, transform=data_transforms['train'])


total_images = len(full_dataset)
print(f"Total number of images: {total_images}")



labels = [label for _, label in full_dataset.samples]

k_folds = 5
skf = StratifiedKFold(n_splits=5)






print("-")
print("-")
print("-")
print("-")
print("-")
print("-")

if args.debug:
    # Debug mode: Print dataset size and exit
    exit()

# Function to select model
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



class MultiLoss(nn.Module):
    def __init__(self, gamma1=0.001, gamma2=0.01, delta1=0.0001, delta2=0.001):
        super(MultiLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.alpha = nn.Parameter(torch.tensor(0.00), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        
        # Define the bounds for scaling
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.delta1 = delta1
        self.delta2 = delta2

    def forward(self, outputs, targets, skip_connections):
        ce_loss = self.cross_entropy_loss(outputs, targets)
        
        # Use two skip connections
        skip_output1 = skip_connections[2]
        skip_output2 = skip_connections[3]
        skip_loss_component1 = torch.mean(skip_output1) / skip_output1.numel()
        skip_loss_component2 = torch.mean(skip_output2) / skip_output2.numel()
        
        # Additional loss component: L2 regularization on the final output
        l2_loss_component = torch.mean(outputs**2)
        
        # Scale alpha and beta using sigmoid function
        alpha_scaled = self.gamma1 + (self.gamma2 - self.gamma1) * torch.sigmoid(self.alpha)
        beta_scaled = self.delta1 + (self.delta2 - self.delta1) * torch.sigmoid(self.beta)
        
        # Dynamically weighted total loss
        total_loss = ce_loss + alpha_scaled * (skip_loss_component1 + skip_loss_component2) + beta_scaled * l2_loss_component
        
        return total_loss



for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    print(f'\nFold {fold + 1}/{k_folds}')
    
    # Subset dataset for this fold
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    
    # Data loaders for this fold
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=4)

    num_classes = 2

    # Model selection (choose ResNet18, ResNet50, or ResNet152)
    model_name = args.model  # Model name selected via command-line argument
    base_model = get_model_with_channel_attention(model_name, num_classes)
    model = ResNetWithSkipConnections(base_model).to(device)





    # Define loss function and optimizer
    criterion = MultiLoss().to(device)
    optimizer = optim.Adam([
        {'params': model.parameters()},
        {'params': criterion.parameters(), 'lr': 0.005}  # Include the learnable parameters in the optimizer
    ], lr=0.005)

    # Training loop
    num_epochs = args.epochs  # Use epochs from argument
    best_val_accuracy = 0.0

    best_model_path = f"best_model_{args.model}.pth"

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    true_labels_all = []
    predicted_probs_all = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, out1, out2, out3, out4 = model(inputs)
            loss = criterion(outputs, labels, [out1, out2, out3, out4])
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_subset)
        train_accuracy = correct_train / total_train

        # Store the training loss and accuracy
        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        true_labels_epoch = []
        predicted_probs_epoch = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, out1, out2, out3, out4 = model(inputs)
                loss = criterion(outputs, labels, [out1, out2, out3, out4])
                val_running_loss += loss.item() * inputs.size(0)
                
                # Convert log-probabilities to probabilities
                probs = torch.exp(outputs)
                
                _, predicted = torch.max(probs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                true_labels_epoch.extend(labels.cpu().numpy())
                predicted_probs_epoch.extend(probs.cpu().numpy()[:, 1])  # Use the probability of the positive class

        val_loss = val_running_loss / len(val_subset)
        val_accuracy = correct / total

        # Store the validation loss and accuracy
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Store true labels and predicted probabilities for ROC curve
        true_labels_all.extend(true_labels_epoch)
        predicted_probs_all.extend(predicted_probs_epoch)

        print(f"Epoch {epoch+1}/{num_epochs}, Model: {args.model}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Save the best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
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




    print(f"\nClassification Report for Model: {args.model}")
    print(classification_report(true_labels, predicted_labels))

    print(f"\nConfusion Matrix for Model: {args.model}")
    print(confusion_matrix(true_labels, predicted_labels))



    if os.path.exists(best_model_path):
        os.remove(best_model_path)
        print(f"Deleted the model file: {best_model_path}")
