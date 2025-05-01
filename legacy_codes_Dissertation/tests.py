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

# Import statistical packages
from scipy import stats
from statsmodels.stats.weightstats import ztest
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind, f_oneway, mannwhitneyu, ks_2samp, fisher_exact


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
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, cohen_kappa_score
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Import the ReduceLROnPlateau scheduler

# Import additional statistical packages
from scipy import stats
from scipy.stats import spearmanr, pearsonr, wilcoxon, ttest_rel, ks_2samp, fisher_exact, mannwhitneyu
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import brier_score_loss


 
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("-")
print("-")
print("-")
print("-")
print("-")
print("-")



print(f"Using device: {device}")

parser = argparse.ArgumentParser(description="Choose dataset: apple or tomato")
parser.add_argument('--dataset', type=str, choices=['apple', 'tomato'], required=True, help="Dataset to use (apple or tomato)")
args = parser.parse_args()

# Set root_dir based on the dataset_flag passed from command line
if args.dataset == "apple":
    root_dir = "/cs/home/psxjs24/data/apple_binary/processed_images"  # Update this path for the apple dataset
elif args.dataset == "tomato":
    root_dir = "/cs/home/psxjs24/data/tomato_binary/data/background"  # Update this path for the tomato dataset




print(f"Selected dataset: {args.dataset}")




print(f"non aug")
print("one block - after")


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
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class FeatureAttention(nn.Module):
    def __init__(self, in_features, ratio=8):
        super(FeatureAttention, self).__init__()
        self.fc1 = nn.Linear(in_features, in_features // ratio, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features // ratio, in_features, bias=False)

    def forward(self, x):
        attn = torch.sigmoid(self.fc2(self.relu(self.fc1(x))))
        return x * attn  # Element-wise multiplication of attention weights with input features



# Modify ResNet50 to include two Channel Attention blocks
from torchvision import models

def get_model_with_channel_attention(model_name, num_classes):
    if model_name == 'resnet50':
        # Load pre-trained ResNet50
        # model = models.resnet50(pretrained=True)


        # Freeze all the pre-trained layers
        for param in model.parameters():
            param.requires_grad = False


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

    elif model_name == 'resnet152':
        # Load pre-trained ResNet50
        model = models.resnet152(pretrained=True)


        # Freeze all the pre-trained layers
        for param in model.parameters():
            param.requires_grad = False



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
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")

    return model


# Determine the number of classes
# Determine the number of classes
num_classes = len(full_dataset.class_names)

# Load the model (ResNet50 with two channel attention blocks and frozen pre-trained layers)
model_name = 'resnet152'
model = get_model_with_channel_attention(model_name, num_classes)
model = model.to(device)

# Define loss function and optimizer (only train attention and final layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.005)

# Learning rate scheduler using ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True, threshold=0.0001, cooldown=1, min_lr=1e-5)

# Training loop with ReduceLROnPlateau
num_epochs = 20
best_val_accuracy = 0.0
best_model_path = f"best_model6_{model_name}.pth"

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


try:
    t_stat, p_val_ttest = ttest_rel(true_labels, predicted_labels)
    print(f"\nPaired T-Test: t-statistic={t_stat}, p-value={p_val_ttest}")
except Exception as e:
    print(f"Error in Paired T-Test calculation: {e}")

# 2. Spearman's Rank Correlation - Monotonic relationship between true and predicted labels
try:
    spearman_corr, p_val_spearman = spearmanr(true_labels, predicted_labels)
    print(f"\nSpearman's Rank Correlation: Spearman Corr={spearman_corr}, p-value={p_val_spearman}")
except Exception as e:
    print(f"Error in Spearman's Rank Correlation calculation: {e}")

# 3. Pearson Correlation Coefficient - Linear relationship between true and predicted labels
try:
    pearson_corr, p_val_pearson = pearsonr(true_labels, predicted_labels)
    print(f"\nPearson Correlation Coefficient: Pearson Corr={pearson_corr}, p-value={p_val_pearson}")
except Exception as e:
    print(f"Error in Pearson Correlation calculation: {e}")

# 4. Wilcoxon Signed-Rank Test - Non-parametric test for related samples
try:
    wilcoxon_stat, p_val_wilcoxon = wilcoxon(true_labels, predicted_labels)
    print(f"\nWilcoxon Signed-Rank Test: Stat={wilcoxon_stat}, p-value={p_val_wilcoxon}")
except Exception as e:
    print(f"Error in Wilcoxon Signed-Rank Test calculation: {e}")

# 5. McNemar's Test - Compare performance of two models with paired nominal data
try:
    contingency_table = confusion_matrix(true_labels, predicted_labels)
    mcnemar_result = mcnemar(contingency_table)
    print(f"\nMcNemar's Test: Statistic={mcnemar_result.statistic}, p-value={mcnemar_result.pvalue}")
except Exception as e:
    print(f"Error in McNemar's Test calculation: {e}")

# 6. Brier Score - Measures the mean squared difference between predicted probabilities and actual outcomes
try:
    # Assuming predicted_labels are probabilities for binary classification
    brier_score = brier_score_loss(true_labels, predicted_labels)
    print(f"\nBrier Score: {brier_score}")
except Exception as e:
    print(f"Error in Brier Score calculation: {e}")

# 7. Matthews Correlation Coefficient (MCC) - Quality measure for binary classification
try:
    mcc = matthews_corrcoef(true_labels, predicted_labels)
    print(f"\nMatthews Correlation Coefficient (MCC): {mcc}")
except Exception as e:
    print(f"Error in Matthews Correlation Coefficient calculation: {e}")

# 8. Cohen's Kappa - Measures agreement between true and predicted labels
try:
    kappa = cohen_kappa_score(true_labels, predicted_labels)
    print(f"\nCohen's Kappa: {kappa}")
except Exception as e:
    print(f"Error in Cohen's Kappa calculation: {e}")

# 9. Kolmogorov-Smirnov (KS) test - Compares the distribution of true vs predicted labels
try:
    ks_stat, p_val_ks = ks_2samp(true_labels, predicted_labels)
    print(f"\nKolmogorov-Smirnov Test: KS-stat={ks_stat}, p-value={p_val_ks}")
except Exception as e:
    print(f"Error in Kolmogorov-Smirnov Test calculation: {e}")

# 10. Fisher's Exact Test - Test for significance in 2x2 contingency tables
try:
    tp = np.sum((true_labels == 1) & (predicted_labels == 1))
    fp = np.sum((true_labels == 0) & (predicted_labels == 1))
    fn = np.sum((true_labels == 1) & (predicted_labels == 0))
    tn = np.sum((true_labels == 0) & (predicted_labels == 0))
    contingency_table = [[tp, fp], [fn, tn]]
    odds_ratio, p_val_fisher = fisher_exact(contingency_table)
    print(f"\nFisher's Exact Test: Odds ratio={odds_ratio}, p-value={p_val_fisher}")
except Exception as e:
    print(f"Error in Fisher's Exact Test calculation: {e}")

# 11. Mann-Whitney U Test - Non-parametric test for comparing two independent samples
try:
    u_stat, p_val_mannwhitney = mannwhitneyu(true_labels, predicted_labels)
    print(f"\nMann-Whitney U Test: U-statistic={u_stat}, p-value={p_val_mannwhitney}")
except Exception as e:
    print(f"Error in Mann-Whitney U Test calculation: {e}")

# Print classification report and confusion matrix
print(f"\nClassification Report for Model: {model_name}")
try:
    print(classification_report(true_labels, predicted_labels))
except Exception as e:
    print(f"Error in Classification Report generation: {e}")

print(f"\nConfusion Matrix for Model: {model_name}")
try:
    print(confusion_matrix(true_labels, predicted_labels))
except Exception as e:
    print(f"Error in Confusion Matrix generation: {e}")

# Remove the best model file after evaluation
if os.path.exists(best_model_path):
    os.remove(best_model_path)
    print(f"Deleted the best model file: {best_model_path}")



# Print classification report and confusion matrix
print(f"\nClassification Report for Model: {model_name}")
print(classification_report(true_labels, predicted_labels, target_names=full_dataset.class_names))

print(f"\nConfusion Matrix for Model: {model_name}")
print(confusion_matrix(true_labels, predicted_labels))

if os.path.exists(best_model_path):
    os.remove(best_model_path)
    print(f"Deleted the best model file: {best_model_path}")
