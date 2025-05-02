import argparse, os, torch, numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold
from models import get_model_with_channel_attention, ResNetWithSkipConnections
from datasets import ImageDataset
from losses import get_loss_function
from evaluation.tests import evaluate_model 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, *skips = model(inputs)
        loss = criterion(outputs, labels, skips)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def validate_epoch(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, *skips = model(inputs)
            loss = criterion(outputs, labels, skips)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def train_main(args):
    print(f"Training with {args.model} and {args.loss} loss on {args.dataset}...")
    dataset = ImageDataset(args.dataset)
    class_map = dataset.class_to_idx
    print("Class mapping:", class_map)

    if args.cross_val:
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f"Fold {fold+1}/5")
            run_training(args, dataset, train_idx, val_idx, fold=fold)
    else:
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
        run_training(args, dataset, train_idx, val_idx)


def run_training(args, dataset, train_idx, val_idx, fold=None):
    model = get_model_with_channel_attention(args.model, num_classes=2)
    model = ResNetWithSkipConnections(model).to(device)

    criterion = get_loss_function(args.loss)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size)

    best_acc, best_path = 0, f"best_{args.model}_fold{fold}.pth" if fold is not None else f"best_{args.model}.pth"

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)

        print(f"[Epoch {epoch+1}] Train: Loss={tr_loss:.4f}, Acc={tr_acc:.4f} | Val: Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)

    print("Best validation accuracy:", best_acc)
    evaluate_model(model, val_loader, criterion, class_names=dataset.classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet18", "resnet50", "resnet152"])
    parser.add_argument("--loss", type=str, default="mares", choices=["mares", "beta_mares", "maresplus", "multi", "ce"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cross_val", action="store_true", help="Enable 5-fold cross validation")
    args = parser.parse_args()

    train_main(args)
