from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from models.model_factory import build_model
from datasets import ImageDataset
from losses.loss_factory import get_loss_function
import torch, numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cross_val_training(data_path="data/PlantVillage_binary", model_name="resnet50", loss_function="beta_mares"):
    dataset = ImageDataset(data_path)
    print("Class mapping:", dataset.class_to_idx)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n--- Fold {fold + 1} ---")
        model = build_model(model_name, num_classes=2, use_attention=True).to(device)
        criterion = get_loss_function(loss_function)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=32)

        for epoch in range(30):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                out, skips = model(x)
                loss = criterion(out, y, skips)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred, _ = model(x)
                pred = torch.argmax(pred, 1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"Fold {fold+1} Accuracy: {acc:.4f}")

if __name__ == "__main__":
    cross_val_training()
