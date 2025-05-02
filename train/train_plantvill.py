import os, torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.model_factory import build_model
from datasets import ImageDataset
from losses.loss_factory import get_loss_function
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    data_path = "data/PlantVillage_binary"
    batch_size = 32
    epochs = 30
    model_name = "resnet18"
    loss_function = "mares"
    use_attention = True
    lr = 1e-4

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = ImageDataset(data_path, transform=transform)

    print("Class-to-Index mapping:", dataset.class_to_idx)

    # Split dataset
    total_size = len(dataset)
    split = int(0.8 * total_size)
    train_set, val_set = torch.utils.data.random_split(dataset, [split, total_size - split])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = build_model(model_name, num_classes=2, use_attention=use_attention).to(device)
    criterion = get_loss_function(loss_function)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            outputs, skips = model(x)
            loss = criterion(outputs, y, skips)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out, _ = model(x)
                preds.extend(torch.argmax(out, 1).cpu().numpy())
                labels.extend(y.cpu().numpy())
        acc = accuracy_score(labels, preds)
        print(f"[Epoch {epoch+1}] Validation Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
