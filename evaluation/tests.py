import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model(model, dataloader, criterion, class_names=None, device='cuda'):
    model.eval()
    y_true, y_pred = [], []
    loss_sum = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out, skips = model(x)
            loss = criterion(out, y, skips)
            preds = torch.argmax(out, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            loss_sum += loss.item() * x.size(0)

    acc = accuracy_score(y_true, y_pred)
    print("\n--- Evaluation Report ---")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    if class_names:
        print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))
    else:
        print("\nClassification Report:\n", classification_report(y_true, y_pred))

    return acc
