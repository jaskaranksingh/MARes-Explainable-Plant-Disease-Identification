import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import torch.nn.functional as F

def plot_roc_auc(model, dataloader, device='cuda'):
    y_true, y_score = [], []

    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            out, _ = model(x)
            probs = F.softmax(out, dim=1)
            y_score.extend(probs[:, 1].cpu().numpy())
            y_true.extend(y.cpu().numpy())

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
