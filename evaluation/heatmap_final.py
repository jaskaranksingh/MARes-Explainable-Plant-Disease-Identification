import os
import torch
import random
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import os
from PIL import Image
import timm, torchmetrics
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Training script with dataset, model, and epochs as arguments')
    
    # Adding arguments
    parser.add_argument('--dataset', type=str, default="tomato", choices=["tomato", "apple"], help='Dataset to use: tomato or apple')
    parser.add_argument('--model', type=str, default="rexnet_150", choices=["resnet18", "resnet50", "resnet152"], help='Model to use for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    
    return parser.parse_args()



class CustomDataset(Dataset):
    
    def __init__(self, root, transformations=None, classes_to_keep='tomato'):
        self.transformations = transformations
        self.root = root
        self.im_paths = []
        self.labels = []
        self.cls_names = {}
        
        if classes_to_keep == 'tomato':
            classes_to_do = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
                             'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
                             'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
        else:
            classes_to_do = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy']

        for idx, cls_name in enumerate(classes_to_do):
            cls_path = os.path.join(root, cls_name)
            if os.path.isdir(cls_path):
                self.cls_names[cls_name] = idx
                for img_name in os.listdir(cls_path):
                    img_path = os.path.join(cls_path, img_name)
                    self.im_paths.append(img_path)
                    self.labels.append(idx)
        
        assert len(self.im_paths) == len(self.labels), "Mismatch between image paths and labels"

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        im_path = self.im_paths[idx]
        im = Image.open(im_path).convert("RGB")
        gt = self.labels[idx]
        if self.transformations is not None:
            im = self.transformations(im)
        return im, gt
    
# Function to get dataloaders
def get_dls(root, transformations, bs, split=[0.8, 0.2, 0.1], ns=4, dataset_type='tomato'):
    dataset = CustomDataset(root=root, transformations=transformations, classes_to_keep=dataset_type)
    
    # Calculate lengths for train, validation, and test splits
    total_len = len(dataset)
    train_len = int(split[0] * total_len)
    val_len = int(split[1] * total_len)
    test_len = total_len - train_len - val_len
    
    # Split the dataset
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])
    
    # Create DataLoaders
    tr_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=ns)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=ns)
    ts_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=ns)
    
    return tr_dl, val_dl, ts_dl, dataset.cls_names

# Training setup
def train_setup(model_name, classes):
    m = timm.create_model(model_name, pretrained=True, num_classes=len(classes)).to("cuda").eval()
    epochs = args.epochs
    device = "cuda"
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=m.parameters(), lr=3e-4)
    return m, epochs, device, loss_fn, optimizer




if __name__ == "__main__":

    args = parse_args()


    root = "/cs/home/psxjs24/data/PlantVillage/background"
    mean, std, im_size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 256
    tfs = T.Compose([T.Resize((im_size, im_size)), T.ToTensor()])
    tr_dl, val_dl, ts_dl, classes = get_dls(root=root, transformations=tfs, bs=4, dataset_type=args.dataset)

    print(len(tr_dl))
    print(len(val_dl))
    print(len(ts_dl))
    print(classes)


    m, epochs, device, loss_fn, optimizer = train_setup(args.model, classes)
    def to_device(batch, device): return batch[0].to(device), batch[1].to(device)
    def get_metrics(model, ims, gts, loss_fn, epoch_loss, epoch_acc, epoch_f1): preds = model(ims); loss = loss_fn(preds, gts); return loss, epoch_loss + (loss.item()), epoch_acc + (torch.argmax(preds, dim = 1) == gts).sum().item(), epoch_f1 + f1_score(preds, gts)

    f1_score = torchmetrics.F1Score(task = "multiclass", num_classes = len(classes)).to(device)
    save_prefix, save_dir = "frutisa", "saved_models"
    print("Start training...")
    best_acc, best_loss, threshold, not_improved, patience = 0, float("inf"), 0.01, 0, 5
    tr_losses, val_losses, tr_accs, val_accs, tr_f1s, val_f1s = [], [], [], [], [], []

    best_loss = float('inf')
        

    for epoch in range(epochs):

        epoch_loss, epoch_acc, epoch_f1 = 0, 0, 0
        for idx, batch in tqdm(enumerate(tr_dl)):

            ims, gts = to_device(batch, device)

            loss, epoch_loss, epoch_acc, epoch_f1 = get_metrics(m, ims, gts, loss_fn, epoch_loss, epoch_acc, epoch_f1)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        tr_loss_to_track = epoch_loss / len(tr_dl)
        tr_acc_to_track  = epoch_acc  / len(tr_dl.dataset)
        tr_f1_to_track   = epoch_f1   / len(tr_dl)
        tr_losses.append(tr_loss_to_track); tr_accs.append(tr_acc_to_track); tr_f1s.append(tr_f1_to_track)

        print(f"{epoch + 1}-epoch train process is completed!")
        print(f"{epoch + 1}-epoch train loss          -> {tr_loss_to_track:.3f}")
        print(f"{epoch + 1}-epoch train accuracy      -> {tr_acc_to_track:.3f}")
        print(f"{epoch + 1}-epoch train f1-score      -> {tr_f1_to_track:.3f}")

        m.eval()
        with torch.no_grad():
            val_epoch_loss, val_epoch_acc, val_epoch_f1 = 0, 0, 0
            for idx, batch in enumerate(val_dl):
                ims, gts = to_device(batch, device)
                loss, val_epoch_loss, val_epoch_acc, val_epoch_f1 = get_metrics(m, ims, gts, loss_fn, val_epoch_loss, val_epoch_acc, val_epoch_f1)

            val_loss_to_track = val_epoch_loss / len(val_dl)
            val_acc_to_track  = val_epoch_acc  / len(val_dl.dataset)
            val_f1_to_track   = val_epoch_f1   / len(val_dl)
            val_losses.append(val_loss_to_track); val_accs.append(val_acc_to_track); val_f1s.append(val_f1_to_track)

            print(f"{epoch + 1}-epoch validation process is completed!")
            print(f"{epoch + 1}-epoch validation loss     -> {val_loss_to_track:.3f}")
            print(f"{epoch + 1}-epoch validation accuracy -> {val_acc_to_track:.3f}")
            print(f"{epoch + 1}-epoch validation f1-score -> {val_f1_to_track:.3f}")

            if val_loss_to_track < (best_loss + threshold):
                os.makedirs(save_dir, exist_ok = True)
                best_loss = val_loss_to_track
                torch.save(m.state_dict(), f"{save_dir}/{save_prefix}_best_model.pth")
                
            else:
                not_improved += 1
                print(f"Loss value did not decrease for {not_improved} epochs")
            



    class SaveFeatures():
        """Extract pretrained activations"""
        features = None
        
        def __init__(self, m):
            self.hook = m.register_forward_hook(self.hook_fn)
            
        def hook_fn(self, module, input, output):
            self.features = ((output.cpu()).data).numpy()
            
        def remove(self): 
            self.hook.remove()

    def getCAM(conv_fs, linear_weights, class_idx):
        bs, chs, h, w = conv_fs.shape
        cam = linear_weights[class_idx].dot(conv_fs[0, :, :, ].reshape((chs, h * w)))
        cam = cam.reshape(h, w)
        return (cam - np.min(cam)) / np.max(cam)  # Normalize CAM to range [0, 1]

    def tensor_2_im(tensor):
        # Convert tensor to numpy image
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)  # Remove the batch dimension if it's there
        image = tensor.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        return Image.fromarray(image)

    def to_device(batch, device):
        # Move the batch to the specified device
        images, labels = batch
        return images.to(device), labels.to(device)

    def inference(model, device, test_dl, num_ims, save_dir, final_conv, fc_params, cls_names=None):
        weight = np.squeeze(fc_params[0].cpu().data.numpy())
        activated_features = SaveFeatures(final_conv)
        
        acc = 0
        preds, scores, images, lbls = [], [], [], []
        
        for idx, batch in tqdm(enumerate(test_dl)):
            im, gt = to_device(batch, device)
            output = model(im)
            score, cls = torch.topk(output, k=2)
            top1_cls = cls[:, 0]  # Get the top-1 prediction
            
            acc += (top1_cls == gt).sum().item()
            images.extend(im.cpu())  # Unpack batch into individual images
            preds.extend(top1_cls.cpu().numpy())  # Unpack batch into individual predictions
            scores.extend(torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy())  # Detach before converting to NumPy
            lbls.extend(gt.cpu().numpy().tolist())
        
        print(f"Accuracy of the model on the test data -> {(acc / len(test_dl.dataset)):.3f}")
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for idx in range(num_ims):
            im = images[idx]
            pred_idx = int(preds[idx])
            heatmap = getCAM(activated_features.features, weight, pred_idx)
            
            # Resize heatmap to match original image size using PyTorch
            heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((im.shape[2], im.shape[1]), Image.LINEAR))
            original_img = tensor_2_im(im.unsqueeze(0))  # Get the original image
            
            # Save Original Image
            original_filename = f"img{idx+1}_Original_{cls_names[int(lbls[idx])]}.png"
            original_img.save(os.path.join(save_dir, original_filename))

            # Save Heatmap Image
            plt.figure()
            plt.imshow(heatmap_resized, cmap='jet')
            plt.axis("off")
            # plt.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=plt.gca(), fraction=0.03, pad=0.0)
            heatmap_filename = f"img{idx+1}_Heatmap.png"
            plt.savefig(os.path.join(save_dir, heatmap_filename), dpi=800, bbox_inches='tight', pad_inches=0)
            plt.close()

            # Save Superimposed Image
            plt.figure()
            plt.imshow(original_img)
            plt.imshow(heatmap_resized, alpha=0.35, cmap='jet')
            plt.axis("off")
            pred_lbl = cls_names[pred_idx]
            superimposed_filename = f"img{idx+1}_PRED_{pred_lbl}_{(scores[idx][pred_idx] * 100):.2f}%.png"
            plt.savefig(os.path.join(save_dir, superimposed_filename), dpi=800, bbox_inches='tight', pad_inches=0)
            plt.close()
        
        activated_features.remove()

    # Load model and run inference
    m.load_state_dict(torch.load(f"{save_dir}/{save_prefix}_best_model.pth"))
    m.to(device)
    m.eval()
    final_conv, fc_params = m.features[-1], list(m.head.fc.parameters())

    # Specify a directory to save the images
    output_dir = "output_images_trainapp"
    inference(model=m, device=device, test_dl=tr_dl, num_ims=500, save_dir=output_dir, cls_names=list(classes.keys()), final_conv=final_conv, fc_params=fc_params)


    output_dir = "output_imagesapp"
    inference(model=m, device=device, test_dl=val_dl, num_ims=500, save_dir=output_dir, cls_names=list(classes.keys()), final_conv=final_conv, fc_params=fc_params)

