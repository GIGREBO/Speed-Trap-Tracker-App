import os
import math
import copy
import random
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# 0. Reproducibility
# ============================================================
SEED = 2003
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 1. Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ============================================================
# 2. Custom Transforms & Preprocessing
# ============================================================
class HighwayROICrop:
    """
    Crops out:
        - Top portion: sky, trees, overhead bridges
        - Bottom portion: UI artifacts, car hood, compass, etc.

    Adaptive cropping adjusts the bottom ratio for wider (dashcam-style)
    aspect ratios. Your Street View images are ~640x448 after the base
    crop, aspect ratio ~1.43, so this branch won't trigger for you unless
    you start pulling images with a different camera setup.
    """
    def __init__(self, top_ratio=0.20, bottom_ratio=0.90, adaptive=True):
        self.top_ratio = top_ratio
        self.bottom_ratio = bottom_ratio
        self.adaptive = adaptive

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            h, w = img.shape[:2]
            if self.adaptive:
                aspect_ratio = w / h
                bottom_ratio = 0.92 if aspect_ratio > 1.6 else self.bottom_ratio
            else:
                bottom_ratio = self.bottom_ratio
            top = int(h * self.top_ratio)
            bottom = int(h * bottom_ratio)
            return img[top:bottom, :, :]
        else:
            w, h = img.size
            if self.adaptive:
                aspect_ratio = w / h
                bottom_ratio = 0.92 if aspect_ratio > 1.6 else self.bottom_ratio
            else:
                bottom_ratio = self.bottom_ratio
            top = int(h * self.top_ratio)
            bottom = int(h * bottom_ratio)
            return img.crop((0, top, w, bottom))


# ============================================================
# ImageNet Normalization
# ============================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# Resolution
#
# Your crop is 640x448 (width x height), aspect ratio 1.4286.
# 224x224 (square) was stretching that crop unevenly: 2x downsample in
# height, 2.86x in width. RESIZE_HW below is 224 height x 320 width —
# an exact uniform 2x downsample in both dimensions (640/2=320, 448/2=224),
# so there's no distortion at all, just a clean half-scale.
# ============================================================
RESIZE_HW = (224, 320)  # (height, width) — torchvision convention
RESIZE_WH_CV2 = (RESIZE_HW[1], RESIZE_HW[0])  # (width, height) — cv2 convention


# ============================================================
# Training Transform
# ============================================================
train_transform = transforms.Compose([
    HighwayROICrop(top_ratio=0.20, bottom_ratio=0.90, adaptive=True),
    transforms.ToPILImage(),
    transforms.Resize(RESIZE_HW),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=5, translate=(0.04, 0.04)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
    ], p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


# ============================================================
# Validation / Test Transform
# ============================================================
eval_transform = transforms.Compose([
    HighwayROICrop(top_ratio=0.20, bottom_ratio=0.90, adaptive=True),
    transforms.ToPILImage(),
    transforms.Resize(RESIZE_HW),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


# ============================================================
# Test-Time Augmentation transforms
# Each one must differ from eval_transform, since TTA averages
# predictions across genuinely different views of the same image.
# ============================================================
tta_transforms = [
    eval_transform,
    transforms.Compose([
        HighwayROICrop(top_ratio=0.20, bottom_ratio=0.90, adaptive=True),
        transforms.ToPILImage(),
        transforms.Resize(RESIZE_HW),
        transforms.RandomHorizontalFlip(p=1.0),  # always flip
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]),
    transforms.Compose([
        HighwayROICrop(top_ratio=0.20, bottom_ratio=0.90, adaptive=True),
        transforms.ToPILImage(),
        transforms.Resize(RESIZE_HW),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]),
]


# ============================================================
# 3. Dataset
# ============================================================
class HighwayDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return {
            "images": image,
            "labels": label,
            "paths": img_path
        }


# ============================================================
# 4. Paths
# ============================================================
# Relative paths — script must be run from the directory containing
# the classifdataset/ folder (e.g. your HighwayProject root).
labels_path = "classifdataset/mldata.csv"
images_dir = "classifdataset/projectpics"


# ============================================================
# 5. Load CSV
# ============================================================
df = pd.read_csv(labels_path)
print("\nDataset size:", len(df))
print("\nCSV columns:")
print(df.columns.tolist())

# mldata.csv has no timestamp column, so "newest" below just means
# "last rows in the CSV" (i.e. whatever order you appended them in),
# same assumption your original script made.
timestamp_col = None
for col in df.columns:
    if 'time' in col.lower() or 'date' in col.lower() or 'capture' in col.lower():
        timestamp_col = col
        break

if timestamp_col:
    df = df.sort_values(timestamp_col)
    print(f"Sorted by {timestamp_col}")
else:
    print("No timestamp column found — using CSV row order (append order) as-is.")


# ============================================================
# Image Paths
# ============================================================
image_paths = [
    os.path.join(images_dir, str(name) + ".png")
    for name in df["image"].tolist()
]


# ============================================================
# Labels
# ============================================================
labels = df.iloc[:, 1:].values.astype(np.float32)
print("\nLabel shape:", labels.shape)

if labels.shape[1] != 3:
    raise ValueError(
        f"Expected exactly 3 label columns after 'image' "
        f"(Shoulder, Gore Area, Viable), but found {labels.shape[1]}."
    )


# ============================================================
# Label Statistics
# ============================================================
print("\n================ DATASET LABEL COUNTS ================")
print(f"Shoulder positives: {int(labels[:, 0].sum())}")
print(f"Shoulder negatives: {int(len(labels) - labels[:, 0].sum())}")
print(f"Gore positives: {int(labels[:, 1].sum())}")
print(f"Gore negatives: {int(len(labels) - labels[:, 1].sum())}")
print(f"Viable positives: {int(labels[:, 2].sum())}")
print(f"Viable negatives: {int(len(labels) - labels[:, 2].sum())}")

pos_weights = torch.tensor([
    len(labels) / max(labels[:, i].sum(), 1)
    for i in range(3)
]).to(device)
print(f"\nClass weights: {pos_weights.cpu().numpy()}")


# ============================================================
# 6. Data Split
# ============================================================
NUM_FORCED_TRAIN = 20

old_image_paths = image_paths[:-NUM_FORCED_TRAIN]
old_labels = labels[:-NUM_FORCED_TRAIN]

new_image_paths = image_paths[-NUM_FORCED_TRAIN:]
new_labels = labels[-NUM_FORCED_TRAIN:]

print("\nForced-training images (last rows in CSV):")
for path in new_image_paths:
    print(os.path.basename(path))

X_train_val_paths, X_test_paths, y_train_val, y_test = train_test_split(
    old_image_paths,
    old_labels,
    test_size=0.20,
    random_state=SEED,
    stratify=old_labels[:, 0]
)

X_train_paths, X_val_paths, y_train, y_val = train_test_split(
    X_train_val_paths,
    y_train_val,
    test_size=0.25,
    random_state=SEED,
    stratify=y_train_val[:, 0]
)

X_train_paths = list(X_train_paths) + list(new_image_paths)
y_train = np.concatenate([y_train, new_labels], axis=0)

print("\n================ DATA SPLIT ================")
print(f"Training images: {len(X_train_paths)}")
print(f"Validation images: {len(X_val_paths)}")
print(f"Test images: {len(X_test_paths)}")
print(f"Forced new images in training: {len(new_image_paths)}")

new_set = set(new_image_paths)
assert new_set.isdisjoint(set(X_val_paths))
assert new_set.isdisjoint(set(X_test_paths))
print("Verified: all newest images are TRAIN ONLY.")


# ============================================================
# 7. Dataset Objects
# ============================================================
train_dataset = HighwayDataset(X_train_paths, y_train, transform=train_transform)
val_dataset = HighwayDataset(X_val_paths, y_val, transform=eval_transform)
test_dataset = HighwayDataset(X_test_paths, y_test, transform=eval_transform)


# ============================================================
# 8. DataLoaders
# ============================================================
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)


# ============================================================
# 9. Model - EfficientNet-B0
# ============================================================
class MLC(nn.Module):
    def __init__(self, num_classes=3):
        super(MLC, self).__init__()
        self.efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)


num_classes = 3
model = MLC(num_classes=num_classes).to(device)
print("\n================ MODEL ================")
print(model)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


# ============================================================
# 10. Loss Function with Class Weights
# ============================================================
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)


# ============================================================
# 11. Optimizer with Different LR for Backbone vs Classifier
# ============================================================
backbone_params = []
classifier_params = []

for name, param in model.named_parameters():
    if 'classifier' in name:
        classifier_params.append(param)
    else:
        backbone_params.append(param)

optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': 1e-5},
    {'params': classifier_params, 'lr': 1e-3}
], weight_decay=1e-2)


# ============================================================
# 12. Learning Rate Scheduler with Warmup
# ============================================================
num_epochs = 50
warmup_epochs = 5


def warmup_cosine_scheduler(epoch):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    else:
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_scheduler)


# ============================================================
# 13. Early Stopping
# ============================================================
patience = 10
wait = 0
best_val_loss = float('inf')
best_weights = None
best_epoch = 0

train_losses = []
val_losses = []


# ============================================================
# 14. Training Loop (now with per-epoch + total timing, so you can
# actually measure the CPU cost of this resolution instead of guessing)
# ============================================================
print("\n================ TRAINING ================\n")
training_start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()

    model.train()
    running_train_loss = 0.0
    for batch in train_loader:
        data = batch["images"].to(device)
        targets = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_train_loss += loss.item() * data.size(0)

    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            data = batch["images"].to(device)
            targets = batch["labels"].to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            running_val_loss += loss.item() * data.size(0)

    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_weights = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        wait = 0
    else:
        wait += 1

    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]
    epoch_duration = time.time() - epoch_start_time

    print(
        f"Epoch [{epoch + 1:02d}/{num_epochs:02d}] "
        f"- Train Loss: {epoch_train_loss:.4f} "
        f"- Val Loss: {epoch_val_loss:.4f} "
        f"- LR: {current_lr:.7f} "
        f"- Time: {epoch_duration:.1f}s"
    )

    if wait >= patience:
        print(f"\nEarly stopping triggered at epoch {epoch + 1}")
        break

total_training_time = time.time() - training_start_time
print(f"\nTotal training time: {total_training_time / 60:.1f} minutes "
      f"({total_training_time / (epoch + 1):.1f}s/epoch average)")


# ============================================================
# 15. Load Best Model
# ============================================================
if best_weights is not None:
    model.load_state_dict(best_weights)
    print(f"\nLoaded best model from epoch {best_epoch + 1}")
print(f"Best validation loss: {best_val_loss:.4f}")


# ============================================================
# 16. Save Model Checkpoint
# ============================================================
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_loss': best_val_loss,
    'epoch': best_epoch
}, "cv_model_best.pth")
print("Saved model checkpoint to cv_model_best.pth")

# ONNX export wrapped in try/except so a missing `onnx` package doesn't
# take down an otherwise-successful training run.
try:
    dummy_input = torch.randn(1, 3, RESIZE_HW[0], RESIZE_HW[1]).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        "cv_model.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Exported model to ONNX format")
except Exception as e:
    print(f"ONNX export skipped ({e})")


# ============================================================
# 17. Plot Loss Curves
# ============================================================
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Model (Epoch {best_epoch + 1})')
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 18. Validation Predictions (no TTA — used only for threshold search)
# ============================================================
def get_predictions(model, loader):
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            data = batch["images"].to(device)
            targets = batch["labels"].to(device)
            probs = torch.sigmoid(model(data))
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_targets, axis=0)


val_probs, val_targets = get_predictions(model, val_loader)


# ============================================================
# 19. Optimize Threshold For F1 Score
# ============================================================
print("\n================ THRESHOLD SEARCH (F1-optimized) ================")
optimal_thresholds = []

for c, class_name in enumerate(["Shoulder", "Gore Area", "Viable"]):
    best_thresh = 0.50
    best_f1 = 0.0
    for thresh in np.arange(0.05, 0.96, 0.01):
        predictions = (val_probs[:, c] >= thresh).astype(int)
        f1 = f1_score(val_targets[:, c], predictions, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    optimal_thresholds.append(best_thresh)
    print(f"{class_name}: Threshold = {best_thresh:.2f} | Validation F1 = {best_f1:.4f}")


# ============================================================
# 20. Test Set Predictions — real TTA
#
# This replaces the previous get_predictions_with_tta, which had a bug:
# it looped over tta_transforms but never actually applied a different
# transform to the input, so it ran the model on the same tensor three
# times and called that "averaging." This version reloads the raw image
# per test example and applies each of the three tta_transforms to it
# for real, then averages the resulting probabilities.
# Slower (per-image loop instead of batched DataLoader), but correct.
# Fine for a test set in the hundreds of images.
# ============================================================
def predict_with_tta(model, image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    tensors = [t(image) for t in tta_transforms]
    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(batch))

    return probs.mean(dim=0).cpu().numpy()


def get_test_predictions(model, image_paths, labels, use_tta=True):
    model.eval()
    all_probs = []
    for path in image_paths:
        if use_tta:
            probs = predict_with_tta(model, path)
        else:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            tensor = eval_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.sigmoid(model(tensor)).cpu().numpy()[0]
        all_probs.append(probs)
    return np.array(all_probs), labels, list(image_paths)


test_probs, all_labels, test_paths = get_test_predictions(
    model, X_test_paths, y_test, use_tta=True
)

all_preds = np.zeros_like(test_probs, dtype=int)
for c in range(num_classes):
    all_preds[:, c] = (test_probs[:, c] >= optimal_thresholds[c]).astype(int)


# ============================================================
# 21. Metrics Computation
# ============================================================
overall_accuracy = (all_preds == all_labels).mean()
shoulder_acc = (all_preds[:, 0] == all_labels[:, 0]).mean()
gore_acc = (all_preds[:, 1] == all_labels[:, 1]).mean()
viable_acc = (all_preds[:, 2] == all_labels[:, 2]).mean()

print("\n================ TEST RESULTS ================")
print(f"Overall Accuracy: {overall_accuracy:.4f}")
print(f"Shoulder Accuracy: {shoulder_acc:.4f}")
print(f"Gore Area Accuracy: {gore_acc:.4f}")
print(f"Viable Accuracy: {viable_acc:.4f}")


# ============================================================
# 22. Detailed Shoulder Metrics
# ============================================================
shoulder_true = all_labels[:, 0]
shoulder_pred = all_preds[:, 0]

print("\n================ SHOULDER METRICS ================")
print(f"Accuracy: {accuracy_score(shoulder_true, shoulder_pred):.4f}")
print(f"Precision: {precision_score(shoulder_true, shoulder_pred, zero_division=0):.4f}")
print(f"Recall: {recall_score(shoulder_true, shoulder_pred, zero_division=0):.4f}")
print(f"F1: {f1_score(shoulder_true, shoulder_pred, zero_division=0):.4f}")

print("\nShoulder Classification Report:")
print(classification_report(
    shoulder_true, shoulder_pred,
    target_names=["No Shoulder", "Shoulder"],
    zero_division=0
))

shoulder_cm = confusion_matrix(shoulder_true, shoulder_pred)
print("\nShoulder Confusion Matrix:")
print(shoulder_cm)
print(f"\nTN: {shoulder_cm[0, 0]}")
print(f"FP: {shoulder_cm[0, 1]}")
print(f"FN: {shoulder_cm[1, 0]}")
print(f"TP: {shoulder_cm[1, 1]}")


# ============================================================
# 23. Gore Metrics
# ============================================================
gore_true = all_labels[:, 1]
gore_pred = all_preds[:, 1]

print("\n================ GORE METRICS ================")
print(f"Accuracy: {accuracy_score(gore_true, gore_pred):.4f}")
print(f"Precision: {precision_score(gore_true, gore_pred, zero_division=0):.4f}")
print(f"Recall: {recall_score(gore_true, gore_pred, zero_division=0):.4f}")
print(f"F1: {f1_score(gore_true, gore_pred, zero_division=0):.4f}")

print("\nGore Classification Report:")
print(classification_report(
    gore_true, gore_pred,
    target_names=["No Gore", "Gore Area"],
    zero_division=0
))


# ============================================================
# 24. Viable Metrics
# ============================================================
viable_true = all_labels[:, 2]
viable_pred = all_preds[:, 2]

print("\n================ VIABLE METRICS ================")
print(f"Accuracy: {accuracy_score(viable_true, viable_pred):.4f}")
print(f"Precision: {precision_score(viable_true, viable_pred, zero_division=0):.4f}")
print(f"Recall: {recall_score(viable_true, viable_pred, zero_division=0):.4f}")
print(f"F1: {f1_score(viable_true, viable_pred, zero_division=0):.4f}")

print("\nViable Classification Report:")
print(classification_report(
    viable_true, viable_pred,
    target_names=["Not Viable", "Viable"],
    zero_division=0
))

viable_cm = confusion_matrix(viable_true, viable_pred)
print("\nViable Confusion Matrix:")
print(viable_cm)
print(f"\nTN: {viable_cm[0, 0]}")
print(f"FP: {viable_cm[0, 1]}")
print(f"FN: {viable_cm[1, 0]}")
print(f"TP: {viable_cm[1, 1]}")


# ============================================================
# 25. Grad-CAM
# ============================================================
def compute_gradcam(model, target_layer, img_tensor, target_idx=0):
    model.eval()
    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    img_input = img_tensor.unsqueeze(0).to(device)
    output = model(img_input)
    score = output[0, target_idx]
    probability = torch.sigmoid(score).item()

    model.zero_grad()
    score.backward()

    h1.remove()
    h2.remove()

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = F.relu(cam).squeeze().detach().cpu().numpy()
    # cv2.resize takes (width, height) — RESIZE_WH_CV2 keeps this in sync
    # with whatever RESIZE_HW is set to above, instead of hardcoding 224x224.
    cam = cv2.resize(cam, RESIZE_WH_CV2)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    return cam, probability


def show_gradcam_grid(indices, model, target_layer, dataset, true_labels, pred_labels, title, target_idx=0, max_samples=20):
    if len(indices) == 0:
        print(f"No samples to display for: {title}")
        return

    if len(indices) > max_samples:
        indices = np.random.choice(indices, max_samples, replace=False)

    cols = 5
    rows = math.ceil(len(indices) / cols)
    plt.figure(figsize=(15, 3 * rows))

    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)

    for i, idx in enumerate(indices):
        item = dataset[idx]
        img_tensor = item["images"]
        cam, probability = compute_gradcam(model, target_layer, img_tensor, target_idx=target_idx)

        img_display = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_display = img_display * std + mean
        img_display = np.clip(img_display, 0, 1)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = 0.5 * img_display + 0.5 * heatmap

        plt.subplot(rows, cols, i + 1)
        plt.imshow(overlay)
        plt.title(
            f"True: {int(true_labels[idx])} | Pred: {int(pred_labels[idx])} | Prob: {probability:.3f}",
            fontsize=9
        )
        plt.axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# ============================================================
# 26. Failure Analysis
# ============================================================
shoulder_failures = np.where(shoulder_true != shoulder_pred)[0]
shoulder_false_negatives = np.where((shoulder_true == 1) & (shoulder_pred == 0))[0]
shoulder_false_positives = np.where((shoulder_true == 0) & (shoulder_pred == 1))[0]
correct_shoulders = np.where((shoulder_true == 1) & (shoulder_pred == 1))[0]
correct_no_shoulders = np.where((shoulder_true == 0) & (shoulder_pred == 0))[0]

gore_failures = np.where(gore_true != gore_pred)[0]
correct_gore = np.where((gore_true == 1) & (gore_pred == 1))[0]

viable_failures = np.where(viable_true != viable_pred)[0]
correct_viable = np.where((viable_true == 1) & (viable_pred == 1))[0]

print("\n================ FAILURE ANALYSIS ================")
print(f"Total shoulder failures: {len(shoulder_failures)}")
print(f"Shoulder false negatives: {len(shoulder_false_negatives)}")
print(f"Shoulder false positives: {len(shoulder_false_positives)}")
print(f"Correct shoulder positives: {len(correct_shoulders)}")
print(f"Correct shoulder negatives: {len(correct_no_shoulders)}")
print(f"Total gore failures: {len(gore_failures)}")
print(f"Total viable failures: {len(viable_failures)}")


# ============================================================
# 27. Grad-CAM Visualizations
# ============================================================
target_conv_layer = model.efficientnet.features[-1]

if len(shoulder_failures) > 0:
    show_gradcam_grid(shoulder_failures, model, target_conv_layer, test_dataset,
                       shoulder_true, shoulder_pred, "Shoulder Grad-CAM - All Failures", target_idx=0)

if len(shoulder_false_negatives) > 0:
    show_gradcam_grid(shoulder_false_negatives, model, target_conv_layer, test_dataset,
                       shoulder_true, shoulder_pred, "Shoulder Grad-CAM - False Negatives", target_idx=0)

if len(shoulder_false_positives) > 0:
    show_gradcam_grid(shoulder_false_positives, model, target_conv_layer, test_dataset,
                       shoulder_true, shoulder_pred, "Shoulder Grad-CAM - False Positives", target_idx=0)

if len(correct_shoulders) > 0:
    show_gradcam_grid(correct_shoulders, model, target_conv_layer, test_dataset,
                       shoulder_true, shoulder_pred, "Shoulder Grad-CAM - Correct", target_idx=0)

if len(gore_failures) > 0:
    show_gradcam_grid(gore_failures, model, target_conv_layer, test_dataset,
                       gore_true, gore_pred, "Gore Grad-CAM - Failures", target_idx=1)

if len(correct_gore) > 0:
    show_gradcam_grid(correct_gore, model, target_conv_layer, test_dataset,
                       gore_true, gore_pred, "Gore Grad-CAM - Correct", target_idx=1)

if len(viable_failures) > 0:
    show_gradcam_grid(viable_failures, model, target_conv_layer, test_dataset,
                       viable_true, viable_pred, "Viable Grad-CAM - Failures", target_idx=2)

if len(correct_viable) > 0:
    show_gradcam_grid(correct_viable, model, target_conv_layer, test_dataset,
                       viable_true, viable_pred, "Viable Grad-CAM - Correct", target_idx=2)


# ============================================================
# 28. Baseline Comparison
# ============================================================
majority_baseline = max(shoulder_true.mean(), 1 - shoulder_true.mean())
print("\n================ BASELINE COMPARISON ================")
print(f"Shoulder majority-class baseline: {majority_baseline:.4f}")
print(f"Shoulder model accuracy: {shoulder_acc:.4f}")


# ============================================================
# 29. Final Summary
# ============================================================
print("\n================ FINAL SUMMARY ================")
print("Model: EfficientNet-B0")
print(f"Input resolution: {RESIZE_HW[0]}x{RESIZE_HW[1]} (height x width)")
print(f"Overall Accuracy: {overall_accuracy:.4f}")
print(f"Shoulder Accuracy: {shoulder_acc:.4f}")
print(f"Gore Accuracy: {gore_acc:.4f}")
print(f"Viable Accuracy: {viable_acc:.4f}")
print("\nOptimal thresholds (F1-optimized):")
print(f"Shoulder: {optimal_thresholds[0]:.2f}")
print(f"Gore: {optimal_thresholds[1]:.2f}")
print(f"Viable: {optimal_thresholds[2]:.2f}")
print(f"\nBest model saved at epoch: {best_epoch + 1}")
print(f"Final validation loss: {best_val_loss:.4f}")
print(f"Total training time: {total_training_time / 60:.1f} minutes")