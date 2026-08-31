import os
import math
import copy
import random
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
from torchvision.models import ResNet18_Weights
from sklearn.model_selection import train_test_split
from torchvision.models import ResNet50_Weights
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.metrics import confusion_matrix, classification_report

# Optional
# from pretrainer import mask_image
# ============================================================
# 0. Reproducibility
# ============================================================
SEED = 2005
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(SEED)
# ============================================================
# 1. Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
# ============================================================
# 2. Custom Transforms & Preprocessing
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        probs = torch.sigmoid(logits)

        pt = torch.where(
            targets == 1,
            probs,
            1 - probs
        )

        focal_weight = (1 - pt) ** self.gamma

        alpha_weight = torch.where(
            targets == 1,
            self.alpha,
            1 - self.alpha
        )

        loss = alpha_weight * focal_weight * bce

        return loss.mean()

class HighwayROICrop:
    """
    Crops out:
        - Top portion: sky, trees, overhead bridges
        - Bottom portion: UI artifacts, car hood, compass, etc.
    """
    def __init__(self, top_ratio=0.20, bottom_ratio=0.90):
        self.top_ratio = top_ratio
        self.bottom_ratio = bottom_ratio
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            h, w = img.shape[:2]
            top = int(h * self.top_ratio)
            bottom = int(h * self.bottom_ratio)
            return img[top:bottom, :, :]
        else:
            w, h = img.size
            top = int(h * self.top_ratio)
            bottom = int(h * self.bottom_ratio)
            return img.crop((0, top, w, bottom))
# ============================================================
# ImageNet Normalization
# ============================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# ============================================================
# Training Transform
# ============================================================
# Reduced augmentation compared to your original version.
#
# Your previous pipeline used:
# ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20)
# RandomAffine(degrees=5, translate=(0.04, 0.04))
#
# We are making this less aggressive because shoulder detection
# may depend strongly on geometry and appearance.
# ============================================================
train_transform = transforms.Compose([
    HighwayROICrop(
        top_ratio=0.20,
        bottom_ratio=0.90
    ),
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(
        p=0.5
    ),
    transforms.RandomAffine(
        degrees=3,
        translate=(0.02, 0.02)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])
# ============================================================
# Validation / Test Transform
# ============================================================
eval_transform = transforms.Compose([
    HighwayROICrop(
        top_ratio=0.20,
        bottom_ratio=0.90
    ),
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])
# ============================================================
# 3. Dataset
# ============================================================
class HighwayDataset(Dataset):
    def __init__(
        self,
        image_paths,
        labels,
        transform=None
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(
                f"Image not found at {img_path}"
            )
        # BGR -> RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )
        # Handle grayscale images
        if len(image.shape) == 2:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB
            )
        # Optional masking
        # image = mask_image(image)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(
            self.labels[idx],
            dtype=torch.float32
        )
        return {
            "images": image,
            "labels": label
        }
# ============================================================
# 4. Paths
# ============================================================
labels_path = (
    r"C:\Users\tasne\Desktop"
    r"\HighwayProject\classifdataset\mldata.csv"
)
images_dir = (
    r"C:\Users\tasne\Desktop"
    r"\HighwayProject\classifdataset\projectpics"
)
# ============================================================
# 5. Load CSV
# ============================================================
df = pd.read_csv(labels_path)
print("\nDataset size:", len(df))
print("\nCSV columns:")
print(df.columns.tolist())
# ============================================================
# Image Paths
# ============================================================
image_paths = [
    os.path.join(
        images_dir,
        str(name) + ".png"
    )
    for name in df["image"].tolist()
]
# ============================================================
# Labels
# ============================================================
labels = df.iloc[:, 1:].values.astype(
    np.float32
)
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
print(
    "Shoulder positives:",
    int(labels[:, 0].sum())
)
print(
    "Shoulder negatives:",
    int(len(labels) - labels[:, 0].sum())
)
print(
    "Gore positives:",
    int(labels[:, 1].sum())
)
print(
    "Gore negatives:",
    int(len(labels) - labels[:, 1].sum())
)

print(
    "Viable positives:",
    int(labels[:, 2].sum())
)

print(
    "Viable negatives:",
    int(len(labels) - labels[:, 2].sum())
)
# ============================================================
# 6. Train / Validation / Test Split
# ============================================================

NUM_FORCED_TRAIN = 20

# ------------------------------------------------------------
# Separate the newest 20 images from the older dataset
# ------------------------------------------------------------

old_image_paths = image_paths[:-NUM_FORCED_TRAIN]
old_labels = labels[:-NUM_FORCED_TRAIN]

new_image_paths = image_paths[-NUM_FORCED_TRAIN:]
new_labels = labels[-NUM_FORCED_TRAIN:]


print("\nForced-training images:")
for path in new_image_paths:
    print(os.path.basename(path))


# ------------------------------------------------------------
# Split ONLY the old images
#
# 33% of old data -> test
# Remaining old data -> train/validation
# ------------------------------------------------------------

X_train_val_paths, X_test_paths, y_train_val, y_test = train_test_split(
    old_image_paths,
    old_labels,
    test_size=0.33,
    random_state=SEED
)


X_train_paths, X_val_paths, y_train, y_val = train_test_split(
    X_train_val_paths,
    y_train_val,
    test_size=0.224,
    random_state=SEED
)


# ------------------------------------------------------------
# Force all 20 new images into TRAINING
# ------------------------------------------------------------

X_train_paths = list(X_train_paths) + list(new_image_paths)

y_train = np.concatenate(
    [
        y_train,
        new_labels
    ],
    axis=0
)


# ------------------------------------------------------------
# Verify
# ------------------------------------------------------------

print("\n================ DATA SPLIT ================")

print("Training images:", len(X_train_paths))
print("Validation images:", len(X_val_paths))
print("Test images:", len(X_test_paths))

print(
    "Forced new images in training:",
    len(new_image_paths)
)


# Make absolutely sure none leaked into val/test
new_set = set(new_image_paths)

assert new_set.isdisjoint(set(X_val_paths))
assert new_set.isdisjoint(set(X_test_paths))

print("Verified: all newest 20 images are TRAIN ONLY.")
# ============================================================
# 7. Dataset Objects
# ============================================================
train_dataset = HighwayDataset(
    X_train_paths,
    y_train,
    transform=train_transform
)
val_dataset = HighwayDataset(
    X_val_paths,
    y_val,
    transform=eval_transform
)
test_dataset = HighwayDataset(
    X_test_paths,
    y_test,
    transform=eval_transform
)
# ============================================================
# 8. DataLoaders
# ============================================================
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)
val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)
# ============================================================
# 9. Model
# ============================================================
class MLC(nn.Module):
    def __init__(self, num_classes=3):
        super(MLC, self).__init__()
        self.resnet = models.resnet18(
        weights=ResNet18_Weights.DEFAULT
        )
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(
                p=0.3
            ),
            nn.Linear(
                in_features,
                num_classes
            )
        )
    def forward(self, x):
        return self.resnet(x)
# ============================================================
# 10. Create Model
# ============================================================
num_classes = 3
model = MLC(
    num_classes=num_classes
).to(device)
print("\n================ MODEL ================")
print(model)
# ============================================================
# 11. Loss Function
# ============================================================
# IMPORTANT:
#
# Previously you were using pos_weight.
#
# For this experiment we are removing it because we want the
# model to learn the natural class distribution rather than
# being pushed toward positive predictions.
#
# Threshold optimization later will handle the final
# classification cutoff separately.
# ============================================================
criterion = nn.BCEWithLogitsLoss()
# ============================================================
# 12. Optimizer
# ============================================================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-2
)
# ============================================================
# 13. Learning Rate Scheduler
# ============================================================
num_epochs = 35
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs
)
# ============================================================
# 14. Training History
# ============================================================
train_losses = []
val_losses = []
best_val_loss = float("inf")
best_weights = None
# ============================================================
# 15. Training Loop
# ============================================================
print("\n================ TRAINING ================\n")
for epoch in range(num_epochs):
    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------
    model.train()
    running_train_loss = 0.0
    for batch in train_loader:
        data = batch["images"].to(device)
        targets = batch["labels"].to(device)
        # Clear gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(data)
        # Loss
        loss = criterion(
            outputs,
            targets
        )
        # Backpropagation
        loss.backward()
        # Update weights
        optimizer.step()
        running_train_loss += (
            loss.item()
            * data.size(0)
        )
    epoch_train_loss = (
        running_train_loss
        / len(train_loader.dataset)
    )
    train_losses.append(
        epoch_train_loss
    )
    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            data = batch["images"].to(device)
            targets = batch["labels"].to(device)
            outputs = model(data)
            loss = criterion(
                outputs,
                targets
            )
            running_val_loss += (
                loss.item()
                * data.size(0)
            )
    epoch_val_loss = (
        running_val_loss
        / len(val_loader.dataset)
    )
    val_losses.append(
        epoch_val_loss
    )
    # --------------------------------------------------------
    # Save BEST model
    # --------------------------------------------------------
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        # IMPORTANT:
        # deepcopy prevents later training from modifying
        # the saved best model weights.
        best_weights = copy.deepcopy(
            model.state_dict()
        )
    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------
    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]
    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------
    print(
        f"Epoch [{epoch + 1:02d}/{num_epochs:02d}] "
        f"- Train Loss: {epoch_train_loss:.4f} "
        f"- Val Loss: {epoch_val_loss:.4f} "
        f"- LR: {current_lr:.7f}"
    )
# ============================================================
# 16. Load Best Model
# ============================================================
if best_weights is not None:
    model.load_state_dict(
        best_weights
    )
print("\nBest validation loss:", best_val_loss)

# ============================================================
# 16b. Save Model Checkpoint
# ============================================================
torch.save(model.state_dict(), r"C:\Users\tasne\Desktop\HighwayProject\cv_model.pth")
print("Saved model checkpoint to cv_model.pth")

# ============================================================
# 17. Plot Loss Curves
# ============================================================
plt.figure(
    figsize=(8, 4)
)
plt.plot(
    train_losses,
    label="Train Loss"
)
plt.plot(
    val_losses,
    label="Validation Loss"
)
plt.title(
    "Training & Validation Loss"
)
plt.xlabel(
    "Epoch"
)
plt.ylabel(
    "Loss"
)
plt.legend()
plt.grid(True)
plt.show()
# ============================================================
# 18. Prediction Function
# ============================================================
def get_predictions_and_targets(
    model,
    loader
):
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            data = batch["images"].to(
                device
            )
            targets = batch["labels"].to(
                device
            )
            logits = model(data)
            probs = torch.sigmoid(
                logits
            )
            all_probs.append(
                probs.cpu().numpy()
            )
            all_targets.append(
                targets.cpu().numpy()
            )
    return (
        np.concatenate(
            all_probs,
            axis=0
        ),
        np.concatenate(
            all_targets,
            axis=0
        )
    )
# ============================================================
# 19. Validation Predictions
# ============================================================
val_probs, val_targets = (
    get_predictions_and_targets(
        model,
        val_loader
    )
)
# ============================================================
# 20. Optimize Threshold For ACCURACY
# ============================================================
print(
    "\n================ THRESHOLD SEARCH ================"
)
optimal_thresholds = []
for c, class_name in enumerate(
    ["Shoulder", "Gore Area", "Viable"]
):
    best_thresh = 0.50
    best_acc = 0.0
    for thresh in np.arange(
        0.05,
        0.96,
        0.01
    ):
        predictions = (
            val_probs[:, c]
            >= thresh
        ).astype(int)
        acc = (
            predictions
            == val_targets[:, c]
        ).mean()
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    optimal_thresholds.append(
        best_thresh
    )
    print(
        f"{class_name}: "
        f"Threshold = {best_thresh:.2f} | "
        f"Validation Accuracy = {best_acc:.4f}"
    )
# ============================================================
# 21. Test Set Predictions
# ============================================================
test_probs, all_labels = (
    get_predictions_and_targets(
        model,
        test_loader
    )
)
# ============================================================
# Apply Class-Specific Thresholds
# ============================================================
all_preds = np.zeros_like(
    test_probs,
    dtype=int
)
for c in range(num_classes):
    all_preds[:, c] = (
        test_probs[:, c]
        >= optimal_thresholds[c]
    ).astype(int)
# ============================================================
# 22. Accuracy
# ============================================================
overall_accuracy = (
    all_preds
    == all_labels
).mean()
shoulder_acc = (
    all_preds[:, 0]
    == all_labels[:, 0]
).mean()
gore_acc = (
    all_preds[:, 1]
    == all_labels[:, 1]
).mean()

viable_acc = (
    all_preds[:, 2]
    == all_labels[:, 2]
).mean()
print(
    "\n================ TEST RESULTS ================"
)
print(
    f"Overall Accuracy: {overall_accuracy:.4f}"
)
print(
    f"Shoulder Accuracy: {shoulder_acc:.4f}"
)
print(
    f"Gore Area Accuracy: {gore_acc:.4f}"
)

print(
    f"Viable Accuracy: {viable_acc:.4f}"
)
# ============================================================
# 23. Detailed Shoulder Metrics
# ============================================================
shoulder_true = all_labels[:, 0]
shoulder_pred = all_preds[:, 0]
print(
    "\n================ SHOULDER METRICS ================"
)
print(
    "Accuracy:",
    accuracy_score(
        shoulder_true,
        shoulder_pred
    )
)
print(
    "Precision:",
    precision_score(
        shoulder_true,
        shoulder_pred,
        zero_division=0
    )
)
print(
    "Recall:",
    recall_score(
        shoulder_true,
        shoulder_pred,
        zero_division=0
    )
)
print(
    "F1:",
    f1_score(
        shoulder_true,
        shoulder_pred,
        zero_division=0
    )
)
print(
    "\nShoulder Classification Report:"
)
print(
    classification_report(
        shoulder_true,
        shoulder_pred,
        target_names=[
            "No Shoulder",
            "Shoulder"
        ],
        zero_division=0
    )
)
# ============================================================
# 24. Shoulder Confusion Matrix
# ============================================================
shoulder_cm = confusion_matrix(
    shoulder_true,
    shoulder_pred
)
print(
    "\nShoulder Confusion Matrix:"
)
print(shoulder_cm)
print(
    "\nTN:",
    shoulder_cm[0, 0]
)
print(
    "FP:",
    shoulder_cm[0, 1]
)
print(
    "FN:",
    shoulder_cm[1, 0]
)
print(
    "TP:",
    shoulder_cm[1, 1]
)
# ============================================================
# 25. Gore Metrics
# ============================================================
gore_true = all_labels[:, 1]
gore_pred = all_preds[:, 1]
print(
    "\n================ GORE METRICS ================"
)
print(
    "Accuracy:",
    accuracy_score(
        gore_true,
        gore_pred
    )
)
print(
    "Precision:",
    precision_score(
        gore_true,
        gore_pred,
        zero_division=0
    )
)
print(
    "Recall:",
    recall_score(
        gore_true,
        gore_pred,
        zero_division=0
    )
)
print(
    "F1:",
    f1_score(
        gore_true,
        gore_pred,
        zero_division=0
    )
)
print(
    "\nGore Classification Report:"
)
print(
    classification_report(
        gore_true,
        gore_pred,
        target_names=[
            "No Gore",
            "Gore Area"
        ],
        zero_division=0
    )
)
# ============================================================
# 26. Viable Metrics
# ============================================================
viable_true = all_labels[:, 2]
viable_pred = all_preds[:, 2]

print(
    "\n================ VIABLE METRICS ================"
)

print(
    "Accuracy:",
    accuracy_score(
        viable_true,
        viable_pred
    )
)

print(
    "Precision:",
    precision_score(
        viable_true,
        viable_pred,
        zero_division=0
    )
)

print(
    "Recall:",
    recall_score(
        viable_true,
        viable_pred,
        zero_division=0
    )
)

print(
    "F1:",
    f1_score(
        viable_true,
        viable_pred,
        zero_division=0
    )
)

print(
    "\nViable Classification Report:"
)

print(
    classification_report(
        viable_true,
        viable_pred,
        target_names=[
            "Not Viable",
            "Viable"
        ],
        zero_division=0
    )
)

viable_cm = confusion_matrix(
    viable_true,
    viable_pred
)

print(
    "\nViable Confusion Matrix:"
)
print(viable_cm)

print("\nTN:", viable_cm[0, 0])
print("FP:", viable_cm[0, 1])
print("FN:", viable_cm[1, 0])
print("TP:", viable_cm[1, 1])


# ============================================================
# 27. Grad-CAM
# ============================================================
def compute_gradcam(
    model,
    target_layer,
    img_tensor,
    target_idx=0
):
    model.eval()
    activations = None
    gradients = None
    # --------------------------------------------------------
    # Forward Hook
    # --------------------------------------------------------
    def forward_hook(
        module,
        input,
        output
    ):
        nonlocal activations
        activations = output
    # --------------------------------------------------------
    # Backward Hook
    # --------------------------------------------------------
    def backward_hook(
        module,
        grad_input,
        grad_output
    ):
        nonlocal gradients
        gradients = grad_output[0]
    # Register hooks
    h1 = target_layer.register_forward_hook(
        forward_hook
    )
    h2 = target_layer.register_full_backward_hook(
        backward_hook
    )
    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    img_input = (
        img_tensor
        .unsqueeze(0)
        .to(device)
    )
    output = model(
        img_input
    )
    score = output[
        0,
        target_idx
    ]
    probability = torch.sigmoid(
    score
    ).item()
    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------
    model.zero_grad()
    score.backward()
    # Remove hooks
    h1.remove()
    h2.remove()
    # --------------------------------------------------------
    # Compute Grad-CAM
    # --------------------------------------------------------
    weights = gradients.mean(
        dim=(2, 3),
        keepdim=True
    )
    cam = (
        weights
        * activations
    ).sum(
        dim=1,
        keepdim=True
    )
    cam = F.relu(
        cam
    ).squeeze().detach().cpu().numpy()
    cam = cv2.resize(
        cam,
        (224, 224)
    )
    cam = (
        cam - cam.min()
    ) / (
        cam.max() - cam.min() + 1e-8
    )
    return cam, probability
# ============================================================
# 27. Grad-CAM Grid
# ============================================================
def show_gradcam_grid(
    indices,
    model,
    target_layer,
    dataset,
    true_labels,
    pred_labels,
    title,
    target_idx=0
):
    if len(indices) == 0:
        print(
            f"No samples to display for: {title}"
        )
        return
    cols = 5
    rows = math.ceil(
        len(indices) / cols
    )
    plt.figure(
        figsize=(15, 3 * rows)
    )
    mean = np.array(
        IMAGENET_MEAN
    )
    std = np.array(
        IMAGENET_STD
    )
    for i, idx in enumerate(indices):
        item = dataset[idx]
        img_tensor = item["images"]
        cam, probability = compute_gradcam(
            model,
            target_layer,
            img_tensor,
            target_idx=target_idx
        )
        # ----------------------------------------------------
        # Unnormalize image
        # ----------------------------------------------------
        img_display = (
            img_tensor
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        img_display = (
            img_display * std
            + mean
        )
        img_display = np.clip(
            img_display,
            0,
            1
        )
        # ----------------------------------------------------
        # Heatmap
        # ----------------------------------------------------
        heatmap = cv2.applyColorMap(
            np.uint8(
                255 * cam
            ),
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB
        ) / 255.0
        overlay = (
            0.5 * img_display
            + 0.5 * heatmap
        )
        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------
        plt.subplot(
            rows,
            cols,
            i + 1
        )
        plt.imshow(
            overlay
        )
        plt.title(
            f"True: {int(true_labels[idx])} | "
            f"Pred: {int(pred_labels[idx])} | "
            f"Prob: {probability:.3f}",
            fontsize=9
        )
        plt.axis(
            "off"
        )
    plt.suptitle(
        title,
        fontsize=14
    )
    plt.tight_layout()
    plt.show()
# ============================================================
# 28. Failure Analysis
# ============================================================
# Shoulder failures
shoulder_failures = np.where(
    shoulder_true
    != shoulder_pred
)[0]
# Shoulder false negatives
shoulder_false_negatives = np.where(
    (shoulder_true == 1)
    & (shoulder_pred == 0)
)[0]
# Shoulder false positives
shoulder_false_positives = np.where(
    (shoulder_true == 0)
    & (shoulder_pred == 1)
)[0]
# Correct shoulders
correct_shoulders = np.where(
    (shoulder_true == 1)
    & (shoulder_pred == 1)
)[0]
# Correct negatives
correct_no_shoulders = np.where(
    (shoulder_true == 0)
    & (shoulder_pred == 0)
)[0]
# Gore failures
gore_failures = np.where(
    gore_true
    != gore_pred
)[0]
# Correct gore
correct_gore = np.where(
    (gore_true == 1)
    & (gore_pred == 1)
)[0]

# Viable failures
viable_failures = np.where(
    viable_true
    != viable_pred
)[0]

# Correct viable positives
correct_viable = np.where(
    (viable_true == 1)
    & (viable_pred == 1)
)[0]
print(
    "\n================ FAILURE ANALYSIS ================"
)
print(
    "Total shoulder failures:",
    len(shoulder_failures)
)
print(
    "Shoulder false negatives:",
    len(shoulder_false_negatives)
)
print(
    "Shoulder false positives:",
    len(shoulder_false_positives)
)
print(
    "Correct shoulder positives:",
    len(correct_shoulders)
)
print(
    "Correct shoulder negatives:",
    len(correct_no_shoulders)
)
print(
    "Total gore failures:",
    len(gore_failures)
)

print(
    "Total viable failures:",
    len(viable_failures)
)
# ============================================================
# 29. Grad-CAM Layer
# ============================================================
target_conv_layer = (
    model.resnet.layer4[-1]
)
# ============================================================
# 30. Shoulder Failure Grad-CAM
# ============================================================
show_gradcam_grid(
    shoulder_failures,
    model,
    target_conv_layer,
    test_dataset,
    shoulder_true,
    shoulder_pred,
    "Shoulder Grad-CAM - All Failures",
    target_idx=0
)
# ============================================================
# 31. Shoulder False Negative Grad-CAM
# ============================================================
show_gradcam_grid(
    shoulder_false_negatives,
    model,
    target_conv_layer,
    test_dataset,
    shoulder_true,
    shoulder_pred,
    "Shoulder Grad-CAM - False Negatives",
    target_idx=0
)
# ============================================================
# 32. Shoulder False Positive Grad-CAM
# ============================================================
show_gradcam_grid(
    shoulder_false_positives,
    model,
    target_conv_layer,
    test_dataset,
    shoulder_true,
    shoulder_pred,
    "Shoulder Grad-CAM - False Positives",
    target_idx=0
)
# ============================================================
# 33. Correct Shoulder Grad-CAM
# ============================================================
show_gradcam_grid(
    correct_shoulders,
    model,
    target_conv_layer,
    test_dataset,
    shoulder_true,
    shoulder_pred,
    "Shoulder Grad-CAM - Correct",
    target_idx=0
)
# ============================================================
# 34. Gore Failure Grad-CAM
# ============================================================
show_gradcam_grid(
    gore_failures,
    model,
    target_conv_layer,
    test_dataset,
    gore_true,
    gore_pred,
    "Gore Grad-CAM - Failures",
    target_idx=1
)
# ============================================================
# 35. Gore Correct Grad-CAM
# ============================================================
show_gradcam_grid(
    correct_gore,
    model,
    target_conv_layer,
    test_dataset,
    gore_true,
    gore_pred,
    "Gore Grad-CAM - Correct",
    target_idx=1
)

# ============================================================
# Viable Failure Grad-CAM
# ============================================================
show_gradcam_grid(
    viable_failures,
    model,
    target_conv_layer,
    test_dataset,
    viable_true,
    viable_pred,
    "Viable Grad-CAM - Failures",
    target_idx=2
)

# ============================================================
# Viable Correct Grad-CAM
# ============================================================
show_gradcam_grid(
    correct_viable,
    model,
    target_conv_layer,
    test_dataset,
    viable_true,
    viable_pred,
    "Viable Grad-CAM - Correct",
    target_idx=2
)
# ============================================================
# 36. Majority Baseline
# ============================================================
majority_baseline = max(
    shoulder_true.mean(),
    1 - shoulder_true.mean()
)
print(
    "\n================ BASELINE COMPARISON ================"
)
print(
    f"Shoulder majority-class baseline: "
    f"{majority_baseline:.4f}"
)
print(
    f"Shoulder model accuracy: "
    f"{shoulder_acc:.4f}"
)
# ============================================================
# 37. Final Summary
# ============================================================
print(
    "\n================ FINAL SUMMARY ================"
)
print(
    f"Overall Accuracy: {overall_accuracy:.4f}"
)
print(
    f"Shoulder Accuracy: {shoulder_acc:.4f}"
)
print(
    f"Gore Accuracy: {gore_acc:.4f}"
)

print(
    f"Viable Accuracy: {viable_acc:.4f}"
)
print(
    "\nOptimal thresholds:"
)
print(
    f"Shoulder: {optimal_thresholds[0]:.2f}"
)
print(
    f"Gore: {optimal_thresholds[1]:.2f}"
)

print(
    f"Viable: {optimal_thresholds[2]:.2f}"
)



print("\nThreshold | Accuracy | Precision | Recall | F1")
print("-" * 60)

for thresh in np.arange(0.20, 0.81, 0.02):

    preds = (
        test_probs[:, 0] >= thresh
    ).astype(int)

    acc = accuracy_score(
        shoulder_true,
        preds
    )

    precision = precision_score(
        shoulder_true,
        preds,
        zero_division=0
    )

    recall = recall_score(
        shoulder_true,
        preds,
        zero_division=0
    )

    f1 = f1_score(
        shoulder_true,
        preds,
        zero_division=0
    )

    print(
        f"{thresh:.2f}      "
        f"{acc:.4f}      "
        f"{precision:.4f}      "
        f"{recall:.4f}      "
        f"{f1:.4f}"
    )


print("\nFALSE NEGATIVE SHOULDER PROBABILITIES")
print("--------------------------------------")

for idx in shoulder_false_negatives:

    prob = test_probs[idx, 0]

    print(
        f"Index: {idx:3d} | "
        f"Probability: {prob:.4f} | "
        f"Threshold: {optimal_thresholds[0]:.2f}"
    )

shoulder_probs = test_probs[:, 0]

positive_probs = shoulder_probs[shoulder_true == 1]
negative_probs = shoulder_probs[shoulder_true == 0]

print("SHOULDER POSITIVE IMAGES")
print("Mean:", positive_probs.mean())
print("Median:", np.median(positive_probs))
print("Min:", positive_probs.min())
print("Max:", positive_probs.max())

print("\nNO-SHOULDER IMAGES")
print("Mean:", negative_probs.mean())
print("Median:", np.median(negative_probs))
print("Min:", negative_probs.min())
print("Max:", negative_probs.max())

plt.figure(figsize=(8, 5))

plt.hist(
    positive_probs,
    bins=15,
    alpha=0.6,
    label="Shoulder"
)

plt.hist(
    negative_probs,
    bins=15,
    alpha=0.6,
    label="No Shoulder"
)

plt.axvline(
    optimal_thresholds[0],
    linestyle="--",
    label=f"Threshold = {optimal_thresholds[0]:.2f}"
)

plt.xlabel("Predicted Shoulder Probability")
plt.ylabel("Number of Images")
plt.title("Shoulder Probability Distribution")
plt.legend()
plt.show()




# ============================================================
# Shoulder Error Rate: Gore vs No Gore
# ============================================================

# Masks
gore_mask = gore_true == 1
no_gore_mask = gore_true == 0

# Where shoulder prediction was wrong
shoulder_error = shoulder_pred != shoulder_true

# Error rates
error_rate_gore = shoulder_error[gore_mask].mean()
error_rate_no_gore = shoulder_error[no_gore_mask].mean()

# Counts
failures_with_gore = np.sum(shoulder_error & gore_mask)
failures_without_gore = np.sum(shoulder_error & no_gore_mask)

total_with_gore = np.sum(gore_mask)
total_without_gore = np.sum(no_gore_mask)

print("\n================ SHOULDER ERRORS BY GORE PRESENCE ================")

print(f"Shoulder error rate WITH gore:    {error_rate_gore:.4f}")
print(f"Shoulder error rate WITHOUT gore: {error_rate_no_gore:.4f}")

print()

print(
    f"Shoulder failures WITH gore: "
    f"{failures_with_gore} / {total_with_gore}"
)

print(
    f"Shoulder failures WITHOUT gore: "
    f"{failures_without_gore} / {total_without_gore}"
)

print()

print(
    f"Percent of shoulder failures containing gore: "
    f"{failures_with_gore / np.sum(shoulder_error) * 100:.2f}%"
)


# Gore images only
gore_shoulder_positive = (
    (gore_true == 1) &
    (shoulder_true == 1)
)

gore_shoulder_negative = (
    (gore_true == 1) &
    (shoulder_true == 0)
)

# Error rates
error_gore_shoulder = shoulder_error[gore_shoulder_positive].mean()
error_gore_no_shoulder = shoulder_error[gore_shoulder_negative].mean()

print(
    "Gore + Shoulder error rate:",
    error_gore_shoulder
)

print(
    "Gore + No Shoulder error rate:",
    error_gore_no_shoulder
)

print()
print(
    "Gore + Shoulder:",
    np.sum(shoulder_error & gore_shoulder_positive),
    "/",
    np.sum(gore_shoulder_positive)
)

print(
    "Gore + No Shoulder:",
    np.sum(shoulder_error & gore_shoulder_negative),
    "/",
    np.sum(gore_shoulder_negative)
)


print("\n================ VIABLE METRICS ================")

print(
    classification_report(
        viable_true,
        viable_pred,
        target_names=["Not Viable", "Viable"]
    )
)

viable_cm = confusion_matrix(
    viable_true,
    viable_pred
)

print("\nViable Confusion Matrix:")
print(viable_cm)

tn, fp, fn, tp = viable_cm.ravel()

print("\nTN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)