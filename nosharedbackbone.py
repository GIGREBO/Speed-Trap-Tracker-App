import os
import math
import copy
import random

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# 0. Reproducibility
# ============================================================

SEED = 7


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


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

        w, h = img.size
        top = int(h * self.top_ratio)
        bottom = int(h * self.bottom_ratio)
        return img.crop((0, top, w, bottom))


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    HighwayROICrop(top_ratio=0.20, bottom_ratio=0.90),
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=3, translate=(0.02, 0.02)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = transforms.Compose([
    HighwayROICrop(top_ratio=0.20, bottom_ratio=0.90),
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ============================================================
# 3. Dataset
# ============================================================

class HighwayDataset(Dataset):
    """
    Keeps BOTH labels in each sample.

    The Shoulder model trains only against labels[:, 0].
    The Gore model trains only against labels[:, 1].

    This lets both independent models use the exact same image split.
    """

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = list(image_paths)
        self.labels = np.asarray(labels, dtype=np.float32)
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

        # Optional UI masking:
        # image = mask_image(image)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return {
            "images": image,
            "labels": label,
            "path": img_path,
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
print("CSV columns:", df.columns.tolist())

image_paths = [
    os.path.join(images_dir, str(name) + ".png")
    for name in df["image"].tolist()
]

labels = df.iloc[:, 1:].values.astype(np.float32)

if labels.shape[1] < 2:
    raise ValueError(
        "Expected at least two label columns after 'image': "
        "column 0 = Shoulder, column 1 = Gore Area."
    )

print("Label shape:", labels.shape)

print("\n================ DATASET LABEL COUNTS ================")
print("Shoulder positives:", int(labels[:, 0].sum()))
print("Shoulder negatives:", int(len(labels) - labels[:, 0].sum()))
print("Gore positives:", int(labels[:, 1].sum()))
print("Gore negatives:", int(len(labels) - labels[:, 1].sum()))


# ============================================================
# 6. Train / Validation / Test Split
# ============================================================
# The last 20 rows are forced into training, exactly like your
# current experiment. Both models use the SAME split.
# ============================================================

NUM_FORCED_TRAIN = 20

if NUM_FORCED_TRAIN > 0:
    old_image_paths = image_paths[:-NUM_FORCED_TRAIN]
    old_labels = labels[:-NUM_FORCED_TRAIN]

    forced_image_paths = image_paths[-NUM_FORCED_TRAIN:]
    forced_labels = labels[-NUM_FORCED_TRAIN:]
else:
    old_image_paths = image_paths
    old_labels = labels
    forced_image_paths = []
    forced_labels = np.empty((0, labels.shape[1]), dtype=np.float32)

print("\nForced-training images:")
for path in forced_image_paths:
    print(os.path.basename(path))

X_train_val_paths, X_test_paths, y_train_val, y_test = train_test_split(
    old_image_paths,
    old_labels,
    test_size=0.33,
    random_state=SEED,
)

X_train_paths, X_val_paths, y_train, y_val = train_test_split(
    X_train_val_paths,
    y_train_val,
    test_size=0.224,
    random_state=SEED,
)

X_train_paths = list(X_train_paths) + list(forced_image_paths)
y_train = np.concatenate([y_train, forced_labels], axis=0)

print("\n================ DATA SPLIT ================")
print("Training images:", len(X_train_paths))
print("Validation images:", len(X_val_paths))
print("Test images:", len(X_test_paths))
print("Forced new images in training:", len(forced_image_paths))

if NUM_FORCED_TRAIN > 0:
    forced_set = set(forced_image_paths)
    assert forced_set.isdisjoint(set(X_val_paths))
    assert forced_set.isdisjoint(set(X_test_paths))
    print("Verified: all newest 20 images are TRAIN ONLY.")


# ============================================================
# 7. Dataset Objects / DataLoaders
# ============================================================

train_dataset = HighwayDataset(X_train_paths, y_train, transform=train_transform)
val_dataset = HighwayDataset(X_val_paths, y_val, transform=eval_transform)
test_dataset = HighwayDataset(X_test_paths, y_test, transform=eval_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# 8. Independent Binary Model
# ============================================================

class BinaryResNet18(nn.Module):
    """
    ONE ResNet18 -> ONE binary output.

    We instantiate this class twice:
        1. shoulder_model
        2. gore_model

    There is NO shared backbone and NO shared classifier head.
    """

    def __init__(self):
        super().__init__()

        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.resnet.fc.in_features

        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        # [batch, 1] -> [batch]
        return self.resnet(x).squeeze(1)


# ============================================================
# 9. Train ONE Independent Model
# ============================================================

NUM_EPOCHS = 35


def train_binary_model(class_name, target_idx, model_seed):
    """
    Trains a completely independent ResNet18 for exactly one target.

    target_idx:
        0 -> Shoulder
        1 -> Gore Area
    """

    print("\n" + "=" * 70)
    print(f"TRAINING INDEPENDENT {class_name.upper()} MODEL")
    print("=" * 70)

    # Reset randomness so each model is independently reproducible.
    set_seed(model_seed)

    model = BinaryResNet18().to(device)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-2,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
    )

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = None

    for epoch in range(NUM_EPOCHS):
        # ------------------------- TRAIN -------------------------
        model.train()
        running_train_loss = 0.0

        for batch in train_loader:
            data = batch["images"].to(device)
            targets = batch["labels"][:, target_idx].to(device)

            optimizer.zero_grad()

            logits = model(data)
            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * data.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # ----------------------- VALIDATION ----------------------
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                data = batch["images"].to(device)
                targets = batch["labels"][:, target_idx].to(device)

                logits = model(data)
                loss = criterion(logits, targets)

                running_val_loss += loss.item() * data.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            best_weights = copy.deepcopy(model.state_dict())

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"{class_name} | "
            f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"LR: {current_lr:.7f}"
        )

    if best_weights is not None:
        model.load_state_dict(best_weights)

    print(
        f"\n{class_name} best validation loss: "
        f"{best_val_loss:.4f} at epoch {best_epoch}"
    )

    # Separate plot for each model.
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(f"{class_name} - Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model


# ============================================================
# 10. Train Shoulder and Gore SEPARATELY
# ============================================================

shoulder_model = train_binary_model(
    class_name="Shoulder",
    target_idx=0,
    model_seed=SEED,
)

gore_model = train_binary_model(
    class_name="Gore Area",
    target_idx=1,
    model_seed=SEED + 1,
)


# ============================================================
# 11. Prediction Function for ONE Binary Model
# ============================================================


def get_binary_predictions_and_targets(model, loader, target_idx):
    model.eval()

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            data = batch["images"].to(device)
            targets = batch["labels"][:, target_idx]

            logits = model(data)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    return (
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_targets, axis=0),
    )


# ============================================================
# 12. Threshold Search
# ============================================================


def find_best_accuracy_threshold(model, loader, target_idx, class_name):
    probs, targets = get_binary_predictions_and_targets(
        model,
        loader,
        target_idx,
    )

    best_thresh = 0.50
    best_acc = 0.0

    for thresh in np.arange(0.05, 0.96, 0.01):
        preds = (probs >= thresh).astype(int)
        acc = (preds == targets).mean()

        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    print(
        f"{class_name}: Threshold = {best_thresh:.2f} | "
        f"Validation Accuracy = {best_acc:.4f}"
    )

    return best_thresh


print("\n================ THRESHOLD SEARCH ================")

shoulder_threshold = find_best_accuracy_threshold(
    shoulder_model,
    val_loader,
    target_idx=0,
    class_name="Shoulder",
)

gore_threshold = find_best_accuracy_threshold(
    gore_model,
    val_loader,
    target_idx=1,
    class_name="Gore Area",
)


# ============================================================
# 13. Test Predictions
# ============================================================

shoulder_probs, shoulder_true = get_binary_predictions_and_targets(
    shoulder_model,
    test_loader,
    target_idx=0,
)

gore_probs, gore_true = get_binary_predictions_and_targets(
    gore_model,
    test_loader,
    target_idx=1,
)

shoulder_pred = (shoulder_probs >= shoulder_threshold).astype(int)
gore_pred = (gore_probs >= gore_threshold).astype(int)

shoulder_acc = accuracy_score(shoulder_true, shoulder_pred)
gore_acc = accuracy_score(gore_true, gore_pred)

# Same meaning as your old "overall accuracy": average accuracy over
# both binary labels, not exact-match accuracy.
overall_label_accuracy = np.mean([
    shoulder_acc,
    gore_acc,
])

# Optional stricter metric: BOTH labels on an image must be correct.
exact_match_accuracy = np.mean(
    (shoulder_pred == shoulder_true) &
    (gore_pred == gore_true)
)

print("\n================ TEST RESULTS ================")
print(f"Overall Label Accuracy: {overall_label_accuracy:.4f}")
print(f"Exact-Match Accuracy:   {exact_match_accuracy:.4f}")
print(f"Shoulder Accuracy:      {shoulder_acc:.4f}")
print(f"Gore Area Accuracy:     {gore_acc:.4f}")


# ============================================================
# 14. Generic Binary Evaluation
# ============================================================


def print_binary_metrics(class_name, true, pred):
    print(f"\n================ {class_name.upper()} METRICS ================")

    print("Accuracy:", accuracy_score(true, pred))
    print("Precision:", precision_score(true, pred, zero_division=0))
    print("Recall:", recall_score(true, pred, zero_division=0))
    print("F1:", f1_score(true, pred, zero_division=0))

    print(f"\n{class_name} Classification Report:")
    print(
        classification_report(
            true,
            pred,
            target_names=[f"No {class_name}", class_name],
            zero_division=0,
        )
    )

    cm = confusion_matrix(true, pred, labels=[0, 1])
    print(f"{class_name} Confusion Matrix:")
    print(cm)

    print("TN:", cm[0, 0])
    print("FP:", cm[0, 1])
    print("FN:", cm[1, 0])
    print("TP:", cm[1, 1])

    return cm


shoulder_cm = print_binary_metrics(
    "Shoulder",
    shoulder_true,
    shoulder_pred,
)

gore_cm = print_binary_metrics(
    "Gore Area",
    gore_true,
    gore_pred,
)


# ============================================================
# 15. Failure Analysis
# ============================================================

shoulder_failures = np.where(shoulder_true != shoulder_pred)[0]
shoulder_false_negatives = np.where(
    (shoulder_true == 1) & (shoulder_pred == 0)
)[0]
shoulder_false_positives = np.where(
    (shoulder_true == 0) & (shoulder_pred == 1)
)[0]
correct_shoulders = np.where(
    (shoulder_true == 1) & (shoulder_pred == 1)
)[0]
correct_no_shoulders = np.where(
    (shoulder_true == 0) & (shoulder_pred == 0)
)[0]

gore_failures = np.where(gore_true != gore_pred)[0]
correct_gore = np.where((gore_true == 1) & (gore_pred == 1))[0]

print("\n================ FAILURE ANALYSIS ================")
print("Total shoulder failures:", len(shoulder_failures))
print("Shoulder false negatives:", len(shoulder_false_negatives))
print("Shoulder false positives:", len(shoulder_false_positives))
print("Correct shoulder positives:", len(correct_shoulders))
print("Correct shoulder negatives:", len(correct_no_shoulders))
print("Total gore failures:", len(gore_failures))


# ============================================================
# 16. Grad-CAM for a SINGLE-output model
# ============================================================


def compute_gradcam(model, target_layer, img_tensor):
    model.eval()

    activations = None
    gradients = None

    def forward_hook(module, inputs, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    img_input = img_tensor.unsqueeze(0).to(device)

    score = model(img_input)[0]
    probability = torch.sigmoid(score).item()

    model.zero_grad()
    score.backward()

    h1.remove()
    h2.remove()

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = F.relu(cam).squeeze().detach().cpu().numpy()

    # Use the actual input tensor size rather than hard-coding 224.
    height = int(img_tensor.shape[-2])
    width = int(img_tensor.shape[-1])
    cam = cv2.resize(cam, (width, height))

    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    return cam, probability


# ============================================================
# 17. Grad-CAM Grid
# ============================================================


def show_gradcam_grid(
    indices,
    model,
    target_layer,
    dataset,
    true_labels,
    pred_labels,
    title,
):
    if len(indices) == 0:
        print(f"No samples to display for: {title}")
        return

    cols = 5
    rows = math.ceil(len(indices) / cols)

    plt.figure(figsize=(15, 3 * rows))

    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)

    for i, idx in enumerate(indices):
        item = dataset[idx]
        img_tensor = item["images"]

        cam, probability = compute_gradcam(
            model,
            target_layer,
            img_tensor,
        )

        img_display = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_display = img_display * std + mean
        img_display = np.clip(img_display, 0, 1)

        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam),
            cv2.COLORMAP_JET,
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

        overlay = 0.5 * img_display + 0.5 * heatmap

        plt.subplot(rows, cols, i + 1)
        plt.imshow(overlay)
        plt.title(
            f"True: {int(true_labels[idx])} | "
            f"Pred: {int(pred_labels[idx])} | "
            f"Prob: {probability:.3f}",
            fontsize=9,
        )
        plt.axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# ============================================================
# 18. Grad-CAM Plots - EACH MODEL USES ITS OWN BACKBONE
# ============================================================

shoulder_target_layer = shoulder_model.resnet.layer4[-1]
gore_target_layer = gore_model.resnet.layer4[-1]

show_gradcam_grid(
    shoulder_failures,
    shoulder_model,
    shoulder_target_layer,
    test_dataset,
    shoulder_true,
    shoulder_pred,
    "Shoulder Grad-CAM - All Failures",
)

show_gradcam_grid(
    shoulder_false_negatives,
    shoulder_model,
    shoulder_target_layer,
    test_dataset,
    shoulder_true,
    shoulder_pred,
    "Shoulder Grad-CAM - False Negatives",
)

show_gradcam_grid(
    shoulder_false_positives,
    shoulder_model,
    shoulder_target_layer,
    test_dataset,
    shoulder_true,
    shoulder_pred,
    "Shoulder Grad-CAM - False Positives",
)

show_gradcam_grid(
    correct_shoulders,
    shoulder_model,
    shoulder_target_layer,
    test_dataset,
    shoulder_true,
    shoulder_pred,
    "Shoulder Grad-CAM - Correct",
)

show_gradcam_grid(
    gore_failures,
    gore_model,
    gore_target_layer,
    test_dataset,
    gore_true,
    gore_pred,
    "Gore Grad-CAM - Failures",
)

show_gradcam_grid(
    correct_gore,
    gore_model,
    gore_target_layer,
    test_dataset,
    gore_true,
    gore_pred,
    "Gore Grad-CAM - Correct",
)


# ============================================================
# 19. Shoulder Majority Baseline
# ============================================================

majority_baseline = max(
    shoulder_true.mean(),
    1 - shoulder_true.mean(),
)

print("\n================ BASELINE COMPARISON ================")
print(f"Shoulder majority-class baseline: {majority_baseline:.4f}")
print(f"Shoulder model accuracy:          {shoulder_acc:.4f}")


# ============================================================
# 20. Final Summary
# ============================================================

print("\n================ FINAL SUMMARY ================")
print(f"Overall Label Accuracy: {overall_label_accuracy:.4f}")
print(f"Exact-Match Accuracy:   {exact_match_accuracy:.4f}")
print(f"Shoulder Accuracy:      {shoulder_acc:.4f}")
print(f"Gore Accuracy:          {gore_acc:.4f}")

print("\nOptimal thresholds:")
print(f"Shoulder: {shoulder_threshold:.2f}")
print(f"Gore:     {gore_threshold:.2f}")


# ============================================================
# 21. Shoulder Threshold Sweep
# ============================================================

print("\nThreshold | Accuracy | Precision | Recall | F1")
print("-" * 60)

for thresh in np.arange(0.20, 0.81, 0.02):
    preds = (shoulder_probs >= thresh).astype(int)

    acc = accuracy_score(shoulder_true, preds)
    precision = precision_score(shoulder_true, preds, zero_division=0)
    recall = recall_score(shoulder_true, preds, zero_division=0)
    f1 = f1_score(shoulder_true, preds, zero_division=0)

    print(
        f"{thresh:.2f}      "
        f"{acc:.4f}      "
        f"{precision:.4f}      "
        f"{recall:.4f}      "
        f"{f1:.4f}"
    )


# ============================================================
# 22. Shoulder False-Negative Probabilities
# ============================================================

print("\nFALSE NEGATIVE SHOULDER PROBABILITIES")
print("--------------------------------------")

for idx in shoulder_false_negatives:
    print(
        f"Index: {idx:3d} | "
        f"Probability: {shoulder_probs[idx]:.4f} | "
        f"Threshold: {shoulder_threshold:.2f}"
    )


# ============================================================
# 23. Shoulder Probability Distribution
# ============================================================

positive_probs = shoulder_probs[shoulder_true == 1]
negative_probs = shoulder_probs[shoulder_true == 0]

print("\nSHOULDER POSITIVE IMAGES")
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
plt.hist(positive_probs, bins=15, alpha=0.6, label="Shoulder")
plt.hist(negative_probs, bins=15, alpha=0.6, label="No Shoulder")
plt.axvline(
    shoulder_threshold,
    linestyle="--",
    label=f"Threshold = {shoulder_threshold:.2f}",
)
plt.xlabel("Predicted Shoulder Probability")
plt.ylabel("Number of Images")
plt.title("Shoulder Probability Distribution")
plt.legend()
plt.show()


# ============================================================
# 24. Shoulder Errors by Gore Presence
# ============================================================
# Gore TRUE labels are still available because both models use
# the exact same test images.
# ============================================================

gore_mask = gore_true == 1
no_gore_mask = gore_true == 0
shoulder_error = shoulder_pred != shoulder_true

error_rate_gore = shoulder_error[gore_mask].mean()
error_rate_no_gore = shoulder_error[no_gore_mask].mean()

failures_with_gore = np.sum(shoulder_error & gore_mask)
failures_without_gore = np.sum(shoulder_error & no_gore_mask)

total_with_gore = np.sum(gore_mask)
total_without_gore = np.sum(no_gore_mask)

print("\n================ SHOULDER ERRORS BY GORE PRESENCE ================")
print(f"Shoulder error rate WITH gore:    {error_rate_gore:.4f}")
print(f"Shoulder error rate WITHOUT gore: {error_rate_no_gore:.4f}")

print(
    f"Shoulder failures WITH gore: "
    f"{failures_with_gore} / {total_with_gore}"
)
print(
    f"Shoulder failures WITHOUT gore: "
    f"{failures_without_gore} / {total_without_gore}"
)

if np.sum(shoulder_error) > 0:
    print(
        "Percent of shoulder failures containing gore: "
        f"{failures_with_gore / np.sum(shoulder_error) * 100:.2f}%"
    )


gore_shoulder_positive = (
    (gore_true == 1) &
    (shoulder_true == 1)
)

gore_shoulder_negative = (
    (gore_true == 1) &
    (shoulder_true == 0)
)

print("\n================ GORE / SHOULDER SUBGROUPS ================")

if np.sum(gore_shoulder_positive) > 0:
    error_gore_shoulder = shoulder_error[gore_shoulder_positive].mean()
    print("Gore + Shoulder error rate:", error_gore_shoulder)
    print(
        "Gore + Shoulder:",
        np.sum(shoulder_error & gore_shoulder_positive),
        "/",
        np.sum(gore_shoulder_positive),
    )

if np.sum(gore_shoulder_negative) > 0:
    error_gore_no_shoulder = shoulder_error[gore_shoulder_negative].mean()
    print("Gore + No Shoulder error rate:", error_gore_no_shoulder)
    print(
        "Gore + No Shoulder:",
        np.sum(shoulder_error & gore_shoulder_negative),
        "/",
        np.sum(gore_shoulder_negative),
    )