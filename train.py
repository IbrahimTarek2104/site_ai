import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import jaccard_score, f1_score

from pipeline.network import AttentionUNet
from pipeline.data_pipeline import ProductionInferencePipeline

# ── 1. CONFIGURATION & DEVICE SETTING ──
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 40
BATCH_SIZE = 16
PATCH_SIZE = 64
STRIDE = 32

print(f"🚀 Initializing training pipeline on device: {DEVICE}")

# ── 2. DATA INGESTION & PIPELINE PREPROCESSING ──
# Point these paths directly to your true dataset files on disk
cov_path = "C:/Users/Ibrahim Tarek/Desktop/CNN_GRADproject/last data/last coverage.tif"
pop_path = "C:/Users/Ibrahim Tarek/Desktop/CNN_GRADproject/last data/lastpop.tif"
elev_path = "C:/Users/Ibrahim Tarek/Desktop/CNN_GRADproject/last data/lastlast elevation.tif"
label_path = "C:/Users/Ibrahim Tarek/Desktop/CNN_GRADproject/last data/poor_coverage_label.tif"


if not (os.path.exists(cov_path) and os.path.exists(pop_path) and os.path.exists(elev_path)):
    raise FileNotFoundError("Missing raw .tif components in data/raw/. Please check file locations.")

# Run your production pipeline to build a clean 3-band feature stack
pipeline = ProductionInferencePipeline(patch_size=PATCH_SIZE)
features_norm, _ = pipeline.extract_and_normalize(cov_path, pop_path, elev_path)

# Transpose features to matching PyTorch layout format: (H, W, C) -> (C, H, W)
features_norm = np.moveaxis(features_norm, -1, 0)

# Load your pre-computed binary target label raster
import rasterio
with rasterio.open(label_path) as src:
    labels = src.read(1).astype(np.float32)

# Calculate minority class balancing weights for cross entropy term
pos_ratio = float(labels.mean())
pos_weight = torch.tensor((1.0 - pos_ratio) / (pos_ratio + 1e-8), dtype=torch.float32)
print(f"📊 Class Balance Check: {pos_ratio*100:.1f}% poor coverage pixels.")
print(f"⚖️ Loss Weight Factor Multiplier: {pos_weight.item():.2f}x")

# ── 3. DATASET PATCH GENERATION & AUGMENTATION ──
class SpatialPatchDataset(Dataset):
    def __init__(self, features, labels, patch_size=64, stride=32, augment=False):
        self.features = features
        self.labels = labels
        self.patch_size = patch_size
        self.augment = augment
        _, H, W = features.shape
        
        # Slice entire scene into grid coordinates
        self.patches = [(r, c) 
                        for r in range(0, H - patch_size + 1, stride) 
                        for c in range(0, W - patch_size + 1, stride)]

    def __len__(self): 
        return len(self.patches)

    def __getitem__(self, idx):
        r, c = self.patches[idx]
        P = self.patch_size
        
        x = self.features[:, r:r+P, c:c+P].copy()
        y = self.labels[r:r+P, c:c+P].copy()
        
        # Spatial Augmentations to force geometric invariance
        if self.augment:
            if np.random.rand() > 0.5: # Horizontal flip
                x, y = np.flip(x, 2).copy(), np.flip(y, 1).copy()
            if np.random.rand() > 0.5: # Vertical flip
                x, y = np.flip(x, 1).copy(), np.flip(y, 0).copy()
            k = np.random.randint(0, 4) # Rotations
            x, y = np.rot90(x, k, (1, 2)).copy(), np.rot90(y, k).copy()
            
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Instantiate splits (70% Train, 15% Val, 15% Test)
full_dataset = SpatialPatchDataset(features_norm, labels, PATCH_SIZE, STRIDE, augment=False)
n_total = len(full_dataset)
n_train = int(0.70 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val

train_split, val_split, test_split = random_split(
    full_dataset, [n_train, n_val, n_test], 
    generator=torch.Generator().manual_seed(42)
)
# Enable augmentations ONLY on training set split slice
train_split.dataset.augment = True

train_loader = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_split, batch_size=BATCH_SIZE, shuffle=False)

# ── 4. LOSS CORES DEFINITIONS ──
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, pred, target):
        p, t = pred.contiguous().view(-1), target.contiguous().view(-1)
        intersection = (p * t).sum()
        return 1.0 - (2.0 * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth)

class ProductionCombinedLoss(nn.Module):
    def __init__(self, pos_weight=None, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.pw = pos_weight
        self.dice = DiceLoss()
    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        w = torch.ones_like(target)
        if self.pw is not None:
            w[target == 1] = self.pw.to(target.device)
        bce_loss = (bce * w).mean()
        dice_loss = self.dice(pred, target)
        return self.alpha * bce_loss + (1.0 - self.alpha) * dice_loss

criterion = ProductionCombinedLoss(pos_weight=pos_weight, alpha=0.5)

# ── 5. MODEL INSTANTIATION & OPTIMIZER TUNNELS ──
model = AttentionUNet(in_channels=3, base=64, drop=0.2).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

# ── 6. MAIN ENGINE TRAINING LOOP ──
def execute_epoch(model, loader, criterion, optimizer=None, is_train=True):
    model.train(is_train)
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.set_grad_enabled(is_train):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE).unsqueeze(1)
            out = model(xb)
            loss = criterion(out, yb)
            
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            total_loss += loss.item() * xb.size(0)
            all_preds.append((out.detach().cpu().numpy().ravel() >= 0.5).astype(np.uint8))
            all_targets.append(yb.detach().cpu().numpy().ravel().astype(np.uint8))
            
    return total_loss / len(loader.dataset), np.concatenate(all_preds), np.concatenate(all_targets)

os.makedirs("models", exist_ok=True)
best_iou = 0.0

print("\n--- Training Initialization Starting ---")
for epoch in range(1, EPOCHS + 1):
    train_loss, _, _ = execute_epoch(model, train_loader, criterion, optimizer, is_train=True)
    val_loss, v_preds, v_targets = execute_epoch(model, val_loader, criterion, None, is_train=False)
    
    scheduler.step()
    
    # Calculate performance validation metrics
    val_iou = jaccard_score(v_targets, v_preds, zero_division=0)
    val_f1 = f1_score(v_targets, v_preds, zero_division=0)
    
    # Checkpoint weight parameters if intersection score beats historical best
    checkpoint_tag = ""
    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.state_dict(), "models/unet_best.pth")
        checkpoint_tag = " ★ [Saved New Best Checkpoint]"
        
    print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | F1-Score: {val_f1:.4f}{checkpoint_tag}")

print(f"\n✅ Optimization complete! Best Model weights saved to: models/unet_best.pth")