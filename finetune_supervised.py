"""Supervised fine-tuning of UCOD-DPL decoder on COD10K with GT masks."""
import os, sys, time, glob, torch, torch.nn as nn, numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch._dynamo
torch._dynamo.config.disable = True

UOCOD_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, UOCOD_ROOT)
os.chdir(UOCOD_ROOT)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from engine.config.config import CfgNode
from models.uscod import baseline
from data.utils.feature_extractor import build_feature_extractor
from models.modules.full_model import ViTLoraWrapper, load_lora
from safetensors.torch import load_file, save_file

DEVICE = 'cuda'
BATCH_SIZE = 16
NUM_EPOCHS = 10
LR = 1e-4
FEATURE_SIZE = 68

# ===== Dataset =====
class COD10KDataset(Dataset):
    def __init__(self, img_dir, gt_dir):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        self.gt_dir = gt_dir
        self.img_transform = transforms.Compose([
            transforms.Resize((518, 518)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((FEATURE_SIZE, FEATURE_SIZE)), transforms.ToTensor(),
        ])
    def __len__(self): return len(self.img_paths)
    def __getitem__(self, idx):
        img = self.img_transform(Image.open(self.img_paths[idx]).convert('RGB'))
        gt_name = os.path.basename(self.img_paths[idx]).replace('.jpg', '.png')
        gt = self.gt_transform(Image.open(os.path.join(self.gt_dir, gt_name)).convert('L'))
        return img, (gt > 0.5).float(), self.img_paths[idx]

# ===== Load model (exact same pattern as uocod_detector.py) =====
print("Loading config...")
cfg = CfgNode.load_with_base(os.path.join(UOCOD_ROOT, "configs/uscod/UCOD-DPL_dinov2.py"))
cfg = CfgNode(cfg)
cfg.model_cfg.feature_size = FEATURE_SIZE
cfg.lora_cfg = CfgNode(dict(r=2, lora_alpha=4, lora_dropout=0.05, bias='none',
    target_modules=['query', 'value', 'key'], lora_task_type='FEATURE_EXTRACTION'))

print("Loading decoder...")
decoder = baseline(cfg.model_cfg)
ckpt = os.path.join(UOCOD_ROOT, "weights/UCOD_DPL_dinov2.safetensors")
decoder.load_state_dict(load_file(ckpt, device=DEVICE))
decoder = decoder.to(DEVICE)

print("Loading backbone...")
backbone = build_feature_extractor(cfg.dataset_cfg.feature_extractor_cfg)
backbone = ViTLoraWrapper(backbone)
backbone = load_lora(cfg.lora_cfg, backbone)
backbone = backbone.to(DEVICE)

# Freeze all backbone except LoRA
for name, param in backbone.named_parameters():
    param.requires_grad = ('lora' in name.lower())

n_train = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
n_train += sum(p.numel() for p in backbone.parameters() if p.requires_grad)
print(f"Trainable: {n_train:,}")

# Hook (exact same as uocod_detector.py)
_key = None
def hook_fn(module, input, output):
    global _key
    output = output[:, 1:, :]
    b, l, c = output.shape
    h = w = int(l**0.5)
    output = output.reshape(b, h, w, c).permute(0, 3, 1, 2)
    output = torch.nn.functional.interpolate(output, size=(FEATURE_SIZE, FEATURE_SIZE), mode='bilinear')
    _key = output

print("Registering hook...")
backbone.ViT.encoder.layer[-1].attention.attention.key.register_forward_hook(hook_fn)
print("Hook registered.")

# ===== Data =====
print("Loading training data...")
train_ds = COD10KDataset("datasets/COD10K_FT/im", "datasets/COD10K_FT/gt")
val_ds = COD10KDataset("datasets/COD10K_FT_VAL/im", "datasets/COD10K_FT_VAL/gt")
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

# ===== Optimizer =====
optimizer = torch.optim.AdamW(
    list(decoder.parameters()) + [p for p in backbone.parameters() if p.requires_grad], lr=LR)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS)

# ===== Training =====
print(f"\nTraining {NUM_EPOCHS} epochs...")
best_val = float('inf')
out_dir = os.path.join(UOCOD_ROOT, "finetune_output")
os.makedirs(out_dir, exist_ok=True)

for epoch in range(NUM_EPOCHS):
    decoder.train()
    backbone.eval()
    train_loss = 0
    t0 = time.time()
    for imgs, gts, _ in train_loader:
        imgs, gts = imgs.to(DEVICE), gts.to(DEVICE)
        optimizer.zero_grad()
        with torch.no_grad():
            backbone(pixel_values=imgs)
        preds, _, _ = decoder(_key)
        loss = criterion(preds, gts)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train = train_loss / len(train_loader)

    decoder.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, gts, _ in val_loader:
            imgs, gts = imgs.to(DEVICE), gts.to(DEVICE)
            backbone(pixel_values=imgs)
            preds, _, _ = decoder(_key)
            val_loss += criterion(preds, gts).item()
    avg_val = val_loss / len(val_loader)
    scheduler.step()

    print(f"Epoch {epoch+1}: train={avg_train:.4f} val={avg_val:.4f} time={time.time()-t0:.0f}s")

    if avg_val < best_val:
        best_val = avg_val
        save_file(decoder.state_dict(), os.path.join(out_dir, "best_decoder.safetensors"))

save_file(decoder.state_dict(), os.path.join(out_dir, "final_decoder.safetensors"))
print(f"Done. Best val_loss: {best_val:.4f}")
