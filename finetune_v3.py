import os, sys, torch, torch.nn as nn, time, glob, numpy as np
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
torch._dynamo.config.disable = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

UOCOD_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, UOCOD_ROOT)
os.chdir(UOCOD_ROOT)

from engine.config.config import CfgNode
from models.uscod import baseline
from data.utils.feature_extractor import build_feature_extractor
from models.modules.full_model import ViTLoraWrapper, load_lora
from safetensors.torch import load_file, save_file
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

DEVICE = 'cuda'
BATCH, EPOCHS, LR = 4, 10, 1e-4
FS = 68

# ===== Dataset =====
class COD10KDataset(Dataset):
    def __init__(self, img_dir, gt_dir):
        self.paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        self.gt_dir = gt_dir
        self.t_img = transforms.Compose([
            transforms.Resize((518, 518)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.t_gt = transforms.Compose([
            transforms.Resize((FS, FS)), transforms.ToTensor()])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = self.t_img(Image.open(self.paths[i]).convert('RGB'))
        gt_path = os.path.join(self.gt_dir, os.path.basename(self.paths[i]).replace('.jpg', '.png'))
        gt = (self.t_gt(Image.open(gt_path).convert('L')) > 0.5).float()
        return img, gt

# ===== Model =====
print("Loading...")
cfg = CfgNode.load_with_base(os.path.join(UOCOD_ROOT, "configs/uscod/UCOD-DPL_dinov2.py"))
cfg = CfgNode(cfg)
cfg.model_cfg.feature_size = FS
cfg.lora_cfg = CfgNode(dict(r=2, lora_alpha=4, lora_dropout=0.05, bias='none',
    target_modules=['query', 'value', 'key'], lora_task_type='FEATURE_EXTRACTION'))

decoder = baseline(cfg.model_cfg)
decoder.load_state_dict(load_file(os.path.join(UOCOD_ROOT, "weights/UCOD_DPL_dinov2.safetensors"), device=DEVICE))
decoder.to(DEVICE)

backbone = build_feature_extractor(cfg.dataset_cfg.feature_extractor_cfg)
backbone = ViTLoraWrapper(backbone)
backbone = load_lora(cfg.lora_cfg, backbone)
backbone.to(DEVICE)
backbone.eval()
for n, p in backbone.named_parameters():
    p.requires_grad = ('lora' in n.lower())

n_train = sum(p.numel() for p in decoder.parameters())
n_train += sum(p.numel() for p in backbone.parameters() if p.requires_grad)
print(f"Trainable: {n_train:,}")

_key = None
def hook_fn(m, i, o):
    global _key
    o = o[:, 1:, :]; b, l, c = o.shape; h = w = int(l**0.5)
    _key = o.reshape(b, h, w, c).permute(0, 3, 1, 2)
    _key = torch.nn.functional.interpolate(_key, size=(FS, FS), mode='bilinear')
backbone.ViT.encoder.layer[-1].attention.attention.key.register_forward_hook(hook_fn)

# ===== Data =====
print("Loading data...")
train_ds = COD10KDataset("datasets/COD10K_FT/im", "datasets/COD10K_FT/gt")
val_ds = COD10KDataset("datasets/COD10K_FT_VAL/im", "datasets/COD10K_FT_VAL/gt")
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

optimizer = torch.optim.AdamW(list(decoder.parameters()) +
    [p for p in backbone.parameters() if p.requires_grad], lr=LR)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

# ===== Train =====
print(f"\nTraining {EPOCHS} epochs...")
out_dir = os.path.join(UOCOD_ROOT, "finetune_output"); os.makedirs(out_dir, exist_ok=True)
best_val = float('inf')

for ep in range(EPOCHS):
    decoder.train()
    t0 = time.time(); tr_loss = 0
    for imgs, gts in train_loader:
        imgs, gts = imgs.to(DEVICE), gts.to(DEVICE)
        optimizer.zero_grad()
        with torch.no_grad(): backbone(pixel_values=imgs)
        preds, _, _ = decoder(_key)
        loss = criterion(preds, gts)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
    avg_tr = tr_loss / len(train_loader)

    decoder.eval()
    vl_loss = 0
    with torch.no_grad():
        for imgs, gts in val_loader:
            imgs, gts = imgs.to(DEVICE), gts.to(DEVICE)
            backbone(pixel_values=imgs)
            preds, _, _ = decoder(_key)
            vl_loss += criterion(preds, gts).item()
    avg_vl = vl_loss / len(val_loader)
    scheduler.step()

    print(f"Epoch {ep+1}: train={avg_tr:.4f} val={avg_vl:.4f} ({time.time()-t0:.0f}s)")

    if avg_vl < best_val:
        best_val = avg_vl
        save_file(decoder.state_dict(), os.path.join(out_dir, "best.safetensors"))

save_file(decoder.state_dict(), os.path.join(out_dir, "final.safetensors"))
print(f"Done. Best val={best_val:.4f}")
