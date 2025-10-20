import os, sys, json, glob, math, argparse
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm 
import torch.nn.functional as F
from skimage.morphology import remove_small_objects, remove_small_holes, opening, closing, dilation, erosion, disk

def postprocess_band(pred_bool, min_obj=64, hole_area=64, open_r=1, close_r=1, band_width=0):
    x = pred_bool.astype(bool)
    if min_obj and min_obj > 0:
        x = remove_small_objects(x, min_size=min_obj)
    if hole_area and hole_area > 0:
        x = remove_small_holes(x, area_threshold=hole_area)
    if open_r and open_r > 0:
        x = opening(x, footprint=disk(open_r))
    if close_r and close_r > 0:
        x = closing(x, footprint=disk(close_r))
    if band_width and band_width != 0:
        se = disk(abs(band_width))
        x = dilation(x, se) if band_width > 0 else erosion(x, se)
    return x

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def list_images(d):
    p = Path(d)
    return sorted([str(x) for x in p.rglob("*") if x.suffix.lower() in IMG_EXTS])

class SegFolder(Dataset):
    def __init__(self, img_dir, mask_dir, imgsz=640, num_classes=1, augment=False):
        self.img_paths = list_images(img_dir)
        self.mask_dir = Path(mask_dir)
        self.imgsz = imgsz
        self.num_classes = num_classes
        self.augment = augment

        try:
            import albumentations as A
            self.A = A
            self.tf = A.Compose([
                A.LongestMaxSize(max_size=imgsz),
                A.PadIfNeeded(min_height=imgsz, min_width=imgsz, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.HorizontalFlip(p=0.5 if augment else 0.0),
                A.VerticalFlip(p=0.25 if augment else 0.0),
            ])
        except Exception:
            self.A = None
            self.tf = None

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        ip = self.img_paths[idx]
        stem = Path(ip).stem
        for ext in [".png", ".jpg", ".jpeg", ".tif"]:
            mp = self.mask_dir / f"{stem}{ext}"
            if mp.exists():
                break
        if not mp.exists():
            raise FileNotFoundError(f"Mask not found for {ip}")

        img = np.array(Image.open(ip).convert("RGB"))
        mask_raw = np.array(Image.open(mp))
        if self.num_classes == 1:
            mask = (mask_raw > 0).astype(np.uint8)
        else:
            mask = mask_raw.astype(np.int64)

        if self.tf is not None:
            aug = self.tf(image=img, mask=mask)
            img = aug["image"]; mask = aug["mask"]
        else:
            img = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            if self.num_classes == 1:
                mask = cv2.resize(mask, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST)
            else:
                mask = cv2.resize(mask, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  
        if self.num_classes == 1:
            mask = mask[None, ...].astype(np.float32)  
        else:
            mask = mask.astype(np.int64)              

        return torch.from_numpy(img), torch.from_numpy(mask), stem


class FPNDecoder(nn.Module):
    def __init__(self, in_channels_list, out_channels=128):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats):
        lat = [l(f) for l, f in zip(self.lateral, feats)]
        x = lat[-1]
        for i in range(len(lat) - 2, -1, -1):
            x = F.interpolate(x, size=lat[i].shape[-2:], mode="bilinear", align_corners=False) + lat[i]
        return self.out_conv(x)

class HRNetV2Seg(nn.Module):
    def __init__(self, encoder_name="hrnet_w18", num_classes=1):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, features_only=True, pretrained=True, out_indices=(0,1,2,3))
        chs = self.encoder.feature_info.channels() 
        self.decoder = FPNDecoder(chs, out_channels=128)
        self.head = nn.Conv2d(128, num_classes if num_classes > 1 else 1, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[-2:]                
        feats = self.encoder(x)             
        x = self.decoder(feats)             
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        logits = self.head(x)               
        return logits


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2,3)) + eps
    den = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + eps
    return 1 - (num / den).mean()

def train_one_epoch(model, loader, optimizer, device, criterion, num_classes=1):
    model.train()
    total = 0.0
    for img, mask, _ in tqdm(loader, desc="Train", leave=False):
        img = img.to(device)
        if num_classes == 1:
            mask = mask.to(device)      
            mask = mask.to(device)    

        optimizer.zero_grad()
        logits = model(img)

        if num_classes == 1:
            loss_bce = F.binary_cross_entropy_with_logits(logits, mask)
            loss_dice = dice_loss(logits, mask)
            loss = loss_bce + loss_dice
        else:
            loss = F.cross_entropy(logits, mask)

        loss.backward()
        optimizer.step()
        total += loss.item() * img.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def eval_one_epoch(model, loader, device, num_classes=1):
    model.eval()
    total_iou = 0.0
    n_pix = 0
    for img, mask, _ in tqdm(loader, desc="Valid", leave=False):
        img = img.to(device)
        logits = model(img)

        if num_classes == 1:
            pred = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
            gt   = mask.numpy()
            inter = (pred * gt).sum()
            uni   = ((pred + gt) > 0).sum()
            total_iou += (inter / (uni + 1e-9))
            n_pix += 1
        else:
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            gt   = mask.numpy()
            inter = ((pred == gt) & (gt > 0)).sum()
            uni   = ((pred > 0) | (gt > 0)).sum()
            total_iou += (inter / (uni + 1e-9))
            n_pix += 1
    return total_iou / max(1, n_pix)


def cmd_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model = HRNetV2Seg(encoder_name=args.backbone, num_classes=args.num_classes).to(device)

    ds_tr = SegFolder(args.images, args.masks, imgsz=args.imgsz, num_classes=args.num_classes, augment=True)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)

    if args.val_images and args.val_masks:
        ds_va = SegFolder(args.val_images, args.val_masks, imgsz=args.imgsz, num_classes=args.num_classes, augment=False)
        dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    else:
        dl_va = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_score, best_path = -1.0, None
    out_w = Path(args.weights); out_w.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, dl_tr, optimizer, device, None, num_classes=args.num_classes)
        if dl_va is not None:
            score = eval_one_epoch(model, dl_va, device, num_classes=args.num_classes)
            print(f"[Epoch {ep}] train_loss={tr_loss:.4f}  val_mIoU={score:.4f}")
            if score > best_score:
                best_score = score
                best_path = str(out_w)
                torch.save({"model": model.state_dict(),
                            "num_classes": args.num_classes,
                            "backbone": args.backbone,
                            "imgsz": args.imgsz}, best_path)
        else:
            print(f"[Epoch {ep}] train_loss={tr_loss:.4f}")

    if best_path is None:
        torch.save({"model": model.state_dict(),
                    "num_classes": args.num_classes,
                    "backbone": args.backbone,
                    "imgsz": args.imgsz}, str(out_w))
        best_path = str(out_w)


@torch.no_grad()
def cmd_predict(args):
    print(f"[predict] threshold={args.thr}, save_prob={args.save_prob}, "
          f"pp(min_obj={args.min_obj}, hole={args.hole_area}, open={args.open_r}, close={args.close_r}, "
          f"band_width={args.band_width}), gbm_idx={args.gbm_idx}")

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    ckpt = torch.load(args.weights, map_location=device)
    num_classes = ckpt.get("num_classes", 1)
    backbone = ckpt.get("backbone", args.backbone)
    model = HRNetV2Seg(encoder_name=backbone, num_classes=num_classes)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    img_paths = list_images(args.source)
    if not img_paths:
        sys.exit(1)

    out_dir = Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)
    comb_dir = out_dir / "combined"; comb_dir.mkdir(exist_ok=True)
    if args.overlay:
        ov_dir = out_dir / "overlay"; ov_dir.mkdir(exist_ok=True)
    prob_dir = out_dir / "probs"
    if args.save_prob:
        prob_dir.mkdir(exist_ok=True)

    for ip in tqdm(img_paths, desc="Predict"):
        img = np.array(Image.open(ip).convert("RGB"))
        H0, W0 = img.shape[:2]
        img_r = cv2.resize(img, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
        ten = torch.from_numpy(img_r.transpose(2,0,1)).float().unsqueeze(0) / 255.0
        ten = ten.to(device)

        logits = model(ten)  
        logits = F.interpolate(logits, size=(H0, W0), mode="bilinear", align_corners=False)

        stem = Path(ip).stem

        if num_classes == 1:
            prob = torch.sigmoid(logits)            
            if args.save_prob:
                np.save(prob_dir / f"{stem}.npy", prob.squeeze().cpu().numpy().astype(np.float32))
            pred = (prob >= args.thr).to(torch.uint8).cpu().numpy()[0,0]  
        else:
            if args.gbm_idx is not None and args.gbm_idx >= 0:
                prob = torch.softmax(logits, dim=1)[:, args.gbm_idx:args.gbm_idx+1]  
                if args.save_prob:
                    np.save(prob_dir / f"{stem}.npy", prob.squeeze().cpu().numpy().astype(np.float32))
                pred = (prob >= args.thr).to(torch.uint8).cpu().numpy()[0,0]        
            else:
                cls = torch.argmax(logits, dim=1).cpu().numpy()[0]               
                pred = (cls > 0).astype(np.uint8)

        pred_pp = postprocess_band(
            pred,
            min_obj=args.min_obj,
            hole_area=args.hole_area,
            open_r=args.open_r,
            close_r=args.close_r,
            band_width=args.band_width
        ).astype(np.uint8)

        out_mask = (pred_pp * 255).astype(np.uint8)
        Image.fromarray(out_mask).save(str(comb_dir / f"{stem}.png"))

        if args.overlay:
            overlay = img.copy()
            color = np.zeros_like(overlay)
            color[:, :, 1] = pred_pp * 255  
            over = cv2.addWeighted(overlay, 1.0, color, 0.35, 0.0)
            cv2.imwrite(str((ov_dir / f"{stem}_overlay.png")), cv2.cvtColor(over, cv2.COLOR_RGB2BGR))

    print("Predict done. Results at:", out_dir)


def build_parser():
    p = argparse.ArgumentParser(description="HRNetV2 Segmentation (timm)")

    sub = p.add_subparsers(dest="cmd", required=True)

    # Train
    pt = sub.add_parser("train", help="Train HRNetV2 segmentation")
    pt.add_argument("--images", required=True)
    pt.add_argument("--masks", required=True)
    pt.add_argument("--val-images", default=None)
    pt.add_argument("--val-masks", default=None)
    pt.add_argument("--epochs", type=int, default=100)
    pt.add_argument("--batch", type=int, default=8)
    pt.add_argument("--imgsz", type=int, default=640)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--workers", type=int, default=8)
    pt.add_argument("--device", type=str, default="0")
    pt.add_argument("--num-classes", type=int, default=1, help="1=binary; >1 for multi-class")
    pt.add_argument("--backbone", type=str, default="hrnet_w18", help="timm HRNetV2, e.g., hrnet_w18, hrnet_w32")
    pt.add_argument("--weights", type=str, default="out/hrnetv2_w18_seg.pt")
    pt.set_defaults(func=cmd_train)

    pp = sub.add_parser("predict", help="Predict single combined mask per image (+optional overlay)")
    pp.add_argument("--weights", required=True)
    pp.add_argument("--source", required=True)
    pp.add_argument("--outdir", type=str, default="pred_hrnet")
    pp.add_argument("--imgsz", type=int, default=640)
    pp.add_argument("--device", type=str, default="0")
    pp.add_argument("--backbone", type=str, default="hrnet_w18")
    pp.add_argument("--overlay", action="store_true")
    pp.set_defaults(func=cmd_predict)
    pp.add_argument("--thr", type=float, default=0.39, help="binary threshold for prob->mask")
    pp.add_argument("--save-prob", action="store_true", help="save probability maps (.npy)")
    pp.add_argument("--gbm-idx", type=int, default=-1, help="class index to binarize (for multi-class); <0 to disable")
    pp.add_argument("--min-obj", type=int, default=64, help="remove small objects (pixels)")
    pp.add_argument("--hole-area", type=int, default=64, help="fill small holes (area in pixels)")
    pp.add_argument("--open-r", type=int, default=1, help="morphological opening radius (disk)")
    pp.add_argument("--close-r", type=int, default=1, help="morphological closing radius (disk)")
    pp.add_argument("--band-width", type=int, default=0, help="band width tuning: dilate(+k)/erode(-k) pixels")


    return p

def main():
    args = build_parser().parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
