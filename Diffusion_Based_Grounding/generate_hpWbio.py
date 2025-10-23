# -*- coding: utf-8 -*-
import os
os.environ["TQDM_DISABLE"] = "1"

from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

# ===== 工程依赖 =====
from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from health_multimodal.vlp import ImageTextInferenceEngine
from data import MSCXR

# =======================
# 配置
# =======================
GPU_ID          = 0
SAVE_DIR        = "/media/yuganlab/ezstore1/Zongye/Code/Diffusion_Based_Grounding/Case_Study/HC_fig1/raw_heatmap_otsu_gap"
TRANSFORM_NAME  = "biovil_t"
CROP_SIZE       = 448

TARGET_SIZE     = 512
HEAT_ALPHA_BASE = 0.55
OTSU_BINS       = 256

# 背景“灰白化”强度（0~1，数值越大背景越接近白）
BG_LIGHTEN      = 0.15

# 仅对非 NaN 区域进行 GAP 平滑（不做向 NaN 扩散）
GAP_KERNEL_SIZE = 99
CONV_PAD_MODE   = "reflect"   # 可选: "reflect" | "replicate" | "constant"

# 绿色框
BOX_COLOR  = (0, 255, 0)
LINE_WIDTH = 4

# =======================
# 模型加载
# =======================
def load_biovil_t(device: torch.device) -> ImageTextInferenceEngine:
    text_inf  = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)
    image_inf = get_image_inference(ImageModelType.BIOVIL_T)
    engine    = ImageTextInferenceEngine(image_inference_engine=image_inf,
                                         text_inference_engine=text_inf)
    engine.to(device)
    return engine

# =======================
# 背景处理（灰度→RGB 并灰白化）
# =======================
def make_background_512(image_path: str) -> Image.Image:
    with Image.open(image_path) as im:
        im = im.convert("L")
        im_t = TF.to_tensor(im)                                   # (1,H,W) [0,1]
        im_t = TF.resize(im_t, [TARGET_SIZE, TARGET_SIZE])
        arr = (im_t.squeeze(0).numpy() * 255).astype(np.uint8)
        rgb = Image.fromarray(arr, mode="L").convert("RGB")

        # 背景稍微调白一点以提升对比度
        if BG_LIGHTEN > 1e-6:
            base_np = np.array(rgb, dtype=np.float32)
            base_np = (1.0 - BG_LIGHTEN) * base_np + BG_LIGHTEN * 255.0
            rgb = Image.fromarray(np.clip(base_np, 0, 255).astype(np.uint8), mode="RGB")
        return rgb

# =======================
# 仅对非 NaN 区域做 GAP（shape 不变）
# =======================
def conv_smooth_nan_aware(sim_01: torch.Tensor, kernel_size: int, pad_mode: str = "reflect") -> torch.Tensor:
    """
    对有限值区域做均值平滑；NaN 区域不参与、也不被填充（保持 NaN）。
    """
    assert kernel_size % 2 == 1
    x = sim_01.to(torch.float32)          # (H,W)
    x4 = x[None, None, ...]               # (1,1,H,W)
    finite = torch.isfinite(x4)           # True 表示有效
    x_zero = torch.where(finite, x4, torch.zeros_like(x4))
    mask   = finite.to(x4.dtype)

    r = kernel_size // 2
    kernel = torch.ones((1,1,kernel_size,kernel_size), dtype=torch.float32, device=x.device)

    if pad_mode not in ("reflect", "replicate", "constant"):
        pad_mode = "reflect"
    pad = (r, r, r, r)

    x_p    = F.pad(x_zero, pad, mode=pad_mode)
    mask_p = F.pad(mask,   pad, mode=pad_mode)

    sum_v   = F.conv2d(x_p,    kernel, stride=1)
    count_v = F.conv2d(mask_p, kernel, stride=1)

    out = (sum_v / (count_v + 1e-12))[0,0]
    # 对没有任何有效邻域的像素维持 NaN
    no_info = (count_v[0,0] <= 0)
    if no_info.any():
        out = out.clone()
        out[no_info] = float('nan')
    return out

# =======================
# 红色深浅叠加（可控是否归一化）
# =======================
def overlay_reddepth_on_rgb_512(base_512: Image.Image,
                                sim_in: torch.Tensor,
                                alpha_base: float = HEAT_ALPHA_BASE,
                                normalize: bool = True) -> Image.Image:
    """
    若 normalize=True：对输入按 min-max 归一化到 [0,1]（原行为）。
    若 normalize=False：不做归一化，仅对输入做 NaN->0 与 [0,1] 限幅。
    """
    base = base_512.convert("RGBA")
    s = torch.as_tensor(sim_in, dtype=torch.float32)

    if normalize:
        finite = torch.isfinite(s)
        if finite.any():
            v = s[finite]
            smin = float(v.min()); smax = float(v.max())
            rng = smax - smin if smax > smin else 1e-12
            s = ((s - smin) / rng)
        else:
            s = torch.zeros_like(s)
        s = torch.where(torch.isfinite(s), s, torch.zeros_like(s)).clamp_(0.0, 1.0)
    else:
        # 不做归一化，只做限幅
        s = torch.nan_to_num(s, nan=0.0).clamp_(0.0, 1.0)

    arr = s.cpu().numpy()
    R = (arr * 255.0)
    A = (np.clip(alpha_base * arr, 0.0, 1.0) * 255.0)

    heat_rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    heat_rgba[..., 0] = R.astype(np.uint8)
    heat_rgba[..., 1] = 0
    heat_rgba[..., 2] = 0
    heat_rgba[..., 3] = A.astype(np.uint8)

    heat = Image.fromarray(heat_rgba, mode="RGBA")
    out = base.copy()
    out.alpha_composite(heat)
    return out.convert("RGB")

# =======================
# Otsu 阈值
# =======================
def otsu_threshold_01(sim_01: torch.Tensor, nbins: int = OTSU_BINS) -> float:
    x = sim_01.detach().to(torch.float32).clamp(0.0, 1.0)
    vals = x.flatten()
    if torch.allclose(vals.max(), vals.min()):
        return float(vals.mean())
    hist = torch.histc(vals, bins=nbins, min=0.0, max=1.0).double()
    if hist.sum() <= 0:
        return 0.5
    probs = hist / hist.sum()
    bin_centers = torch.linspace(0.0, 1.0, steps=nbins, dtype=torch.double)
    omega = torch.cumsum(probs, dim=0)
    mu_k  = torch.cumsum(probs * bin_centers, dim=0)
    mu_t  = (probs * bin_centers).sum()
    sigma_b2 = (mu_t * omega - mu_k) ** 2 / ((omega * (1.0 - omega)).clamp_min(1e-12))
    sigma_b2[(omega <= 1e-8) | (omega >= 1.0 - 1.0e-8)] = -1.0
    return float(bin_centers[int(torch.argmax(sigma_b2).item())])

# =======================
# bbox 缩放与绘制
# =======================
def scale_boxes_from_original_to_512(original_bbox: List[Tuple[float, float, float, float]],
                                     orig_w: int, orig_h: int, tgt: int = TARGET_SIZE):
    sx = tgt / float(orig_w); sy = tgt / float(orig_h)
    out = []
    for (x, y, w, h) in original_bbox:
        x1 = int(round(x * sx)); y1 = int(round(y * sy))
        x2 = int(round((x + w) * sx)); y2 = int(round((y + h) * sy))
        x1 = max(0, min(tgt, x1)); y1 = max(0, min(tgt, y1))
        x2 = max(0, min(tgt, x2)); y2 = max(0, min(tgt, y2))
        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2, y2))
    return out

def draw_boxes_512(img_512: Image.Image, boxes_xyxy, color=BOX_COLOR, width=LINE_WIDTH):
    if not boxes_xyxy:
        return img_512
    im = img_512.copy().convert("RGB")
    draw = ImageDraw.Draw(im)
    for (x1, y1, x2, y2) in boxes_xyxy:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return im

# =======================
# 主流程：原始 heatmap →（仅对非NaN做GAP）→ resize → Otsu（含背景灰白化）
# =======================
@torch.no_grad()
def main():
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(SAVE_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    engine = load_biovil_t(device=device)
    ds = MSCXR(transform_name=TRANSFORM_NAME, crop_size=CROP_SIZE)

    print(f"[INFO] Total={len(ds)} | Raw heatmap + GAP(non-NaN only) + Otsu (no diffusion) with background lightening")

    for idx in range(len(ds)):
        try:
            sample = ds[idx]
            if len(sample) >= 6:
                _, _, original_bbox, prompt, image_id, _ = sample
            else:
                _, _, _, prompt, image_id, _ = sample
                original_bbox = []

            stem = Path(image_id).stem
            base_512 = make_background_512(image_id)

            # 原图尺寸 -> bbox 缩放
            with Image.open(image_id) as im_info:
                orig_w, orig_h = im_info.size
            boxes_xyxy_512 = scale_boxes_from_original_to_512(original_bbox, orig_w, orig_h)

            # 原始相似度
            sim_np = engine.get_similarity_map_from_raw_data(Path(image_id), prompt)
            sim = torch.from_numpy(sim_np).float()

            # 归一化到 [0,1]（保留 NaN 位置）
            finite = torch.isfinite(sim)
            if not finite.any():
                print(f"[WARN] idx={idx} all NaN heatmap")
                continue
            v = sim[finite]
            smin, smax = float(v.min()), float(v.max())
            sim01 = ((sim - smin) / max(smax - smin, 1e-12)).clamp(0.0, 1.0)

            # === 仅对非 NaN 区域做 GAP 平滑（不向 NaN 扩散）===
            sim01_gap = conv_smooth_nan_aware(sim01, kernel_size=GAP_KERNEL_SIZE, pad_mode=CONV_PAD_MODE)

            # resize 到 512×512（使用最近邻避免插值造成泄露）
            sim_512 = F.interpolate(sim01_gap[None, None, ...], size=(TARGET_SIZE, TARGET_SIZE),
                                    mode="nearest")[0, 0]

            # (A) GAP 后的红色深浅 + 框（背景已灰白化）
            # 保持“全幅图”原有可视化风格 -> normalize=True
            heat_full = overlay_reddepth_on_rgb_512(base_512, sim_512, alpha_base=HEAT_ALPHA_BASE, normalize=True)
            heat_full_boxes = draw_boxes_512(heat_full, boxes_xyxy_512)
            out_full = out_dir / f"idx{idx}_{stem}_gap_full_boxes.png"
            heat_full_boxes.save(out_full)

            # (B) Otsu 阈值 + 框（在 GAP 后的 heatmap 上做）
            thr = otsu_threshold_01(sim_512)
            sim_otsu = torch.where(sim_512 >= thr, sim_512, torch.zeros_like(sim_512))
            # 关键：Otsu 分支 **不再二次归一化**（normalize=False）
            heat_otsu = overlay_reddepth_on_rgb_512(base_512, sim_otsu, alpha_base=HEAT_ALPHA_BASE, normalize=False)
            heat_otsu_boxes = draw_boxes_512(heat_otsu, boxes_xyxy_512)
            out_otsu = out_dir / f"idx{idx}_{stem}_gap_otsu_boxes.png"
            heat_otsu_boxes.save(out_otsu)

            if idx % 25 == 0:
                print(f"[{idx+1}/{len(ds)}] Saved {out_full.name} | {out_otsu.name}")

        except Exception as e:
            print(f"[ERROR] idx={idx} | {e}")

    print(f"[DONE] Saved to: {out_dir}")

if __name__ == "__main__":
    main()
