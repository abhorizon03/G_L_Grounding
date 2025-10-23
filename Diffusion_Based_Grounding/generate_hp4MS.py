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

# local
from data import MSCXR
from sampler import Sampler
from models_local import get_models

# =======================
# 配置
# =======================
GPU_ID: int = 2
TRANSFORM_NAME: str = "ddpm"
NUM_TIMESTEPS: int = 300

STAGE1_CFG: str = "/media/yuganlab/ezstore1/Zongye/Code/Diffusion_Based_Grounding/configs/stage1/aekl_v0.yaml"
DIFFUSION_CFG: str = "/media/yuganlab/ezstore1/Zongye/Code/Diffusion_Based_Grounding/configs/ldm/ldm_v0.yaml"
STAGE1_WEIGHTS: str = "/media/yuganlab/ezstore1/Zongye/Code/G_L_Grounding/output_runs/simalign_0.005-topk_0.1-lr_1e-05_456/ckpt/e001_s009859/vae.pth"
DIFFUSION_WEIGHTS: str = "/media/yuganlab/ezstore1/Zongye/Code/G_L_Grounding/output_runs/simalign_0.005-topk_0.1-lr_1e-05_456/ckpt/e001_s009859/unet.pth"

# 采样窗口
T_MIN, T_MAX = 120, 180  # [T_MIN, T_MAX)
GUIDANCE_SCALE = 0.0
SCALE_FACTOR = 0.3
CLS_NAME_FOR_ATTR = None

# 可视化
SAVE_DIR = "Case_Study/HC_fig1/ours"
TARGET_SIZE = 512

# 热力图叠加相关
HEAT_ALPHA_BASE = 0.55   # 基础透明度系数（再乘以 sim 强度）
BG_LIGHTEN = 0.15        # 背景向白色偏移比例，0~1，数值越大越白

# 画框（深绿色）
BOX_COLOR = (0, 255, 0)
LINE_WIDTH = 4


# =======================
# 生成 512×512 背景（灰度→RGB）
# =======================
def make_transformed_background_like_ref(image_path: str) -> Image.Image:
    with Image.open(image_path) as im:
        im = im.convert("L")
        im_t = TF.to_tensor(im)  # (1,H,W), [0,1]
        im_t = TF.resize(im_t, size=TARGET_SIZE)
        im_t = TF.center_crop(im_t, output_size=[TARGET_SIZE, TARGET_SIZE])
        arr = (im_t.squeeze(0).numpy() * 255.0).astype(np.uint8)
        rgb = Image.fromarray(arr, mode="L").convert("RGB")

        # 背景稍微调白一点以提升对比度
        if BG_LIGHTEN > 1e-6:
            base_np = np.array(rgb, dtype=np.float32)
            white = 255.0
            base_np = (1.0 - BG_LIGHTEN) * base_np + BG_LIGHTEN * white
            rgb = Image.fromarray(np.clip(base_np, 0, 255).astype(np.uint8), mode="RGB")
        return rgb


# =======================
# 热力图 → 512×512
# =======================
def to_512_from_any(sim_map: torch.Tensor, tgt: int = TARGET_SIZE) -> torch.Tensor:
    sim = sim_map[None, None, ...]  # (1,1,H,W)
    sim_512 = F.interpolate(sim, size=(tgt, tgt), mode="bilinear", align_corners=False)[0, 0]
    return sim_512


# =======================
# bbox 从原图坐标 → 512×512
# =======================
def scale_boxes_from_original_to_512(
    boxes_xywh: List[Tuple[float, float, float, float]],
    orig_w: int, orig_h: int,
    tgt: int = TARGET_SIZE
) -> List[Tuple[int, int, int, int]]:
    sx = tgt / float(orig_w)
    sy = tgt / float(orig_h)
    scaled: List[Tuple[int, int, int, int]] = []
    for (x, y, w, h) in boxes_xywh:
        x1 = int(round(x * sx))
        y1 = int(round(y * sy))
        x2 = int(round((x + w) * sx))
        y2 = int(round((y + h) * sy))
        x1 = max(0, min(tgt, x1)); y1 = max(0, min(tgt, y1))
        x2 = max(0, min(tgt, x2)); y2 = max(0, min(tgt, y2))
        if x2 > x1 and y2 > y1:
            scaled.append((x1, y1, x2, y2))
    return scaled


# =======================
# 叠“红色深度”热力图到 512×512 背景
# - 红色通道= sim * 255
# - 透明度 = HEAT_ALPHA_BASE * sim
# =======================
def overlay_reddepth_on_rgb_512(
    base_512: Image.Image,
    sim_512: torch.Tensor,
    alpha_base: float = HEAT_ALPHA_BASE
) -> Image.Image:
    base = base_512.convert("RGB")
    sim = sim_512.detach().clone()

    # 归一化
    smin, smax = float(sim.min()), float(sim.max())
    if smax - smin > 1e-12:
        sim = (sim - smin) / (smax - smin)
    else:
        sim = torch.zeros_like(sim)

    arr = sim.cpu().numpy().astype(np.float32)  # [0,1], (H,W)

    # 构造 RGBA 热力图：R=arr*255, G=0, B=0, A=alpha_base*arr*255
    R = (arr * 255.0)
    A = (np.clip(alpha_base * arr, 0.0, 1.0) * 255.0)

    heat_rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    heat_rgba[..., 0] = R.astype(np.uint8)
    heat_rgba[..., 1] = 0
    heat_rgba[..., 2] = 0
    heat_rgba[..., 3] = A.astype(np.uint8)

    heat = Image.fromarray(heat_rgba, mode="RGBA")
    out = base.copy().convert("RGBA")
    out.alpha_composite(heat)  # 以像素级透明度叠加
    return out.convert("RGB")


# =======================
# 画框（单描边深绿色）
# =======================
def draw_boxes_512(
    img_512: Image.Image,
    boxes_xyxy_512: List[Tuple[int, int, int, int]],
    color=BOX_COLOR,
    width=LINE_WIDTH
) -> Image.Image:
    im = img_512.copy().convert("RGB")
    draw = ImageDraw.Draw(im)
    for (x1, y1, x2, y2) in boxes_xyxy_512:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return im


# =======================
# 主流程：整套 MS_CXR
# =======================
def main():
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 数据与模型
    ds = MSCXR(transform_name=TRANSFORM_NAME)
    stage1, diffusion, scheduler, tokenizer, text_encoder = get_models.get_modules(
        STAGE1_CFG, STAGE1_WEIGHTS,
        DIFFUSION_CFG, DIFFUSION_WEIGHTS,
        device=device, num_timesteps=NUM_TIMESTEPS
    )
    sampler = Sampler()

    total = len(ds)
    print(f"[INFO] Total samples in MS_CXR: {total}")

    for idx in range(total):
        try:
            img, bbox, original_bbox, prompt, image_id, cls_name = ds[idx]
            img = img.to(device=device, dtype=torch.float32)  # (1,512,512)

            # === 1) 生成热力图 ===
            out = sampler.sampling_fn(
                img[None, ...],
                prompt,
                stage1,
                diffusion,
                scheduler,
                text_encoder,
                tokenizer,
                range(T_MIN, T_MAX),
                guidance_scale=GUIDANCE_SCALE,
                scale_factor=SCALE_FACTOR,
                cls_name=CLS_NAME_FOR_ATTR
            )
            sim_map = out["heatmap"]
            if sim_map.dim() == 3:
                sim_map = sim_map.squeeze(0)

            # === 2) 原图尺寸（bbox 缩放用）===
            with Image.open(image_id) as im_info:
                orig_w, orig_h = im_info.size

            # === 3) 统一到 512×512 ===
            sim_512 = to_512_from_any(sim_map, tgt=TARGET_SIZE)
            bg_512 = make_transformed_background_like_ref(image_id)

            # === 4) bbox → 512×512 ===
            boxes_xyxy_512 = scale_boxes_from_original_to_512(
                original_bbox, orig_w, orig_h, tgt=TARGET_SIZE
            )

            # ====== 导出 A：原图 + 框 ======
            bg_with_boxes = draw_boxes_512(bg_512, boxes_xyxy_512, color=BOX_COLOR, width=LINE_WIDTH)

            # ====== 导出 B：原图 + 红色深度热力图 + 框 ======
            bg_with_heat = overlay_reddepth_on_rgb_512(bg_512, sim_512, alpha_base=HEAT_ALPHA_BASE)
            heat_with_boxes = draw_boxes_512(bg_with_heat, boxes_xyxy_512, color=BOX_COLOR, width=LINE_WIDTH)

            # === 5) 保存 ===
            stem = f"idx{idx}_{Path(image_id).stem}"
            out_boxes = Path(SAVE_DIR) / f"idx{idx}_{stem}_bg512_boxes.png"
            out_heat_boxes = Path(SAVE_DIR) / f"idx{idx}_{stem}_bg512_heatmap_boxes.png"

            bg_with_boxes.save(out_boxes)
            heat_with_boxes.save(out_heat_boxes)

            if idx % 25 == 0:
                print(f"[{idx+1}/{total}] Saved: {out_boxes.name} | {out_heat_boxes.name}")

        except Exception as e:
            print(f"[ERROR] idx={idx} | {e}")

    print(f"[DONE] Saved all images to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
