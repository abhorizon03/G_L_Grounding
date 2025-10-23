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
STAGE1_WEIGHTS: str = "/media/yuganlab/ezstore1/Zongye/Code/Diffusion_Based_Grounding/models/autoencoder.pth"
DIFFUSION_WEIGHTS: str = "/media/yuganlab/ezstore1/Zongye/Code/Diffusion_Based_Grounding/models/diffusion_model.pth"

# 采样窗口
T_MIN, T_MAX = 120, 180  # [T_MIN, T_MAX)
GUIDANCE_SCALE = 0.0
SCALE_FACTOR = 0.3
CLS_NAME_FOR_ATTR = None

# 可视化
SAVE_DIR = "Case_Study/HC_fig1/ldm"
TARGET_SIZE = 512

# 红色深浅热力图（逐像素透明度）
HEAT_ALPHA_BASE = 0.55  # 最终透明度 = HEAT_ALPHA_BASE * sim_norm

# 绿色框
BOX_COLOR = (0, 255, 0)
LINE_WIDTH = 4


# =======================
# 生成 512×512 背景（与你原来的前处理保持一致）
# =======================
def make_transformed_background_like_ref(image_path: str) -> Image.Image:
    """
    - 打开 -> 转 'L' 灰度 -> to_tensor -> resize(512) -> center_crop(512) -> 转回 RGB
    """
    with Image.open(image_path) as im:
        im = im.convert("L")
        im_t = TF.to_tensor(im)  # (1,H,W), [0,1]
        im_t = TF.resize(im_t, size=TARGET_SIZE)  # 双线性
        im_t = TF.center_crop(im_t, output_size=[TARGET_SIZE, TARGET_SIZE])
        arr = (im_t.squeeze(0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(arr, mode="L").convert("RGB")


# =======================
# 热力图 → 512×512（与背景同几何）
# =======================
def to_512_from_any(sim_map: torch.Tensor, tgt: int = TARGET_SIZE) -> torch.Tensor:
    """
    将任意尺寸的单通道热力图张量 (H,W) 插值为 (tgt,tgt)。
    """
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
    """
    线性缩放：x方向乘 sx=tgt/W，y 方向乘 sy=tgt/H，并输出 int 的 xyxy 坐标。
    """
    sx = tgt / float(orig_w)
    sy = tgt / float(orig_h)
    scaled: List[Tuple[int, int, int, int]] = []
    for (x, y, w, h) in boxes_xywh:
        x1 = int(round(x * sx))
        y1 = int(round(y * sy))
        x2 = int(round((x + w) * sx))
        y2 = int(round((y + h) * sy))
        # 裁剪到画布
        x1 = max(0, min(tgt, x1)); y1 = max(0, min(tgt, y1))
        x2 = max(0, min(tgt, x2)); y2 = max(0, min(tgt, y2))
        if x2 > x1 and y2 > y1:
            scaled.append((x1, y1, x2, y2))
    return scaled

def overlay_reddepth_on_rgb_512(
    base_512: Image.Image,
    sim_512: torch.Tensor,
    alpha_base: float = HEAT_ALPHA_BASE
) -> Image.Image:
    base_rgba = base_512.convert("RGBA")

    s = sim_512.detach().clone().to(torch.float32)
    finite = torch.isfinite(s)

    if finite.any():
        v = s[finite]
        smin = float(v.min()); smax = float(v.max())
        rng = smax - smin
        if rng < 1e-12:
            s_norm = torch.zeros_like(s)
        else:
            s_norm = (s - smin) / rng
            s_norm = s_norm.clamp_(0.0, 1.0)
    else:
        s_norm = torch.zeros_like(s)

    s_norm = torch.where(finite, s_norm, torch.zeros_like(s_norm))

    arr = s_norm.cpu().numpy().astype(np.float32)  # (H,W) in [0,1]
    R = (arr * 255.0)
    A = (np.clip(alpha_base * arr, 0.0, 1.0) * 255.0)

    heat_rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    heat_rgba[..., 0] = R.astype(np.uint8)  # R
    heat_rgba[..., 1] = 0                   # G
    heat_rgba[..., 2] = 0                   # B
    heat_rgba[..., 3] = A.astype(np.uint8)  # A

    heat = Image.fromarray(heat_rgba, mode="RGBA")
    out = base_rgba.copy()
    out.alpha_composite(heat)
    return out.convert("RGB")


# =======================
# 画绿色框（单描边）
# =======================
def draw_boxes_green_512(
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
    device = torch.device(f"cuda:%d" % GPU_ID if torch.cuda.is_available() else "cpu")
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

            # === 1) 推理得到热力图（与 ddpm 输入同几何）===
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

            # === 2) 原图尺寸（用于 bbox 比例缩放）===
            with Image.open(image_id) as im_info:
                orig_w, orig_h = im_info.size

            # === 3) 将热力图整幅拉伸到 512×512（与背景一致）===
            sim_512 = to_512_from_any(sim_map, tgt=TARGET_SIZE)

            # === 4) 生成 512×512 背景 ===
            bg_512 = make_transformed_background_like_ref(image_id)

            # === 5) bbox 从原图坐标缩放到 512×512 ===
            boxes_xyxy_512 = scale_boxes_from_original_to_512(
                original_bbox, orig_w, orig_h, tgt=TARGET_SIZE
            )

            # ====== 导出 A：原图 + 绿色框 ======
            img_boxes = draw_boxes_green_512(bg_512, boxes_xyxy_512, color=BOX_COLOR, width=LINE_WIDTH)

            # ====== 导出 B：原图 + 红色深度热力图 + 绿色框 ======
            heat_over = overlay_reddepth_on_rgb_512(bg_512, sim_512, alpha_base=HEAT_ALPHA_BASE)
            heat_over_boxes = draw_boxes_green_512(heat_over, boxes_xyxy_512, color=BOX_COLOR, width=LINE_WIDTH)

            # === 6) 保存 ===
            stem = f"idx{idx}_{Path(image_id).stem}"
            out_path_boxes = Path(SAVE_DIR) / f"{stem}_bg512_boxes.png"
            out_path_heat_boxes = Path(SAVE_DIR) / f"{stem}_heatmap_reddepth_boxes.png"

            img_boxes.save(out_path_boxes)
            heat_over_boxes.save(out_path_heat_boxes)

            if idx % 25 == 0:
                print(f"[{idx+1}/{total}] Saved: {out_path_boxes.name} | {out_path_heat_boxes.name}")

        except Exception as e:
            print(f"[ERROR] idx={idx} | {e}")

    print(f"[DONE] Saved all overlays to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
