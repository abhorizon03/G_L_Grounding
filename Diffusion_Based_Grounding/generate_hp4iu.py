# -*- coding: utf-8 -*-
import os
os.environ["TQDM_DISABLE"] = "1"

from pathlib import Path
from typing import Any, Optional, Dict, Union, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from scipy import ndimage
from skimage.filters import threshold_multiotsu

# ==== 你的工程模块 ====
from data import IU_Xray
from models_local import attention, get_models
# ======================

# ========= 配置 =========
GPU_ID              = 1
TRANSFORM_NAME      = "ddpm"
NUM_DIFFUSION_STEPS = 300
WINDOW_SIZE         = 60
GUIDANCE_SCALE      = 0.0
SCALE_FACTOR        = 0.30
CLS_NAME_FOR_ATTR   = None  # 若要仅聚焦 prompt 中某子串可填写（如 "edema"）

STAGE1_CFG_PATH    = "/media/yuganlab/ezstore1/Zongye/Code/Diffusion_Based_Grounding/configs/stage1/aekl_v0.yaml"
DIFFUSION_CFG_PATH = "/media/yuganlab/ezstore1/Zongye/Code/Diffusion_Based_Grounding/configs/ldm/ldm_v0.yaml"
STAGE1_WEIGHTS     = "/media/yuganlab/ezstore1/Zongye/Code/G_L_Grounding/output_runs/simalign_0.1-topk_0.05-lr_1e-05/ckpt/e001_s009859/vae.pth"
DIFFUSION_WEIGHTS  = "/media/yuganlab/ezstore1/Zongye/Code/G_L_Grounding/output_runs/simalign_0.1-topk_0.05-lr_1e-05/ckpt/e001_s009859/unet.pth"

SAVE_ROOT           = "/media/yuganlab/ezstore1/Zongye/Code/Diffusion_Based_Grounding/heatmap_outputs_IU_reddepth_boxes"
MAX_BASENAME_BYTES  = 240  # 单个文件名（basename）最大UTF-8字节数，超过则跳过

# 可视化相关
TARGET_SIZE      = 512
HEAT_ALPHA_BASE  = 0.55        # 最终像素透明度 = HEAT_ALPHA_BASE * sim_norm

# =======================

def make_transformed_background(image_path: str) -> Image.Image:
    """生成与模型输入同几何的 512x512 灰度底图并转 RGB。"""
    with Image.open(image_path) as im:
        im = im.convert("L")
        im_t = TF.to_tensor(im)  # (1,H,W) [0,1]
        im_t = TF.resize(im_t, size=TARGET_SIZE, interpolation=InterpolationMode.BILINEAR, antialias=None)
        im_t = TF.center_crop(im_t, output_size=[TARGET_SIZE, TARGET_SIZE])  # (1,512,512)
        arr = (im_t.squeeze(0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(arr, mode="L").convert("RGB")

def overlay_reddepth_on_rgb_512(
    base_512: Image.Image,
    sim_01: Union[torch.Tensor, np.ndarray],
    alpha_base: float = HEAT_ALPHA_BASE
) -> Image.Image:
    """
    使用“红色深浅”替代 jet：
      - R = sim_norm * 255
      - G = B = 0
      - A = alpha_base * sim_norm * 255
    其中 sim_01 为 [0,1] 且与 512×512 对齐。
    """
    base = base_512.convert("RGBA")

    if isinstance(sim_01, np.ndarray):
        arr = np.asarray(sim_01, dtype=np.float32)
    else:
        s = torch.as_tensor(sim_01).detach().to(torch.float32)
        finite = torch.isfinite(s)
        if finite.any():
            v = s[finite]
            smin = float(v.min()); smax = float(v.max())
            rng = smax - smin if smax > smin else 1e-12
            s_norm = ((s - smin) / rng).clamp_(0.0, 1.0)
        else:
            s_norm = torch.zeros_like(s)
        s_norm = torch.where(finite, s_norm, torch.zeros_like(s_norm))
        arr = s_norm.cpu().numpy().astype(np.float32)

    arr = np.nan_to_num(arr, nan=0.0)
    arr = np.clip(arr, 0.0, 1.0)

    R = (arr * 255.0)
    A = (np.clip(alpha_base * arr, 0.0, 1.0) * 255.0)

    heat_rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    heat_rgba[..., 0] = R.astype(np.uint8)  # R
    heat_rgba[..., 1] = 0                   # G
    heat_rgba[..., 2] = 0                   # B
    heat_rgba[..., 3] = A.astype(np.uint8)  # A

    heat = Image.fromarray(heat_rgba, mode="RGBA")
    out = base.copy()
    out.alpha_composite(heat)
    return out.convert("RGB")

def get_mask_otsu(heatmap: torch.Tensor, n_classes: int = 2) -> torch.Tensor:
    """
    返回与 heatmap 同尺寸的二值掩码（float32）。
    自动处理 NaN、常数图等边界；确保与 heatmap 同 device/dtype。
    """
    x = torch.nan_to_num(heatmap.detach().float(), nan=0.0).clamp(0.0, 1.0)
    x_np = x.cpu().numpy()

    if float(x.max()) == float(x.min()):
        return torch.zeros_like(x, dtype=torch.float32)

    try:
        thr_np = threshold_multiotsu(x_np, classes=n_classes)  # shape=(n_classes-1,)
    except Exception:
        return (x > x.mean()).to(dtype=torch.float32)

    thr_t = torch.as_tensor(thr_np, dtype=x.dtype, device=x.device)

    if thr_t.numel() == 1:
        mask = x > thr_t.item()
    else:
        mask = (x.unsqueeze(-1) > thr_t.view(1, 1, -1)).any(dim=-1)

    return mask.to(dtype=torch.float32)

def get_heatmap_from_attention(
    diffusion: nn.Module,
    token_init_pos: int,
    token_final_pos: int,
    steps: range,
    final_size: int = 512
) -> torch.Tensor:
    """从已收集的 cross-attention 生成空间热力图并上采样到 final_size。"""
    cross_layers = attention.find_cross_attention_layers(diffusion)
    maps_per_layer = []

    for li, layer in enumerate(cross_layers):
        if li not in [3, 4, 6, 7]:
            continue
        attn = torch.stack(layer.attention_scores_list, -1)  # [B, HW, T, Steps]
        B, HW, T, S = attn.shape
        spatial = int(HW ** 0.5)
        attn = attn.reshape(B, spatial, spatial, T, S)
        attn = attn[1]  # conditional 分支（通常 B=2：0=uncond, 1=cond）
        attn = attn[..., token_init_pos:token_final_pos, steps].mean(-2)  # [H,W,|steps|]

        max_ = attn.max(0)[0].max(0)[0]
        min_ = attn.min(0)[0].min(0)[0]
        attn = (attn - min_) / (max_ - min_ + 1e-12)

        attn = attn.permute(2, 0, 1)
        from torchvision.transforms import Resize
        attn = Resize(final_size)(attn)
        maps_per_layer.append(attn)

    attn_all = torch.stack(maps_per_layer)  # [L, |steps|, H, W]
    heat = attn_all.mean([0, 1])            # [H,W]
    heat = torch.tensor(ndimage.gaussian_filter(heat.numpy(), sigma=(2.5, 2.5), order=0))
    return heat

class Sampler:
    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()
    def sampling_fn(
        self,
        image: torch.Tensor,           # (1,1,512,512) float in [0,1]
        prompt: str,
        autoencoder_model: nn.Module,
        diffusion_model: nn.Module,
        scheduler: nn.Module,
        text_encoder: nn.Module,
        tokenizer: Any,
        steps: range,
        guidance_scale: float = 7.0,
        scale_factor: float = 0.3,
        cls_name: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:

        prompts = ['', prompt]
        prompt_embeds = get_models.get_prompt_embeds(prompts, tokenizer, text_encoder).to(
            device=image.device, dtype=torch.float32
        )

        x = autoencoder_model.encode(image)[0] * scale_factor

        attention.modify_cross_attention_layers(diffusion_model)

        for t in scheduler.timesteps.flip(0):
            noise_in = torch.cat([x] * 2)
            model_out = diffusion_model(
                noise_in, timesteps=torch.tensor((t,), device=image.device).long(), context=prompt_embeds
            )
            uncond, text = model_out.chunk(2)
            noise_pred = uncond + guidance_scale * (text - uncond)
            x, _ = scheduler.reversed_step(noise_pred, t, x)

        text_inputs = tokenizer(prompt, max_length=tokenizer.model_max_length,
                                truncation=True, return_tensors="pt")
        token_init_pos, token_final_pos = 1, len(text_inputs.input_ids.squeeze()) - 1

        if cls_name is not None:
            cls_ids = tokenizer(cls_name, max_length=tokenizer.model_max_length,
                                truncation=True, return_tensors="pt").input_ids.squeeze()[1:-1]
            match = torch.nonzero(text_inputs.input_ids.squeeze()[:, None] == cls_ids, as_tuple=True)[0]
            if match.numel() > 0:
                token_init_pos, token_final_pos = match[0].item(), match[-1].item()
                if token_init_pos == token_final_pos:
                    token_final_pos += 1
            else:
                H = image.shape[-1]
                return {"heatmap": torch.full((H, H), torch.nan)}

        heatmap = get_heatmap_from_attention(
            diffusion_model, token_init_pos, token_final_pos, steps, final_size=image.shape[-1]
        )
        heatmap = torch.clamp(heatmap, 0.0, 1.0)
        heatmap = heatmap * get_mask_otsu(heatmap, n_classes=2)

        return {"heatmap": heatmap}

def build_filename_from_imageid_and_prompt(
    image_id: str,
    prompt: str,
    max_basename_bytes: int = MAX_BASENAME_BYTES
) -> Optional[str]:
    """
    根据 image_id 和 prompt 构造文件名。
    若生成的 basename 的 UTF-8 字节长度超过 max_basename_bytes，则返回 None 用于跳过。
    （已移除框绘制，文件名后缀改为 _reddepth.png）
    """
    stem = Path(image_id).stem  # 例：28_IM-1231-1001.dcm
    cleaned_prompt = (prompt or "").replace("/", "_").replace("\\", "_").replace("\n", " ").strip()
    basename = f"{stem}_{cleaned_prompt or 'empty_prompt'}_reddepth.png"
    if len(basename.encode("utf-8")) > max_basename_bytes:
        return None
    return basename

def main():
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    save_root = Path(SAVE_ROOT)
    save_root.mkdir(parents=True, exist_ok=True)

    # 数据集：IU_Xray
    ds = IU_Xray(transform_name=TRANSFORM_NAME)

    # 模型
    stage1, diffusion, scheduler, tokenizer, text_encoder = get_models.get_modules(
        STAGE1_CFG_PATH, STAGE1_WEIGHTS,
        DIFFUSION_CFG_PATH, DIFFUSION_WEIGHTS,
        device=device, num_timesteps=NUM_DIFFUSION_STEPS
    )
    sampler = Sampler()

    for idx in range(124, len(ds)):
        try:
            # original_bbox 不再使用，以下用占位符 _ 接收
            img, bbox, _original_bbox, prompt, image_id, cls_name = ds[idx]
            img = img.to(device=device, dtype=torch.float32)[None, ...]  # (1,1,512,512)

            # 背景（与预处理对齐）
            bg_512 = make_transformed_background(image_id)

            print(f"\n==> [{idx+1}/{len(ds)}]")
            print(f"    image : {image_id}")
            print(f"    prompt: {prompt}")

            # 采样步窗口
            start = 120
            end = min(start + WINDOW_SIZE, NUM_DIFFUSION_STEPS)
            steps_range = range(start, end)

            out = sampler.sampling_fn(
                img,
                prompt,
                stage1,
                diffusion,
                scheduler,
                text_encoder,
                tokenizer,
                steps=steps_range,
                guidance_scale=GUIDANCE_SCALE,
                scale_factor=SCALE_FACTOR,
                cls_name=CLS_NAME_FOR_ATTR
            )

            heat = out["heatmap"]
            if heat.dim() == 3:
                heat = heat.squeeze(0)

            # 统一到 512×512（与背景一致）
            if heat.shape[0] != TARGET_SIZE or heat.shape[1] != TARGET_SIZE:
                from torchvision.transforms import Resize
                heat = Resize([TARGET_SIZE, TARGET_SIZE])(heat.unsqueeze(0)).squeeze(0)

            # 叠加红色深浅热力图（不再画绿色框）
            overlay_img = overlay_reddepth_on_rgb_512(bg_512, heat, alpha_base=HEAT_ALPHA_BASE)

            # 保存；文件名来源于 image_id + 原始 prompt
            filename = build_filename_from_imageid_and_prompt(image_id, prompt)
            if filename is None:
                print("    [SKIP] 文件名过长（基于 UTF-8 字节检测），已跳过该样本。")
                continue

            save_path = save_root / filename
            overlay_img.save(save_path)

            print(f"    saved overlay -> {save_path}")

        except Exception as e:
            print(f"[ERROR] idx={idx} | {e}")

    print("\n[OK] 仅叠加（红色深浅）结果已生成，目录：", SAVE_ROOT)

if __name__ == "__main__":
    main()
