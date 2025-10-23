from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_tp1 import (
    load_text_side_from_hf,
    build_vae_from_config_or_weights,
    build_unet_from_config_or_weights,
    build_ddpm_scheduler_from_config,
    encode_report_for_ldm_from_ids,
    encode_sentences_individually,
    CrossAttnQKVCapturer,
)

# =========================
# Pipeline
# =========================
class GroundedLDMPipeline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        device = torch.device(cfg.DEVICE)

        # 1) Text side
        self.tokenizer, self.text_encoder = load_text_side_from_hf(
            cfg.TEXT_HF_NAME, cfg.TEXT_TOKENIZER_SUBFOLDER, cfg.TEXT_ENCODER_SUBFOLDER, device=device
        )
        self.text_encoder.to(device=device, dtype=cfg.DTYPE)
        self.text_encoder.train(cfg.TRAIN_TEXT_ENCODER)
        for p in self.text_encoder.parameters():
            p.requires_grad = cfg.TRAIN_TEXT_ENCODER

        # 2) VAE
        self.vae = build_vae_from_config_or_weights(
            stage1_config_file=cfg.STAGE1_CONFIG,
            device=device,
            weight_path=cfg.VAE_WEIGHTS,
            strict=True,
            eval_mode=not cfg.TRAIN_VAE,
        ).to(device=device, dtype=cfg.DTYPE)
        self.vae.train(cfg.TRAIN_VAE)
        for p in self.vae.parameters():
            p.requires_grad = cfg.TRAIN_VAE

        # 3) UNet
        self.unet = build_unet_from_config_or_weights(
            diffusion_config_file=cfg.LDM_CONFIG,
            device=device,
            weight_path=cfg.UNET_WEIGHTS,
            strict=True,
            eval_mode=not cfg.TRAIN_UNET,
        ).to(device=device, dtype=cfg.DTYPE)
        self.unet.train(cfg.TRAIN_UNET)
        for p in self.unet.parameters():
            p.requires_grad = cfg.TRAIN_UNET

        # 4) Scheduler（必须具备 add_noise/get_velocity/step/set_timesteps）
        self.scheduler = build_ddpm_scheduler_from_config(
            diffusion_config_file=cfg.LDM_CONFIG,
            num_inference_steps=cfg.DDIM_STEPS,
            device=device,
            clip_sample=cfg.CLIP_SAMPLE,
            timestep_spacing=cfg.TIMESTEP_SPACING,
        )

        # 5) Cross-Attn capturer（仅在训练时使用 sentences 作为 K/V 源做捕获）
        self.capturer = CrossAttnQKVCapturer(
            target_indices=cfg.CAPTURE_LAYERS,
            keep_per_head=cfg.CAPTURE_PER_HEAD,
            detach_for_contrast=cfg.CONTRAST_DETACH,
            l2norm_for_contrast=cfg.CONTRAST_L2NORM,
        )
        self.capturer.attach(self.unet)
        if cfg.VERBOSE:
            print(f"[Pipeline] Cross-attn hook registered: True")

    # ------------------------------------------------------
    # encode 文本：report -> UNet cross-attn context；sentences 仅训练时用于 QKV 捕获源
    # ------------------------------------------------------
    @torch.no_grad()
    def encode_report_tokens(
        self,
        ids_f: torch.Tensor,                 # [B, L]
        attention_mask_f: torch.Tensor,      # [B, L]
    ) -> torch.Tensor:
        return encode_report_for_ldm_from_ids(
            ids_f=ids_f, attention_mask_f=attention_mask_f,
            text_encoder=self.text_encoder, tokenizer=self.tokenizer
        )  # [B, Lenc<=Lmax, D]

    @torch.no_grad()
    def encode_sentence_embeddings(
        self,
        sentences: List[List[str]],          # List of B lists
        pool: str = "mean",
        strip_special_tokens: bool = True,
        strip_punct_tokens: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.text_encoder.parameters()).device
        return encode_sentences_individually(
            sentences=sentences,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            device=device,
            pool=pool,
            strip_special_tokens=strip_special_tokens,
            strip_punct_tokens=strip_punct_tokens,
        )  # (sent_emb:[B,S,D], sent_mask:[B,S])

    # ------------------------------------------------------
    # VAE encode/decode
    # ------------------------------------------------------
    def encode_image_to_latents(self, img: torch.Tensor) -> torch.Tensor:
        z_mu, z_sigma = self.vae.encode(img)          # encode 输出均值，按训练约定缩放
        lat = z_mu * self.cfg.VAE_SCALING
        return lat, z_mu, z_sigma

    def decode_latents_to_image(self, latents: torch.Tensor) -> torch.Tensor:
        z = latents / self.cfg.VAE_SCALING
        img_dec = self.vae.decode_stage_2_outputs(z)
        return img_dec.clamp(0, 1)

    # ------------------------------------------------------
    # forward
    # ------------------------------------------------------
    def forward(
        self,
        img: torch.Tensor,                       # [B, C, H, W] in [0,1]
        ids_f: torch.Tensor,                     # [B, L]
        attention_mask_f: torch.Tensor,          # [B, L]
        sentences: Optional[List[List[str]]] = None,  # 仅训练时用于 QKV 捕获；推理不需要
        *,
        # ------- 推理 -------
        num_steps: Optional[int] = None,
        strength: float = 0.999,                 # img2img 起点（0=不加噪，~1=接近纯噪）
        return_qkv: bool = False,                # 推理默认不抓 QKV

        # ------- 训练 -------
        train_diffusion: bool = False,
        timesteps_train: Optional[torch.Tensor] = None,  # Long[B]，可外部指定
    ) -> Dict[str, Any]:

        device = next(self.parameters()).device
        dtype  = self.cfg.DTYPE
        B = img.shape[0]

        # 1) 报告文本 -> cross-attn context
        if self.training and self.cfg.TRAIN_TEXT_ENCODER:
            token_hidden = encode_report_for_ldm_from_ids(
                ids_f=ids_f.to(device), attention_mask_f=attention_mask_f.to(device),
                text_encoder=self.text_encoder, tokenizer=self.tokenizer
            )
        else:
            with torch.no_grad():
                token_hidden = self.encode_report_tokens(ids_f.to(device), attention_mask_f.to(device))

        # 2) sentences 仅训练时用于 QKV 捕获
        sent_emb, sent_mask = None, None
        if train_diffusion and return_qkv and sentences is not None:
            with torch.set_grad_enabled(self.training and not self.cfg.CONTRAST_DETACH):
                sent_emb, sent_mask = self.encode_sentence_embeddings(
                    sentences, pool="mean", strip_special_tokens=True, strip_punct_tokens=False
                )

        # 3) 图像 -> 干净 latent (x0)
        latents_clean, z_mu, z_sigma = self.encode_image_to_latents(img.to(device=device, dtype=dtype))  # [B, C_lat, h, w]

        scheduler = self.scheduler

        # =============== 训练分支（严格对齐“对方写法”） ===============
        if train_diffusion:
            if timesteps_train is None:
                nT = int(scheduler.num_train_timesteps)  # 假定一定存在
                timesteps_train = torch.randint(low=0, high=nT, size=(B,),
                                                device=device, dtype=torch.long)
            else:
                timesteps_train = timesteps_train.to(device=device, dtype=torch.long)

            noise = torch.randn_like(latents_clean)
            xt = scheduler.add_noise(original_samples=latents_clean, noise=noise, timesteps=timesteps_train)

            pred_type = scheduler.prediction_type
            if pred_type == "v_prediction":
                target = scheduler.get_velocity(latents_clean, noise, timesteps_train)
            elif pred_type == "epsilon":
                target = noise
            else:
                raise ValueError(f"Unsupported prediction_type for strict alignment: {pred_type}")

            if return_qkv and (sent_emb is not None) and (sent_mask is not None):
                self.capturer.clear()
                self.capturer.set_kv_source(kv_tokens=sent_emb, kv_mask=sent_mask)

            pred = self.unet(x=xt, timesteps=timesteps_train, context=token_hidden)

            target_img = self.decode_latents_to_image(latents_clean)  # [B,C,H,W] in [0,1]

            out: Dict[str, Any] = {
                "pred": pred,                          
                "target": target,                      
                "timesteps_train": timesteps_train,   
                "noisy_latents": xt,                
                "latents_clean": latents_clean,        
                "target_img": target_img,
                "enc_mu": z_mu,
                "enc_sigma": z_sigma,   
            }
            if return_qkv and (sent_emb is not None) and (sent_mask is not None):
                out["qkv"] = self.capturer.captures
                out["sent_mask"] = sent_mask
            return out

        # =============== 推理分支（img2img） ===============
        if (num_steps is not None) and (num_steps != getattr(scheduler, "num_inference_steps", None)):
            scheduler.set_timesteps(num_steps)

        timesteps = scheduler.timesteps
        T = len(timesteps)

        # strength: 0→接近0步(噪声小)，1→接近T-1(噪声大)。降序时间轴下更直观的映射：
        s = float(max(0.0, min(0.999, strength)))
        t_index = int((1.0 - s) * (T - 1))                 
        t_start = timesteps[t_index]                        

        noise = torch.randn_like(latents_clean)
        x = scheduler.add_noise(original_samples=latents_clean, noise=noise, timesteps=t_start)

        idxs = (timesteps == t_start).nonzero(as_tuple=False)
        start_i = int(idxs[0].item()) if idxs.numel() > 0 else t_index

        x_t = x
        for t in timesteps[start_i:]:                      
            t_scalar = int(t.item()) if torch.is_tensor(t) else int(t)
            t_batch = torch.full((B,), t_scalar, device=device, dtype=torch.long)

            model_out = self.unet(x=x_t, timesteps=t_batch, context=token_hidden)

            step_out = scheduler.step(model_out, t_scalar, x_t)
            if isinstance(step_out, dict):
                x_t = step_out.get("prev_sample", step_out.get("sample", x_t))
            elif hasattr(step_out, "prev_sample"):
                x_t = step_out.prev_sample
            elif isinstance(step_out, (tuple, list)) and len(step_out) >= 1:
                x_t = step_out[0]
            else:
                x_t = step_out

        # 解码到像素空间
        decoded_img = self.decode_latents_to_image(x_t)     # [B,C,H,W] in [0,1]

        out: Dict[str, Any] = {
            "latents_clean":       latents_clean,
            "latents_noisy_start": x,
            "latents_final":       x_t,
            "decoded_img":         decoded_img,
        }
        return out