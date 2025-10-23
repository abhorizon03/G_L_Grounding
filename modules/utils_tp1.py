
import functools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import types

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler, DDPMScheduler
from generative.networks.schedulers.ddim import DDIMPredictionType
from transformers import CLIPTokenizer, CLIPTextModel
from generative.networks.nets.diffusion_model_unet import BasicTransformerBlock, CrossAttention
from typing import Dict, Any, Literal, Tuple, List

# ======================
# VAE / UNet
# ======================
def build_vae_from_config_or_weights(
    stage1_config_file: str,
    device: Union[str, torch.device] = "cuda",
    weight_path: Optional[str] = None,
    strict: bool = True,
    eval_mode: bool = True,
) -> AutoencoderKL:
    conf = OmegaConf.load(stage1_config_file)
    vae_kwargs = dict(conf["stage1"]["params"])
    vae = AutoencoderKL(**vae_kwargs).to(device)
    if weight_path:
        state = torch.load(weight_path, map_location=device)
        vae.load_state_dict(state, strict=strict)
        print(f"[VAE] loaded weights from: {weight_path}")
    else:
        print("[VAE] initialized randomly from config.")
    if eval_mode:
        vae.eval()
    return vae


def build_unet_from_config_or_weights(
    diffusion_config_file: str,
    device: Union[str, torch.device] = "cuda",
    weight_path: Optional[str] = None,
    strict: bool = True,
    eval_mode: bool = True,
) -> DiffusionModelUNet:
    conf = OmegaConf.load(diffusion_config_file)
    unet_kwargs = dict(conf["ldm"].get("params", {}))
    unet = DiffusionModelUNet(**unet_kwargs).to(device)
    if weight_path:
        state = torch.load(weight_path, map_location=device)
        unet.load_state_dict(state, strict=strict)
        print(f"[UNet] loaded weights from: {weight_path}")
    else:
        print("[UNet] initialized randomly from config.")
    if eval_mode:
        unet.eval()
    return unet


def build_ddpm_scheduler_from_config(
    diffusion_config_file: str,
    *,
    num_inference_steps: int = None,
    device: Union[str, torch.device] = "cuda",
    clip_sample: bool = False,
    timestep_spacing: str = "trailing",
) -> DDIMScheduler:
    conf = OmegaConf.load(diffusion_config_file)
    sc = conf["ldm"]["scheduler"]
    scheduler = DDPMScheduler(
        num_train_timesteps=sc["num_train_timesteps"],
        beta_start=sc["beta_start"],
        beta_end=sc["beta_end"],
        schedule=sc["schedule"],
        prediction_type=sc["prediction_type"],
        clip_sample=clip_sample,
    )
    scheduler.set_timesteps(num_inference_steps, device)
    return scheduler


# ======================
# Text
# ======================
def load_text_side_from_hf(
    hf_name: str,
    tokenizer_subfolder: str,
    text_encoder_subfolder: str,
    device: Union[str, torch.device] = "cuda",
) -> Tuple[CLIPTokenizer, CLIPTextModel]:
    tok = CLIPTokenizer.from_pretrained(hf_name, subfolder=tokenizer_subfolder)
    txt = CLIPTextModel.from_pretrained(hf_name, subfolder=text_encoder_subfolder).to(device)
    txt.eval()
    return tok, txt

@torch.no_grad()
def encode_report_for_ldm_from_ids(
    ids_f: torch.Tensor,             # [B, L_any]  —— collate 输出
    attention_mask_f: torch.Tensor,  # [B, L_any]
    text_encoder,                    # e.g. CLIPTextModel
    tokenizer,                       # 与 text_encoder 匹配的 tokenizer
) -> torch.Tensor:
    device = next(text_encoder.parameters()).device
    ids_f = ids_f.to(device)
    attention_mask_f = attention_mask_f.to(device)

    Lmax = tokenizer.model_max_length
    if ids_f.shape[1] > Lmax:
        ids_f = ids_f[:, :Lmax]
        attention_mask_f = attention_mask_f[:, :Lmax]

    out = text_encoder(input_ids=ids_f, attention_mask=attention_mask_f, return_dict=True)
    token_hidden = out.last_hidden_state  # [B, Lmax, D]
    return token_hidden

@torch.no_grad()
def encode_sentences_individually(
    sentences: List[List[str]],                
    tokenizer,                                
    text_encoder: nn.Module,                    
    device: torch.device,
    pool: Literal["mean", "cls"] = "mean",
    strip_special_tokens: bool = True,
    strip_punct_tokens: bool = False,          
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = len(sentences)
    flat_texts: List[str] = []
    owner_idx: List[int] = []
    for bi, sents in enumerate(sentences):
        for s in sents:
            s = (s or "").strip()
            if s == "":
                continue
            flat_texts.append(s)
            owner_idx.append(bi)

    N_all = len(flat_texts)
    if N_all == 0:
        return torch.zeros(B, 0, getattr(text_encoder.config, "hidden_size", 768), device=device), \
               torch.zeros(B, 0, dtype=torch.long, device=device)

    Lmax = getattr(tokenizer, "model_max_length", 512)
    tok = tokenizer(
        flat_texts,
        padding=True,
        truncation=True,
        max_length=Lmax,
        return_tensors="pt",
    )
    tok = {k: v.to(device) for k, v in tok.items()}
    input_ids: torch.Tensor = tok["input_ids"]            
    attention_mask: torch.Tensor = tok["attention_mask"]   
    Ls = input_ids.shape[1]

    out = text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    hidden: torch.Tensor = out.last_hidden_state             
    D = hidden.shape[-1]

    content_mask = (attention_mask > 0) 
    if strip_special_tokens:
        for name in ["bos_token_id", "eos_token_id", "cls_token_id", "sep_token_id", "pad_token_id"]:
            tid = getattr(tokenizer, name, None)
            if tid is not None:
                content_mask &= (input_ids != tid)

    if strip_punct_tokens:
        tokens_str = tokenizer.convert_ids_to_tokens(input_ids.flatten().tolist())
        tokens_str = [t if isinstance(t, str) else str(t) for t in tokens_str]
        punct_re = re.compile(r"^\W+$", flags=re.UNICODE)
        punct_mask = torch.tensor(
            [bool(punct_re.match(t)) for t in tokens_str],
            device=device, dtype=torch.bool
        ).view(N_all, Ls)
        content_mask &= ~punct_mask

    content_attn = content_mask.unsqueeze(-1).float()


    if pool == "mean":
        denom = torch.clamp(content_attn.sum(dim=1, keepdim=True), min=1.0)  
        sent_vec = (hidden * content_attn).sum(dim=1) / denom.squeeze(1)     
    elif pool == "cls":
        sent_vec = hidden[:, 0, :]                                            
    else:
        raise ValueError(f"Unknown pool: {pool}")

    sent_counts = [len([s for s in sents if (s or '').strip()]) for sents in sentences]
    Smax = max(sent_counts) if B > 0 else 0

    out_emb  = torch.zeros(B, Smax, D, device=device)
    out_mask = torch.zeros(B, Smax, dtype=torch.long, device=device)

    cursors = [0] * B
    for v, bi in zip(sent_vec, owner_idx):
        si = cursors[bi]
        if si < Smax:
            out_emb[bi, si]  = v
            out_mask[bi, si] = 1
            cursors[bi] += 1

    return out_emb, out_mask

#=======================
#Capture Cross-Attention Q/K/V
#=======================
def tag_cross_attention_layers(unet: nn.Module) -> List[nn.Module]:
    cross_attn_layers = []
    for name, mod in unet.named_modules():
        if isinstance(mod, BasicTransformerBlock):
            if hasattr(mod, "attn1") and isinstance(mod.attn1, CrossAttention):
                mod.attn1._is_cross_attn = False
                mod.attn1._block_name = f"{name}.attn1"
            if hasattr(mod, "attn2") and isinstance(mod.attn2, CrossAttention):
                mod.attn2._is_cross_attn = True
                mod.attn2._block_name = f"{name}.attn2"
                cross_attn_layers.append(mod.attn2)
    return cross_attn_layers


class CrossAttnQKVCapturer:
    def __init__(self, target_indices: Optional[List[int]] = None, keep_per_head: bool = False,
                 detach_for_contrast: bool = False, l2norm_for_contrast: bool = True):
        self.target_indices = None if target_indices is None else set(int(i) for i in target_indices)
        self.keep_per_head = keep_per_head
        self.detach_for_contrast = detach_for_contrast
        self.l2norm_for_contrast = l2norm_for_contrast

        self._orig_forwards: Dict[nn.Module, Any] = {}
        self._index_map: Dict[nn.Module, int] = {}
        self._patched = False
        self.captures: List[Dict[str, Any]] = []

        self._kv_override: Optional[torch.Tensor] = None
        self._kv_override_mask: Optional[torch.Tensor] = None

    def set_kv_source(self, kv_tokens: Optional[torch.Tensor], kv_mask: Optional[torch.Tensor] = None):
        """
        kv_tokens: [B, S_sent, D]  —— 句子聚合后的 tokens（如 mean-pool 得到的句向量）
        kv_mask  : [B, S_sent]     —— 有效句的掩码（1/0），可为 None
        """
        self._kv_override = kv_tokens
        self._kv_override_mask = kv_mask

    def _apply_mask(self, t: torch.Tensor, mask_2d: Optional[torch.Tensor]) -> torch.Tensor:
        # t: [B, S, D], mask_2d: [B, S]
        if mask_2d is None: 
            return t
        return t * mask_2d.unsqueeze(-1).to(t.dtype)

    def _l2_normalize(self, x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)

    def attach(self, unet: nn.Module):
        if self._patched:
            return
        cross = tag_cross_attention_layers(unet)  # 仅 attn2
        for idx, ca in enumerate(cross):
            self._index_map[ca] = idx
            orig_forward = ca.forward
            self._orig_forwards[ca] = orig_forward

            def make_wrapped(module: nn.Module, module_idx: int):
                def wrapped_forward(_self, *f_args, **f_kwargs):
                    # 1) 取 x（第一个位置参数必须是 x）
                    if len(f_args) == 0:
                        raise RuntimeError("CrossAttention wrapped_forward expects at least 1 positional arg for `x`.")
                    x = f_args[0]

                    # 2) 取 context（关键字优先，其次用第二个位置参数兜底）
                    context = f_kwargs.get("context", None)
                    if context is None and len(f_args) >= 2:
                        context = f_args[1]

                    # 3) 真实注意力仍然用 report 的 context（保持原模型行为）
                    q_lin = module.to_q(x)  # [B, Tq, H*Dh]
                    ctx_report = context if context is not None else x
                    k_lin_report = module.to_k(ctx_report)
                    v_lin_report = module.to_v(ctx_report)

                    # 4) 记录用于对比学习的句子版 K/V（不参与真实注意力）
                    if getattr(module, "_is_cross_attn", False):
                        if (self.target_indices is None) or (module_idx in self.target_indices):
                            if self._kv_override is not None:
                                kv_sents = self._apply_mask(self._kv_override, self._kv_override_mask)  # [B, S_sent, D]
                                k_cap = module.to_k(kv_sents)
                                v_cap = module.to_v(kv_sents)
                                q_cap = q_lin

                                if self.l2norm_for_contrast:
                                    q_cap = self._l2_normalize(q_cap)
                                    k_cap = self._l2_normalize(k_cap)
                                    v_cap = self._l2_normalize(v_cap)
                                if self.detach_for_contrast:
                                    q_cap, k_cap, v_cap = q_cap.detach(), k_cap.detach(), v_cap.detach()

                                rec = {
                                    "idx": module_idx,
                                    "name": getattr(module, "_block_name", f"attn2_{module_idx}"),
                                    "q": q_cap, "k": k_cap, "v": v_cap,
                                    "kv_src": "sent_emb",
                                }
                                if self.keep_per_head:
                                    H = module.num_heads
                                    Dh = q_lin.shape[-1] // H
                                    def split_heads(t):
                                        b, tlen, _ = t.shape
                                        return (t.reshape(b, tlen, H, Dh)
                                                .permute(0, 2, 1, 3)
                                                .reshape(b * H, tlen, Dh))
                                    qh = split_heads(q_cap); kh = split_heads(k_cap); vh = split_heads(v_cap)
                                    if self.l2norm_for_contrast:
                                        qh = self._l2_normalize(qh); kh = self._l2_normalize(kh); vh = self._l2_normalize(vh)
                                    rec.update({"qh": qh, "kh": kh, "vh": vh})
                                self.captures.append(rec)

                    query = module.reshape_heads_to_batch_dim(q_lin)
                    key   = module.reshape_heads_to_batch_dim(k_lin_report)
                    value = module.reshape_heads_to_batch_dim(v_lin_report)
                    
                    if module.use_flash_attention:
                        x_out = module._memory_efficient_attention_xformers(query, key, value)
                    else:
                        x_out = module._attention(query, key, value)

                    x_out = module.reshape_batch_dim_to_heads(x_out).to(query.dtype)
                    return module.to_out(x_out)
                return wrapped_forward

            ca.forward = types.MethodType(make_wrapped(ca, idx), ca)

        self._patched = True

    def detach(self):
        if not self._patched:
            return
        for mod, f in self._orig_forwards.items():
            mod.forward = f
        self._orig_forwards.clear()
        self._index_map.clear()
        self._patched = False
        self._kv_override = None
        self._kv_override_mask = None

    def clear(self):
        self.captures.clear()
