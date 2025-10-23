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
import re

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

@torch.no_grad()
def encode_sentences_tokenwise(
    sentences: List[List[str]],
    tokenizer,
    text_encoder: nn.Module,
    device: torch.device,
    strip_special_tokens: bool = True,
    strip_punct_tokens: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        tiny = tokenizer(["x"], padding=True, truncation=True, max_length=2, return_tensors="pt")
        tiny = {k: v.to(device) for k, v in tiny.items()}
        tiny_out = text_encoder(**tiny, return_dict=True)
        D = int(tiny_out.last_hidden_state.shape[-1])

        tokens_padded = torch.zeros(B, 0, 0, D, device=device)
        sent_mask     = torch.zeros(B, 0, dtype=torch.long, device=device)
        tok_mask      = torch.zeros(B, 0, 0, dtype=torch.long, device=device)
        return tokens_padded, sent_mask, tok_mask

    Lmax_tok = getattr(tokenizer, "model_max_length", 512)
    tok = tokenizer(
        flat_texts,
        padding=True,
        truncation=True,
        max_length=Lmax_tok,
        return_tensors="pt",
    )
    tok = {k: v.to(device) for k, v in tok.items()}
    input_ids: torch.Tensor = tok["input_ids"]        # [N_all, Lpad]
    attention_mask: torch.Tensor = tok["attention_mask"]  # [N_all, Lpad]

    out = text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    hidden: torch.Tensor = out.last_hidden_state      # [N_all, Lpad, D]
    D = int(hidden.shape[-1])

    # 4) 构造内容 token 掩码：去 pad/特殊符号/（可选）标点
    content_mask = (attention_mask > 0)
    if strip_special_tokens:
        for name in ["bos_token_id", "eos_token_id", "cls_token_id", "sep_token_id", "pad_token_id"]:
            tid = getattr(tokenizer, name, None)
            if tid is not None:
                content_mask &= (input_ids != tid)

    if strip_punct_tokens:
        toks = tokenizer.convert_ids_to_tokens(input_ids.flatten().tolist())
        toks = [t if isinstance(t, str) else str(t) for t in toks]
        punct_re = re.compile(r"^\W+$", flags=re.UNICODE)
        punct_mask = torch.tensor(
            [bool(punct_re.match(t)) for t in toks],
            device=device, dtype=torch.bool
        ).view(N_all, -1)
        content_mask &= ~punct_mask

    # 5) 统计 Smax / Lmax（按“有效 token”计数）
    per_b_sent_counts = [0] * B
    per_sent_lengths: List[int] = []
    for i in range(N_all):
        per_b_sent_counts[owner_idx[i]] += 1
        per_sent_lengths.append(int(content_mask[i].sum().item()))

    Smax = max(per_b_sent_counts) if B > 0 else 0
    Lmax = max(per_sent_lengths) if N_all > 0 else 0

    # 6) 分配输出张量并填充
    tokens_padded = torch.zeros(B, Smax, Lmax, D, device=device)
    sent_mask     = torch.zeros(B, Smax, dtype=torch.long, device=device)
    tok_mask      = torch.zeros(B, Smax, Lmax, dtype=torch.long, device=device)

    cursors = [0] * B  # 逐样本句子写入位置
    for i in range(N_all):
        bi = owner_idx[i]
        si = cursors[bi]
        if si >= Smax:
            continue

        valid_idx = content_mask[i].nonzero(as_tuple=False).squeeze(-1)  # [Li]
        Li = int(valid_idx.numel())
        if Li > 0:
            h_i = hidden[i, valid_idx, :]             # [Li, D]
            tokens_padded[bi, si, :Li, :] = h_i
            tok_mask[bi, si, :Li] = 1

        sent_mask[bi, si] = 1
        cursors[bi] += 1

    return tokens_padded, sent_mask, tok_mask


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
    def __init__(
        self,
        target_indices: Optional[List[int]] = None,
        keep_per_head: bool = False,
        detach_for_contrast: bool = False,
        l2norm_for_contrast: bool = True,
        replicate_q_per_sentence: bool = False,
        capture_v_for_override: bool = False,  # 留作扩展；默认不记录 V
    ):
        self.target_indices = None if target_indices is None else set(int(i) for i in target_indices)
        self.keep_per_head = keep_per_head
        self.detach_for_contrast = detach_for_contrast
        self.l2norm_for_contrast = l2norm_for_contrast
        self.replicate_q_per_sentence = replicate_q_per_sentence
        self.capture_v_for_override = capture_v_for_override

        self._orig_forwards: Dict[nn.Module, Any] = {}
        self._index_map: Dict[nn.Module, int] = {}
        self._patched = False
        self.captures: List[Dict[str, Any]] = []

        self._k_tokens_padded: Optional[torch.Tensor] = None   # [B, Smax, Lmax, D]
        self._k_sent_mask: Optional[torch.Tensor] = None       # [B, Smax]
        self._k_tok_mask: Optional[torch.Tensor] = None        # [B, Smax, Lmax]

        self._kv_override: Optional[torch.Tensor] = None       # [B, S, D]
        self._kv_override_mask: Optional[torch.Tensor] = None  # [B, S]

    # === 供外部设置：推荐使用 tokenwise 输入 ===
    def set_k_source_tokens(
        self,
        tokens_padded: torch.Tensor,   # [B, Smax, Lmax, D]
        sent_mask: torch.Tensor,       # [B, Smax]
        tok_mask: torch.Tensor,        # [B, Smax, Lmax]
    ):
        self._k_tokens_padded = tokens_padded
        self._k_sent_mask = sent_mask
        self._k_tok_mask = tok_mask

    # === 兼容旧接口：句向量（会在内部升维到 Lmax=1）===
    def set_kv_source(self, kv_tokens: Optional[torch.Tensor], kv_mask: Optional[torch.Tensor] = None):
        self._kv_override = kv_tokens
        self._kv_override_mask = kv_mask

    # === 工具函数 ===
    @staticmethod
    def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)

    @staticmethod
    def _apply_mask_lastdim(t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # t: [..., D], mask: [...]（与 t 对齐到最后一维前）
        return t * mask.unsqueeze(-1).to(t.dtype)

    def attach(self, unet: nn.Module):
        if self._patched:
            return

        cross = tag_cross_attention_layers(unet)  # 仅 attn2 层
        for idx, ca in enumerate(cross):
            self._index_map[ca] = idx
            orig_forward = ca.forward
            self._orig_forwards[ca] = orig_forward

            def make_wrapped(module: nn.Module, module_idx: int):
                def wrapped_forward(_self, *f_args, **f_kwargs):
                    # ---- 参数解析 ----
                    if len(f_args) == 0:
                        raise RuntimeError("CrossAttention wrapped_forward expects at least 1 positional arg for `x`.")
                    x = f_args[0]
                    context = f_kwargs.get("context", None)
                    if context is None and len(f_args) >= 2:
                        context = f_args[1]

                    # ---- 真实前向：保持不变 ----
                    q_lin = module.to_q(x)  # [B, Tq, H*Dh]
                    ctx_report = context if context is not None else x
                    k_lin_report = module.to_k(ctx_report)
                    v_lin_report = module.to_v(ctx_report)

                    # ---- 记录路径：仅 cross-attn 且命中目标层 ----
                    if getattr(module, "_is_cross_attn", False):
                        if (self.target_indices is None) or (module_idx in self.target_indices):
                            rec: Dict[str, Any] = {
                                "idx": module_idx,
                                "name": getattr(module, "_block_name", f"attn2_{module_idx}"),
                                "kv_src": None,
                            }

                            # === Q：用真实 q_lin；如需句维复制则复制（值相同） ===
                            q_cap = q_lin
                            if self.replicate_q_per_sentence and (self._k_sent_mask is not None):
                                Smax = int(self._k_sent_mask.shape[1])
                                q_cap = q_cap.unsqueeze(1).expand(-1, Smax, -1, -1)  # [B, Smax, Tq, H*Dh]

                            # === K：优先 tokenwise 三件套；否则退回句向量（升为 Lmax=1） ===
                            k_cap = None
                            v_cap = None

                            if (self._k_tokens_padded is not None) and (self._k_tok_mask is not None):
                                B, Smax, Lmax, D = self._k_tokens_padded.shape
                                x_k = self._apply_mask_lastdim(self._k_tokens_padded, self._k_tok_mask)  # [B,S,L,D]
                                x_k_bs = x_k.view(B * Smax, Lmax, D)
                                k_bs = module.to_k(x_k_bs)  # [B*S, Lmax, H*Dh]
                                k_cap = k_bs.view(B, Smax, Lmax, -1)  # [B,S,L,H*Dh]
                                k_cap = self._apply_mask_lastdim(k_cap, self._k_tok_mask)

                                if self.capture_v_for_override:
                                    v_bs = module.to_v(x_k_bs)
                                    v_cap = v_bs.view(B, Smax, Lmax, -1)
                                    v_cap = self._apply_mask_lastdim(v_cap, self._k_tok_mask)

                                rec["kv_src"] = "tokenwise"
                                rec["sent_mask"] = self._k_sent_mask
                                rec["tok_mask"] = self._k_tok_mask

                            elif self._kv_override is not None:
                                # 兼容旧：句向量 [B,S,D] -> 升到 [B,S,1,D]，mask -> [B,S,1]
                                kv_sents = self._kv_override
                                kv_mask = self._kv_override_mask
                                if kv_mask is None:
                                    kv_mask = torch.ones(kv_sents.shape[:2], dtype=torch.long, device=kv_sents.device)
                                kv_sents = kv_sents.unsqueeze(2)        # [B,S,1,D]
                                kv_mask = kv_mask.unsqueeze(2)          # [B,S,1]
                                B, Smax, Lmax, D = kv_sents.shape       # Lmax=1
                                x_k = self._apply_mask_lastdim(kv_sents, kv_mask)  # 先清零
                                x_k_bs = x_k.view(B * Smax, Lmax, D)   # [B*S,1,D]
                                k_bs = module.to_k(x_k_bs)
                                k_cap = k_bs.view(B, Smax, Lmax, -1)
                                k_cap = self._apply_mask_lastdim(k_cap, kv_mask)  # 再清零

                                if self.capture_v_for_override:
                                    v_bs = module.to_v(x_k_bs)
                                    v_cap = v_bs.view(B, Smax, Lmax, -1)
                                    v_cap = self._apply_mask_lastdim(v_cap, kv_mask)

                                rec["kv_src"] = "sent_emb"
                                rec["sent_mask"] = kv_mask.squeeze(2)       # [B,S]
                                rec["tok_mask"] = kv_mask                   # [B,S,1]

                            # === Normalize / detach ===
                            if k_cap is not None:
                                if self.l2norm_for_contrast:
                                    # q_cap 可能是 [B,Tq,H*Dh] 或 [B,S,Tq,H*Dh]，均按最后一维归一化
                                    q_cap = self._l2_normalize(q_cap)
                                    k_cap = self._l2_normalize(k_cap)
                                    if v_cap is not None:
                                        v_cap = self._l2_normalize(v_cap)
                                if self.detach_for_contrast:
                                    q_cap = q_cap.detach()
                                    k_cap = k_cap.detach()
                                    if v_cap is not None:
                                        v_cap = v_cap.detach()

                                # === 组装记录 ===
                                rec["q"] = q_cap
                                rec["k_tok"] = k_cap
                                if v_cap is not None:
                                    rec["v"] = v_cap

                                # （可选）导出每个 head 的视图
                                if self.keep_per_head:
                                    H = module.num_heads
                                    Dh = q_lin.shape[-1] // H

                                    def split_heads_q(t: torch.Tensor) -> torch.Tensor:
                                        # [B,T,H*Dh] -> [B*H,T,Dh]
                                        if t.dim() == 3:
                                            b, tlen, _ = t.shape
                                            return (t.reshape(b, tlen, H, Dh)
                                                      .permute(0, 2, 1, 3)
                                                      .reshape(b * H, tlen, Dh))
                                        # [B,S,T,H*Dh] -> [B*H,S,T,Dh]
                                        b, s, tlen, _ = t.shape
                                        return (t.reshape(b, s, tlen, H, Dh)
                                                  .permute(0, 3, 1, 2, 4)
                                                  .reshape(b * H, s, tlen, Dh))

                                    def split_heads_k(t: torch.Tensor) -> torch.Tensor:
                                        # [B,S,L,H*Dh] -> [B*H,S,L,Dh]
                                        b, s, l, _ = t.shape
                                        return (t.reshape(b, s, l, H, Dh)
                                                  .permute(0, 3, 1, 2, 4)
                                                  .reshape(b * H, s, l, Dh))

                                    qh = split_heads_q(q_cap)
                                    kh = split_heads_k(k_cap)
                                    if self.l2norm_for_contrast:
                                        qh = self._l2_normalize(qh)
                                        kh = self._l2_normalize(kh)
                                    rec["qh"] = qh
                                    rec["kh"] = kh

                                self.captures.append(rec)

                    # ---- 真实注意力计算（不变） ----
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
        """恢复原 forward，并清空来源。"""
        if not self._patched:
            return
        for mod, f in self._orig_forwards.items():
            mod.forward = f
        self._orig_forwards.clear()
        self._index_map.clear()
        self._patched = False

        self._k_tokens_padded = None
        self._k_sent_mask = None
        self._k_tok_mask = None
        self._kv_override = None
        self._kv_override_mask = None

    def clear(self):
        """仅清空捕获记录。"""
        self.captures.clear()