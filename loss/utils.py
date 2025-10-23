# /media/yuganlab/ezstore1/Zongye/Code/G_L_Grounding/loss/utils.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, Callable, List

import math
import torch
import torch.nn.functional as F


# -------------------------------
# Diffusion (UNet) MSE loss —— 未加权
# -------------------------------
def diffusion_loss(
    outputs: Dict[str, Any],
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    comps: Dict[str, torch.Tensor] = {}
    pred = outputs.get("pred", None)
    target = outputs.get("target", None)
    if pred is None or target is None:
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return zero, comps

    loss_mse = F.mse_loss(pred.float(), target.float(), reduction="mean")
    comps["diffusion_mse"] = loss_mse
    return loss_mse, comps


# -------------------------------
# VAE loss（未加权）：L1 + KL +（可选）Perceptual +（可选）Adv
# -------------------------------
def vae_loss(
    outputs: Dict[str, Any],
    *,
    device: torch.device,
    input_img: Optional[torch.Tensor] = None,
    perceptual_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    discriminator: Optional[torch.nn.Module] = None,
    adv_loss_fn: Optional[Callable[..., torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    comps: Dict[str, torch.Tensor] = {}

    recon = outputs.get("target_img", None)
    if recon is None or input_img is None:
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return zero, comps

    # 1) L1
    l1 = F.l1_loss(recon.float(), input_img.float(), reduction="mean")

    # 2) KL
    mu = outputs.get("enc_mu")
    sigma = outputs.get("enc_sigma")
    if (mu is None) or (sigma is None):
        raise KeyError("[vae_loss] need mu/sigma in outputs.")
    sigma = torch.clamp(sigma, min=1e-6)
    sigma2 = sigma.pow(2)
    kl_per_dim = 0.5 * (mu.pow(2) + sigma2 - torch.log(sigma2) - 1.0)
    reduce_dims = tuple(range(1, kl_per_dim.ndim))
    kl = torch.sum(kl_per_dim, dim=reduce_dims).mean()

    # 3) Perceptual（可选）
    if perceptual_loss_fn is not None:
        p_loss = perceptual_loss_fn(recon.float(), input_img.float()).mean()
    else:
        p_loss = torch.zeros((), device=device, dtype=torch.float32)

    # 4) Adv（可选）
    if (discriminator is not None) and (adv_loss_fn is not None):
        logits_fake = discriminator(recon.contiguous().float())
        if isinstance(logits_fake, (list, tuple)):
            logits_fake = logits_fake[-1]
        g_loss = adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False).mean()
    else:
        g_loss = torch.zeros((), device=device, dtype=torch.float32)

    total_unweighted = l1 + kl + p_loss + g_loss
    comps.update({
        "vae_l1": l1,
        "vae_kl": kl,
        "vae_perc": p_loss,
        "vae_g_adv": g_loss,
        "vae_total_unweighted": total_unweighted,
    })
    return total_unweighted, comps


def _single_layer_token_attention_loss(
    *,
    q: torch.Tensor,                 # [B, Tq, D] 或 [B, Smax, Tq, D]
    k_tok: torch.Tensor,             # [B, Smax, Lmax, D]   —— 已在 capturer 中经 to_k
    sent_mask: torch.Tensor,         # [B, Smax]            —— 有效句 1/0
    tok_mask: torch.Tensor,          # [B, Smax, Lmax]      —— 有效 token 1/0
    topk_ratio: float = 0.1,         # 句内按比例选取 top-k queries
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    assert k_tok.dim() == 4 and tok_mask.dim() == 3 and sent_mask.dim() == 2, \
        "k_tok [B,S,L,D], tok_mask [B,S,L], sent_mask [B,S]"
    B, Smax, Lmax, D = k_tok.shape
    assert tok_mask.shape == (B, Smax, Lmax)
    assert sent_mask.shape == (B, Smax)

    if q.dim() == 3:
        Bq, Tq, Dq = q.shape
        assert Bq == B and Dq == D
        q_is_replicated = False
    elif q.dim() == 4:
        Bq, Sq, Tq, Dq = q.shape
        assert Bq == B and Sq == Smax and Dq == D
        q_is_replicated = True
    else:
        raise ValueError(f"Unexpected q shape: {q.shape}")

    inv_sqrt_d = 1.0 / math.sqrt(D)

    batch_loss_sum = torch.zeros((), device=q.device, dtype=torch.float32)
    valid_samples = 0
    all_topk: List[torch.Tensor] = []
    eps = 1e-12

    for b in range(B):
        sent_mask_b = (sent_mask[b] > 0)         # [Smax]
        if not torch.any(sent_mask_b):
            continue

        sent_scores_b: List[torch.Tensor] = []

        s_indices = torch.nonzero(sent_mask_b, as_tuple=False).flatten()
        for s in s_indices.tolist():
            tok_mask_bs = (tok_mask[b, s] > 0)   # [Lmax]
            if not torch.any(tok_mask_bs):
                continue

            K_s = k_tok[b, s][tok_mask_bs]
            Q_s = q[b, s] if q_is_replicated else q[b]

            logits = (Q_s @ K_s.t()) * inv_sqrt_d

            attn_q_given_token = torch.softmax(logits, dim=0)
            
            q_scores = attn_q_given_token.mean(dim=1)
            Kq = max(1, min(Tq, int(float(topk_ratio) * Tq)))
            topk_val, _ = torch.topk(q_scores, k=Kq, dim=0)

            sent_score = topk_val.mean()

            sent_scores_b.append(sent_score)
            all_topk.append(torch.tensor(Kq, device=q.device, dtype=torch.float32))

        if len(sent_scores_b) == 0:
            continue

        sent_scores_b_t = torch.stack(sent_scores_b, dim=0)                 # [S_eff]
        sent_losses_b   = -torch.log(torch.clamp(sent_scores_b_t, min=eps)) # [S_eff]
        loss_b          = sent_losses_b.mean()                               # 标量（该样本句级平均）

        batch_loss_sum = batch_loss_sum + loss_b
        valid_samples += 1

    if valid_samples == 0:
        return torch.zeros((), device=q.device, dtype=torch.float32), {
            "tokalign_valid": torch.tensor(0, device=q.device, dtype=torch.float32)
        }

    loss = batch_loss_sum / float(valid_samples)
    stats = {
        "tokalign_valid": torch.tensor(valid_samples, device=q.device, dtype=torch.float32),
        "tokalign_topk": (torch.stack(all_topk, 0).float().mean() if all_topk
                          else torch.tensor(0.0, device=q.device, dtype=torch.float32)),
        "tokalign_loss_layer": loss,
    }
    return loss, stats

def token_attention_alignment_loss(
    qkv_list: List[Dict[str, torch.Tensor]],
    *,
    use_layers: Optional[List[int]] = None,
    topk_ratio: float = 0.1,
    inputs_are_unit: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    indices = range(len(qkv_list)) if use_layers is None else use_layers
    per_layer_losses = []
    stats_all: Dict[str, List[torch.Tensor]] = {}

    if device is None:
        for rec in qkv_list:
            if "q" in rec:
                device = rec["q"].device
                break
        if device is None:
            device = torch.device("cpu")

    for li in indices:
        rec = qkv_list[li]
        if not all(k in rec for k in ("q", "k_tok", "tok_mask", "sent_mask")):
            continue

        loss_l, stats_l = _single_layer_token_attention_loss(
            q=rec["q"],
            k_tok=rec["k_tok"],
            sent_mask=rec["sent_mask"],
            tok_mask=rec["tok_mask"],
            topk_ratio=topk_ratio,
        )
        per_layer_losses.append(loss_l)
        for kname, kval in stats_l.items():
            stats_all.setdefault(kname, []).append(kval)

    total = torch.stack(per_layer_losses, dim=0).mean() if per_layer_losses else torch.zeros((), device=device)
    log_stats = {k: torch.stack(v, 0).mean() for k, v in stats_all.items()} if stats_all else {}
    log_stats["tokalign_loss"] = total
    return total, log_stats


# -------------------------------
# 总损失组合（保持外部接口不变；simalign_* 参数被保留但不再使用 tau）
# -------------------------------
def compute_losses(
    outputs: Dict[str, Any],
    *,
    device: torch.device,
    weights: Dict[str, float],                       # e.g. {"diffusion":1.0, "vae_l1":..., "simalign":...}
    input_img: Optional[torch.Tensor] = None,
    perceptual_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    discriminator: Optional[torch.nn.Module] = None,
    adv_loss_fn: Optional[Callable[..., torch.Tensor]] = None,
    simalign_tau: float = 0.07,                      # 保留但在本实现中不使用
    simalign_topk_ratio: float = 0.1,
    simalign_use_layers: Optional[List[int]] = None,
    simalign_inputs_are_unit: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    total = torch.zeros((), device=device, dtype=torch.float32)
    comps: Dict[str, torch.Tensor] = {}

    # 1) Diffusion
    diff_unweighted, diff_comps = diffusion_loss(outputs, device=device)
    total = total + weights.get("diffusion", 0.0) * diff_unweighted
    comps.update(diff_comps)
    comps["diffusion_total"] = weights.get("diffusion", 0.0) * diff_unweighted

    # 2) VAE 分量（按权重组合）
    vae_unweighted_total, vae_comps = vae_loss(
        outputs,
        device=device,
        input_img=input_img,
        perceptual_loss_fn=perceptual_loss_fn,
        discriminator=discriminator,
        adv_loss_fn=adv_loss_fn,
    )
    l1   = vae_comps.get("vae_l1",   torch.zeros((), device=device))
    kl   = vae_comps.get("vae_kl",   torch.zeros((), device=device))
    perc = vae_comps.get("vae_perc", torch.zeros((), device=device))
    gadv = vae_comps.get("vae_g_adv",torch.zeros((), device=device))

    l_vae = (
        weights.get("vae_l1", 0.0)  * l1  +
        weights.get("vae_kl", 0.0)  * kl  +
        weights.get("vae_perc", 0.0)* perc+
        weights.get("vae_adv", 0.0) * gadv
    )
    total = total + l_vae
    comps.update(vae_comps)
    comps["vae_total"] = l_vae

    # 3) Token-level 对齐损失（不回退旧实现）
    qkv_list = outputs.get("qkv", None)
    if qkv_list is not None:
        tok_loss, tok_stats = token_attention_alignment_loss(
            qkv_list=qkv_list,
            use_layers=simalign_use_layers,
            topk_ratio=simalign_topk_ratio,
            inputs_are_unit=simalign_inputs_are_unit,
            device=device,
        )
        total = total + weights.get("simalign", 0.0) * tok_loss
        comps.update(tok_stats)
        comps["simalign_total"] = weights.get("simalign", 0.0) * tok_loss

    return total, comps