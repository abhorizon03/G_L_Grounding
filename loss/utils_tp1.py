# /media/yuganlab/ezstore1/Zongye/Code/G_L_Grounding/loss/utils.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, Callable, List

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
#   是否参与由外部权重决定；此处不读 cfg、不乘权
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

    # 未加权总和（仅用于观测；真正加权在外部）
    total_unweighted = l1 + kl + p_loss + g_loss
    comps.update({
        "vae_l1": l1,
        "vae_kl": kl,
        "vae_perc": p_loss,
        "vae_g_adv": g_loss,
        "vae_total_unweighted": total_unweighted,
    })
    return total_unweighted, comps


# -------------------------------
# 单层对齐损失（未加权）：句内均值 → batch 均值
# -------------------------------
def _single_layer_sim_align_loss(
    q: torch.Tensor,          # [B, Nq, D]
    k: torch.Tensor,          # [B, S, D]
    sent_mask: torch.Tensor,  # [B, S] (bool/0-1)
    topk_ratio: float = 0.1,
    tau: float = 0.07,
    inputs_are_unit: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    assert q.dim() == 3 and k.dim() == 3
    B, Nq, D = q.shape
    assert sent_mask.shape[0] == B

    K = max(1, min(Nq, int(float(topk_ratio) * Nq)))

    if inputs_are_unit:
        q_src, k_src = q, k
    else:
        q_src, k_src = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

    batch_loss_sum = 0.0
    valid_samples = 0

    for b in range(B):
        mask_b = (sent_mask[b] > 0) if sent_mask.dtype != torch.bool else sent_mask[b]
        S_eff = int(mask_b.sum().item())
        if S_eff == 0:
            continue

        q_b = q_src[b]              # [Nq, D]
        k_b = k_src[b][mask_b]      # [S', D]
        sim = q_b @ k_b.t()         # [Nq, S']  (cosine if normalized)

        _, topk_idx = torch.topk(sim.t(), k=K, dim=1)                 # [S', K]
        q_b_exp = q_b.unsqueeze(0).expand(S_eff, Nq, D)               # [S', Nq, D]
        topk_idx_exp = topk_idx.unsqueeze(-1).expand(S_eff, K, D)     # [S', K, D]
        topk_q = torch.gather(q_b_exp, dim=1, index=topk_idx_exp)     # [S', K, D]

        f_align = topk_q.mean(dim=1)                                  # [S', D]  GAP
        f_align = F.normalize(f_align, dim=-1)

        logits_vt = (f_align @ k_b.t()) / tau                         # [S', S']
        labels = torch.arange(S_eff, device=q.device)

        loss_rows = F.cross_entropy(logits_vt,     labels, reduction="mean")
        loss_cols = F.cross_entropy(logits_vt.t(), labels, reduction="mean")
        loss_b = 0.5 * (loss_rows + loss_cols)

        batch_loss_sum += loss_b
        valid_samples += 1

    if valid_samples == 0:
        return torch.zeros((), device=q.device, dtype=torch.float32), {
            "simalign_valid": torch.tensor(0, device=q.device, dtype=torch.float32)
        }

    loss = batch_loss_sum / float(valid_samples)
    return loss, {
        "simalign_valid": torch.tensor(valid_samples, device=q.device, dtype=torch.float32),
        "simalign_loss_layer": loss,
        "simalign_topk": torch.tensor(K, device=q.device, dtype=torch.float32),
    }


# -------------------------------
# 多层对齐损失（未加权）
# -------------------------------
def contrastive_alignment_loss(
    qkv_list: List[Dict[str, torch.Tensor]],
    sent_mask: torch.Tensor,              # [B, S]
    *,
    topk_ratio: float = 0.1,
    tau: float = 0.07,
    use_layers: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
    inputs_are_unit: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if device is None:
        device = qkv_list[0]["q"].device
    indices = range(len(qkv_list)) if use_layers is None else use_layers

    per_layer_losses = []
    stats_all: Dict[str, List[torch.Tensor]] = {}

    for li in indices:
        rec = qkv_list[li]
        loss_l, stats_l = _single_layer_sim_align_loss(
            q=rec["q"], k=rec["k"], sent_mask=sent_mask,
            topk_ratio=topk_ratio, tau=tau, inputs_are_unit=inputs_are_unit
        )
        per_layer_losses.append(loss_l)
        for kname, kval in stats_l.items():
            stats_all.setdefault(kname, []).append(kval)

    total = torch.stack(per_layer_losses, dim=0).mean() if per_layer_losses else torch.zeros((), device=device)
    log_stats = {k: torch.stack(v, 0).mean() for k, v in stats_all.items()}
    log_stats["simalign_loss"] = total
    return total, log_stats

def compute_losses(
    outputs: Dict[str, Any],
    *,
    device: torch.device,
    weights: Dict[str, float],                       
    input_img: Optional[torch.Tensor] = None,
    perceptual_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    discriminator: Optional[torch.nn.Module] = None,
    adv_loss_fn: Optional[Callable[..., torch.Tensor]] = None,
    simalign_tau: float = 0.07,
    simalign_topk_ratio: float = 0.1,
    simalign_use_layers: Optional[List[int]] = None,
    simalign_inputs_are_unit: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    total = torch.zeros((), device=device, dtype=torch.float32)
    comps: Dict[str, torch.Tensor] = {}

    diff_unweighted, diff_comps = diffusion_loss(outputs, device=device)
    total = total + weights.get("diffusion", 0.0) * diff_unweighted
    comps.update(diff_comps)
    comps["diffusion_total"] = weights.get("diffusion", 0.0) * diff_unweighted

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

    qkv_list = outputs.get("qkv", None)
    sent_mask = outputs.get("sent_mask", None)
    if (qkv_list is not None) and (sent_mask is not None):
        sim_unweighted, sim_stats = contrastive_alignment_loss(
            qkv_list=qkv_list,
            sent_mask=sent_mask,
            topk_ratio=simalign_topk_ratio,
            tau=simalign_tau,
            use_layers=simalign_use_layers,
            device=device,
            inputs_are_unit=simalign_inputs_are_unit,
        )
        total = total + weights.get("simalign", 0.0) * sim_unweighted
        comps.update(sim_stats)
        comps["simalign_total"] = weights.get("simalign", 0.0) * sim_unweighted

    return total, comps
