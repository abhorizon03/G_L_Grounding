# train.py
import os, sys, time, argparse, json
from typing import Any, Dict, Optional, List

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig

# ----- 项目路径 -----
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
for sub in ("modules", "datasets", "loss"):
    p = os.path.join(ROOT, sub)
    if p not in sys.path:
        sys.path.append(p)

# ----- 项目内模块 -----
from modules.pipeline import GroundedLDMPipeline
from datasets.dataset import mimiccxrDDPM, collate_dynamic, debug_print_batch
from loss.utils import compute_losses
from types import SimpleNamespace



# -----------------------------
# 解析命令行
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser("Train Grounded LDM (dir or individual cfgs)")
    ap.add_argument("--config", type=str, default=None,
                    help="配置目录，包含 dataset.yaml / pipeline.yaml / train.yaml / loss.yaml")
    ap.add_argument("--data-cfg", type=str, default=None, help="(可选) dataset.yaml")
    ap.add_argument("--pipe-cfg", type=str, default=None, help="(可选) pipeline.yaml")
    ap.add_argument("--train-cfg", type=str, default=None, help="(可选) train.yaml")
    ap.add_argument("--loss-cfg", type=str, default=None, help="(可选) loss.yaml")
    return ap.parse_args()


# -----------------------------
# 日志辅助
# -----------------------------
class Logger:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.txt_log = os.path.join(run_dir, "train.log")
        self.jsonl_dir = os.path.join(run_dir, "logs")
        os.makedirs(self.jsonl_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.jsonl_dir, "metrics.jsonl")

    def write_line(self, msg: str):
        print(msg)
        with open(self.txt_log, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def write_metrics(self, payload: Dict[str, Any]):
        # 逐行 JSON 追加，便于后处理
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# -----------------------------
# 构建 Pipeline / Dataloader
# -----------------------------
def _parse_dtype(s: str) -> torch.dtype:
    s = str(s).lower()
    if s in ("float32","fp32","torch.float32"): return torch.float32
    if s in ("float16","fp16","torch.float16"): return torch.float16
    if s in ("bfloat16","bf16","torch.bfloat16"): return torch.bfloat16
    raise ValueError(f"Unknown dtype: {s}")

def build_pipeline(pcfg: DictConfig) -> GroundedLDMPipeline:
    # 转成可变的普通对象（不会再触发 OmegaConf 的类型限制）
    pcfg_obj = SimpleNamespace(**OmegaConf.to_container(pcfg, resolve=True))
    if isinstance(pcfg_obj.DTYPE, str):
        pcfg_obj.DTYPE = _parse_dtype(pcfg_obj.DTYPE)
    pipe = GroundedLDMPipeline(pcfg_obj).to(torch.device(pcfg_obj.DEVICE))
    ...
    return pipe



def build_dataloader(tokenizer, dcfg: DictConfig) -> DataLoader:
    ds = mimiccxrDDPM(dcfg, tokenizer=tokenizer)
    if len(ds) == 0:
        raise SystemExit("No items after filtering; check JSON/CSV/PA view and paths.")
    print(f"[DATA] items={len(ds)}  mode={dcfg.MODE}  crop={dcfg.CROP_SIZE}")
    return DataLoader(
        ds,
        batch_size=int(dcfg.BATCH_SIZE),
        shuffle=bool(dcfg.SHUFFLE),
        num_workers=int(dcfg.NUM_WORKERS),
        collate_fn=collate_dynamic,
        pin_memory=False,
        drop_last=False,
    )


# -----------------------------
# 运行目录命名（包含关键信息）
# -----------------------------
def make_run_dir(base_out: str, tcfg: DictConfig, lcfg: DictConfig) -> str:
    # 从 loss.yaml 读取 simalign 权重与 topk；从 train.yaml 取 LR
    w_simalign = float(lcfg.WEIGHTS.get("simalign", 0.0)) if "WEIGHTS" in lcfg else 0.0
    topk = float(lcfg.SIMALIGN.get("simalign_topk_ratio", 0.0)) if "SIMALIGN" in lcfg else 0.0
    lr = float(tcfg.LR)

    run_name = f"simalign_{w_simalign:g}-topk_{topk:g}-lr_{lr:g}"
    run_dir = os.path.join(str(tcfg.OUT_DIR), run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# -----------------------------
# 保存 checkpoint（按 epoch 或 step）
# -----------------------------
def save_checkpoint(
    pipe: "GroundedLDMPipeline",
    optimizer,
    *,
    epoch: int,
    global_step: int,
    out_dir: str,
    pcfg: DictConfig,
    tcfg: DictConfig,
    dcfg: DictConfig,
    lcfg: DictConfig,
):
    ckpt_root = os.path.join(out_dir, "ckpt")
    os.makedirs(ckpt_root, exist_ok=True)
    tag = f"e{epoch:03d}_s{global_step:06d}"
    ckpt_dir = os.path.join(ckpt_root, tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    def _sd_cpu(sd):
        return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in sd.items()}

    # 1) 分模块参数
    torch.save(_sd_cpu(pipe.unet.state_dict()), os.path.join(ckpt_dir, "unet.pth"))
    torch.save(_sd_cpu(pipe.vae.state_dict()), os.path.join(ckpt_dir, "vae.pth"))
    torch.save(_sd_cpu(pipe.text_encoder.state_dict()), os.path.join(ckpt_dir, "text_encoder.pth"))

    # 2) 优化器
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pth"))

    # 3) 三/四份配置 YAML
    OmegaConf.save(config=pcfg, f=os.path.join(ckpt_dir, "pipeline.yaml"), resolve=True)
    OmegaConf.save(config=tcfg, f=os.path.join(ckpt_dir, "train.yaml"), resolve=True)
    OmegaConf.save(config=dcfg, f=os.path.join(ckpt_dir, "dataset.yaml"), resolve=True)
    OmegaConf.save(config=lcfg, f=os.path.join(ckpt_dir, "loss.yaml"), resolve=True)

    # 4) meta.json
    meta = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(ckpt_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 5) latest 指针
    latest = os.path.join(ckpt_root, "latest")
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            try:
                os.remove(latest)
            except OSError:
                pass
        os.symlink(os.path.relpath(ckpt_dir, ckpt_root), latest)
    except OSError:
        with open(os.path.join(ckpt_root, "latest_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


# -----------------------------
# 单 epoch 训练
# -----------------------------
def train_one_epoch(
    pipe: GroundedLDMPipeline,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    tcfg: DictConfig,
    lcfg: DictConfig,
    logger: Logger,
    device: torch.device,
    epoch: int,
    *,
    global_step_start: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    save_cb=None,
) -> int:
    pipe.train()
    t0 = time.time()

    amp = bool(tcfg.AMP)
    grad_accum = int(tcfg.GRAD_ACCUM_STEPS)
    log_interval = int(tcfg.LOG_INTERVAL)
    save_interval_step = int(tcfg.get("SAVE_INTERVAL_STEP", 0))  # 0=不按 step 存

    global_step = global_step_start

    for bi, batch in enumerate(loader):
        if bi == 0:
            try:
                debug_print_batch(batch, loader.dataset.cfg, bi)
            except Exception:
                pass

        img = batch.get("img", None)
        ids_f = batch["ids_f"]
        attn  = batch["attention_mask_f"]
        sentences = batch.get("sentences", None)

        img = img.to(device=device) if img is not None else None
        ids_f = ids_f.to(device=device)
        attn  = attn.to(device=device)

        with torch.cuda.amp.autocast(enabled=amp):
            out = pipe(
                img=img,
                ids_f=ids_f,
                attention_mask_f=attn,
                sentences=sentences,
                train_diffusion=True,
                timesteps_train=None,
                return_qkv=True,
            )

            # 从 loss.yaml 获取权重/参数
            weights = dict(lcfg.WEIGHTS) if "WEIGHTS" in lcfg else {}
            sim_cfg = dict(lcfg.SIMALIGN) if "SIMALIGN" in lcfg else {}
            total_loss, logs = compute_losses(
                out,
                device=device,
                weights=weights,
                input_img=img,
                perceptual_loss_fn=None,
                discriminator=None,
                adv_loss_fn=None,
                simalign_tau=float(sim_cfg.get("simalign_tau", 0.07)),
                simalign_topk_ratio=float(sim_cfg.get("simalign_topk_ratio", 0.1)),
                simalign_use_layers=sim_cfg.get("simalign_use_layers", None),
                simalign_inputs_are_unit=bool(sim_cfg.get("simalign_inputs_are_unit", False)),
            )

        if scaler is not None and amp:
            scaler.scale(total_loss).backward()
            if ((bi + 1) % grad_accum) == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        else:
            total_loss.backward()
            if ((bi + 1) % grad_accum) == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        # 日志（console + 文件）
        if (global_step % log_interval) == 0:
            took = time.time() - t0
            msg = f"[E{epoch:03d} S{global_step:07d}] loss={float(total_loss.item()):.6f} | {took:.1f}s"
            for k in ("diffusion_mse", "tokalign_loss", "tokalign_topk", "tokalign_loss_layer",
                      "simalign_loss", "simalign_topk", "simalign_loss_layer"):
                if k in logs:
                    try:
                        msg += f"  {k}={float(logs[k].item()):.6f}"
                    except Exception:
                        pass
            logger.write_line(msg)
            logger.write_metrics({
                "epoch": epoch,
                "global_step": global_step,
                "loss_total": float(total_loss.item()),
                **{k: float(v.item()) for k, v in logs.items() if hasattr(v, "item")}
            })
            t0 = time.time()

        # Step 级保存
        if save_interval_step > 0 and (global_step % save_interval_step) == 0 and save_cb is not None:
            save_cb(epoch=epoch, global_step=global_step)

    return global_step


# -----------------------------
# 主流程
# -----------------------------
def main():
    args = parse_args()
    t1 = time.time()

    # 1) 载入 4 份 CFG（支持目录或逐个文件）
    if args.config is not None:
        cfg_dir = args.config
        dcfg = OmegaConf.load(os.path.join(cfg_dir, "dataset.yaml"))
        pcfg = OmegaConf.load(os.path.join(cfg_dir, "pipeline.yaml"))
        tcfg = OmegaConf.load(os.path.join(cfg_dir, "train.yaml"))
        lcfg = OmegaConf.load(os.path.join(cfg_dir, "loss.yaml"))
    else:
        if not (args.data_cfg and args.pipe_cfg and args.train_cfg and args.loss_cfg):
            raise SystemExit("Please provide --config DIR, or all of --data-cfg/--pipe-cfg/--train-cfg/--loss-cfg")
        dcfg = OmegaConf.load(args.data_cfg)
        pcfg = OmegaConf.load(args.pipe_cfg)
        tcfg = OmegaConf.load(args.train_cfg)
        lcfg = OmegaConf.load(args.loss_cfg)

    # 2) 随机种子 & 设备
    torch.manual_seed(int(dcfg.SEED))
    device = torch.device(pcfg.DEVICE)
    print(f"[ENV] device={device}, AMP={bool(tcfg.AMP)}")

    # 3) 运行目录（含 simalign / topk / LR）
    run_dir = make_run_dir(str(tcfg.OUT_DIR), tcfg, lcfg)
    logger = Logger(run_dir)
    logger.write_line(f"[RUN] dir={run_dir}")

    # 4) Pipeline / Tokenizer / Dataloader
    pipe = build_pipeline(pcfg)
    tokenizer = pipe.tokenizer
    loader = build_dataloader(tokenizer, dcfg)

    # 5) Optimizer（只训练 requires_grad 的参数）
    params = [p for p in pipe.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(tcfg.LR), weight_decay=float(tcfg.WD))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(tcfg.AMP))

    # 便捷封装 save_cb，供 step/epoch 复用
    def save_cb(epoch: int, global_step: int):
        save_checkpoint(
            pipe, optimizer,
            epoch=epoch, global_step=global_step,
            out_dir=run_dir, pcfg=pcfg, tcfg=tcfg, dcfg=dcfg, lcfg=lcfg
        )
        logger.write_line(f"[SAVE] e{epoch:03d} s{global_step:07d}")

    # 6) 训练循环
    global_step = 0
    max_epochs = int(tcfg.MAX_EPOCHS)
    save_interval_epoch = int(tcfg.SAVE_INTERVAL_EPOCH)

    for epoch in range(1, max_epochs + 1):
        global_step = train_one_epoch(
            pipe, loader, optimizer, tcfg, lcfg, logger, device, epoch,
            global_step_start=global_step, scaler=scaler, save_cb=save_cb
        )
        if (epoch % save_interval_epoch) == 0:
            save_cb(epoch=epoch, global_step=global_step)

    logger.write_line("[OK] Training finished.")
    elapsed = time.time() - t1
    
    print(f"[TIME] total elapsed: {elapsed} secs")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
