import logging
import argparse
import os
os.environ["TQDM_DISABLE"] = "1"
from pathlib import Path
from math import floor, ceil
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
# local
from data import MSCXR
from metrics import *   # 需包含 Dice(sim_map, gt_boxes, thresholds=None, num_bins=5)
from sampler import Sampler
from models_local import get_models


parser = argparse.ArgumentParser(description="Evaluate LDM on phrase grounding")
parser.add_argument("--num-timesteps", type=int, default=300, help="Number of timesteps for DDIM inversion")
parser.add_argument("--gpu-id", type=int, default=1)
parser.add_argument("--log-steps", type=int, default=50, help="Number of steps for logging calls")
args = parser.parse_args()

# logger
logging.basicConfig(
    filename="output_simalign_0.001-topk_0.1-lr_1e-05_456_WD.log",  # TODO: change fname if necessary
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode="a",
    force=True
)

device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
# data
ds = MSCXR(transform_name="ddpm")
# models
stage1_config_file_path = '/media/yuganlab/ezstore1/Zongye/Code/Diffusion_Based_Grounding/configs/stage1/aekl_v0.yaml'
diffusion_config_file_path = '/media/yuganlab/ezstore1/Zongye/Code/Diffusion_Based_Grounding/configs/ldm/ldm_v0.yaml'
stage1_path = '/media/yuganlab/ezstore1/Zongye/Code/G_L_Grounding/output_runs/simalign_0.001-topk_0.1-lr_1e-05_456/ckpt/e001_s009859/vae.pth'
diffusion_path = '/media/yuganlab/ezstore1/Zongye/Code/G_L_Grounding/output_runs/simalign_0.001-topk_0.1-lr_1e-05_456/ckpt/e001_s009859/unet.pth'
stage1, diffusion, scheduler, tokenizer, text_encoder = get_models.get_modules(
    stage1_config_file_path, stage1_path, diffusion_config_file_path, diffusion_path,
    device=device, num_timesteps=args.num_timesteps
)
sampler = Sampler()
# heuristic for timestep selection
t_min, t_max = args.num_timesteps // 2 - 10 * args.num_timesteps // 100, args.num_timesteps // 2 + 10 * args.num_timesteps // 100

# metrics containers
auc_roc = AUC_ROC()
class_names = ds.get_class_names()
cnrs        = {k: [] for k in class_names}
mious       = {k: [] for k in class_names}
aucrocs     = {k: [] for k in class_names}
nonabs_cnrs = {k: [] for k in class_names}
dices       = {k: [] for k in class_names}   # <<< NEW: Dice

for idx in range(len(ds)):
    if idx % 10 == 0:
        print(f"Processing sample {idx}/{len(ds)}", flush=True)
    img, bbox, original_bbox, prompt, image_id, cls_name = ds[idx]
    img = img.to(device=device, dtype=torch.float32)

    if len(bbox) < 1:
        print(f"Skipped sample {idx}", flush=True)
        continue

    output_dict = sampler.sampling_fn(
        img.to(device=device, dtype=torch.float32)[None, ...],
        prompt,
        stage1, 
        diffusion, 
        scheduler, 
        text_encoder,
        tokenizer,
        range(t_min, t_max),
        guidance_scale = 0,
        scale_factor = 0.3,
        cls_name = None  # None or cls_name (for attribution experiment)
    )
    sim_map = output_dict["heatmap"]

    # to original image dimensions
    w, h = Image.open(Path(image_id)).size
    smallest_dim = min(w, h)
    target_size = smallest_dim, smallest_dim
    sim_map = F.interpolate(
        sim_map[None, None, ...],
        size=target_size,
        mode="nearest",
        align_corners=None,
    )[0, 0]
    margin_w, margin_h = (w - target_size[0]), (h - target_size[1])
    margins_for_pad = (floor(margin_w / 2), ceil(margin_w / 2), floor(margin_h / 2), ceil(margin_h / 2))
    sim_map = F.pad(sim_map, margins_for_pad, value=float("NaN"))

    # metrics
    cnr        = CNR(sim_map, original_bbox)
    cnr_nonabs = CNR(sim_map, original_bbox, non_absolute=True)
    miou       = mIoU(sim_map, original_bbox)
    aucroc     = auc_roc(sim_map, original_bbox)
    dice_val   = Dice(sim_map, original_bbox)          # <<< NEW: Dice 统一口径（并集 + 多阈值平均）

    cnrs[cls_name].append(cnr)
    mious[cls_name].append(miou)
    nonabs_cnrs[cls_name].append(cnr_nonabs)
    aucrocs[cls_name].append(aucroc)
    dices[cls_name].append(dice_val)                   # <<< NEW: 记录 Dice

    # logging
    if idx % args.log_steps == 0:
        print("test pass")
        logging.info(
            "After %d samples - MS-CXR results (len=%d)\n"
            "CNR: %s\n"
            "mIoU: %s\n"
            "Dice: %s\n"          # <<< NEW
            "AUC-ROC: %s",
            idx + 1, len(ds),
            [(k, np.mean(v)) for k, v in cnrs.items() if len(v) > 0],
            [(k, np.mean(v)) for k, v in mious.items() if len(v) > 0],
            [(k, np.mean(v)) for k, v in dices.items() if len(v) > 0],  # <<< NEW
            [(k, np.mean(v)) for k, v in aucrocs.items() if len(v) > 0],
        )
        logging.info("Avg |CNR|: %.4f", np.mean([np.mean(v) for v in cnrs.values() if len(v) > 0]))
        logging.info("Avg mIoU: %.4f", np.mean([np.mean(v) for v in mious.values() if len(v) > 0]))
        logging.info("Avg Dice: %.4f", np.mean([np.mean(v) for v in dices.values() if len(v) > 0]))   # <<< NEW
        logging.info("Avg AUC-ROC: %.4f", np.mean([np.mean(v) for v in aucrocs.values() if len(v) > 0]))
        logging.info("Avg CNR: %.4f", np.mean([np.mean(v) for v in nonabs_cnrs.values() if len(v) > 0]))

# final summary
logging.info(
    "MS-CXR results (len=%d)\n"
    "|CNR|: %.4f +- %.4f\n"
    "mIoU: %.4f +- %.4f\n"
    "Dice: %.4f +- %.4f\n"     # <<< NEW
    "AUC-ROC: %.4f +- %.4f\n"
    "Avg CNR: %.4f +- %.4f",
    len(ds),
    np.mean([np.mean(v) for v in cnrs.values()]) , np.std([np.mean(v) for v in cnrs.values()]),
    np.mean([np.mean(v) for v in mious.values()]) , np.std([np.mean(v) for v in mious.values()]),
    np.mean([np.mean(v) for v in dices.values()]) , np.std([np.mean(v) for v in dices.values()]),    # <<< NEW
    np.mean([np.mean(v) for v in aucrocs.values()]) , np.std([np.mean(v) for v in aucrocs.values()]),
    np.mean([np.mean(v) for v in nonabs_cnrs.values() if len(v) > 0]) ,
    np.std([np.mean(v) for v in nonabs_cnrs.values()])
)

logging.info("******CNR results******")
for k, v in cnrs.items():
    logging.info("%s: %.4f +- %.4f", k, np.mean(v), np.std(v))

logging.info("******mIoU results******")
for k, v in mious.items():
    logging.info("%s: %.4f +- %.4f", k, np.mean(v), np.std(v))

logging.info("******Dice results******")  # <<< NEW
for k, v in dices.items():
    logging.info("%s: %.4f +- %.4f", k, np.mean(v), np.std(v))

logging.info("******Non-absolute CNR results******")
for k, v in nonabs_cnrs.items():
    logging.info("%s: %.4f +- %.4f", k, np.mean(v), np.std(v))

logging.info("******AUC-ROC results******")
for k, v in aucrocs.items():
    logging.info("%s: %.4f +- %.4f", k, np.mean(v), np.std(v))
