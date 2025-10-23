import logging
import argparse
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
import csv                           # <-- CSV 逐样本写入
from PIL import Image                # <-- 读取原图尺寸

# local
from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from health_multimodal.vlp import ImageTextInferenceEngine
from data import MSCXR
from metrics import *   # 需包含: CNR, mIoU, AUC_ROC, Dice

parser = argparse.ArgumentParser(description="Evaluate BioVIL(-T) on phrase grounding")
parser.add_argument("--model-name", type=str, default="biovil", choices=["biovil", "biovil_t"])
parser.add_argument("--gpu-id", type=int, default=1)
parser.add_argument("--log-steps", type=int, default=50, help="Number of steps for logging calls")
args = parser.parse_args()

OUTPUT_DIR = Path("/media/yuganlab/ezstore1/Zongye/Code/Diffusion_Based_Grounding/output_metric/biovil")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# logger
logging.basicConfig(
    filename="output_metric/biovil/metric.log",
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode="a",
    force=True
)

# === per-sample CSV（与日志同目录；自动建表头） ===
csv_path = Path("output_metric/biovil/per_sample_metrics.csv")
if not csv_path.exists():
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "idx", "image_id", "class", "img_w", "img_h",
            "CNR", "CNR_nonabs", "mIoU", "Dice", "AUC_ROC"
        ])

device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

# models
logging.info(f"Model: {args.model_name}")
if args.model_name == "biovil":
    bert_encoder_type = BertEncoderType.CXR_BERT
    image_model_type = ImageModelType.BIOVIL
    crop_size = 480
else:
    bert_encoder_type = BertEncoderType.BIOVIL_T_BERT
    image_model_type = ImageModelType.BIOVIL_T
    crop_size = 448

text_inference = get_bert_inference(bert_encoder_type)
image_inference = get_image_inference(image_model_type)
image_text_inference = ImageTextInferenceEngine(
    image_inference_engine=image_inference,
    text_inference_engine=text_inference,
)
image_text_inference.to(device)

# data
ds = MSCXR(transform_name="biovil_t", crop_size=crop_size)

# metrics containers
auc_roc = AUC_ROC()
class_names = ds.get_class_names()
cnrs        = {k: [] for k in class_names}
mious       = {k: [] for k in class_names}
dices       = {k: [] for k in class_names}   # <-- NEW: Dice
aucrocs     = {k: [] for k in class_names}
nonabs_cnrs = {k: [] for k in class_names}

for idx in range(len(ds)):
    img, bbox, original_bbox, prompt, image_id, cls_name = ds[idx]

    if len(bbox) < 1:
        print(f"Skipped sample {idx}", flush=True)
        continue

    # similarity map
    sim_map = image_text_inference.get_similarity_map_from_raw_data(Path(image_id), prompt)
    sim_map = torch.from_numpy(sim_map)
    sim_map = sim_map.clamp(min=0)  # 经验：不做 (sim_map+1)/2

    # === metrics ===
    cnr         = CNR(sim_map, original_bbox)
    cnr_nonabs  = CNR(sim_map, original_bbox, non_absolute=True)
    miou        = mIoU(sim_map, original_bbox)
    dice_val    = Dice(sim_map, original_bbox)       # <-- NEW
    aucroc      = auc_roc(sim_map, original_bbox)

    cnrs[cls_name].append(cnr)
    mious[cls_name].append(miou)
    dices[cls_name].append(dice_val)                 # <-- NEW
    aucrocs[cls_name].append(aucroc)
    nonabs_cnrs[cls_name].append(cnr_nonabs)

    # === per-sample 保存（逐项写入 CSV） ===
    try:
        w, h = Image.open(Path(image_id)).size
    except Exception:
        w, h = None, None
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            idx, str(image_id), str(cls_name), w, h,
            float(cnr), float(cnr_nonabs), float(miou), float(dice_val), float(aucroc)
        ])

    # logging（加入 Dice 的整体均值日志）
    if idx % args.log_steps == 0:
        logging.info(
            f"After {idx + 1} samples - MS-CXR results (len={len(ds)})\n"
            f"CNR: {[(k, np.mean(v)) for k, v in cnrs.items() if len(v) > 0]}\n"
            f"mIoU: {[(k, np.mean(v)) for k, v in mious.items() if len(v) > 0]}\n"
            f"Dice: {[(k, np.mean(v)) for k, v in dices.items() if len(v) > 0]}\n"
            f"AUC-ROC: {[(k, np.mean(v)) for k, v in aucrocs.items() if len(v) > 0]}"
        )
        logging.info(f"Avg |CNR|: {np.mean([np.mean(v) for v in cnrs.values() if len(v) > 0]) :.4f}")
        logging.info(f"Avg mIoU: {np.mean([np.mean(v) for v in mious.values() if len(v) > 0]) :.4f}")
        logging.info(f"Avg Dice: {np.mean([np.mean(v) for v in dices.values() if len(v) > 0]) :.4f}")   # <-- NEW
        logging.info(f"Avg AUC-ROC: {np.mean([np.mean(v) for v in aucrocs.values() if len(v) > 0]) :.4f}")
        logging.info(f"Avg CNR: {np.mean([np.mean(v) for v in nonabs_cnrs.values() if len(v) > 0]) :.4f}")

# ===== Final summary（加入 Dice 的整体均值±方差） =====
logging.info(
    f"MS-CXR results (len={len(ds)})\n"
    f"|CNR|: {np.mean([np.mean(v) for v in cnrs.values()]) :.4f} +- {np.std([np.mean(v) for v in cnrs.values()]) :.4f}\n"
    f"mIoU: {np.mean([np.mean(v) for v in mious.values()]) :.4f} +- {np.std([np.mean(v) for v in mious.values()]) :.4f}\n"
    f"Dice: {np.mean([np.mean(v) for v in dices.values()]) :.4f} +- {np.std([np.mean(v) for v in dices.values()]) :.4f}\n"
    f"AUC-ROC: {np.mean([np.mean(v) for v in aucrocs.values()]) :.4f} +- {np.std([np.mean(v) for v in aucrocs.values()]) :.4f}\n"
    f"Avg CNR: {np.mean([np.mean(v) for v in nonabs_cnrs.values() if len(v) > 0]) :.4f} +- {np.std([np.mean(v) for v in nonabs_cnrs.values()]) :.4f}"
)

logging.info("******CNR results******")
for k, v in cnrs.items():
    logging.info(f"{k}: {np.mean(v) :.4f} +- {np.std(v) :.4f}")

logging.info("******mIoU results******")
for k, v in mious.items():
    logging.info(f"{k}: {np.mean(v) :.4f} +- {np.std(v) :.4f}")

logging.info("******Dice results******")  # <-- NEW
for k, v in dices.items():
    logging.info(f"{k}: {np.mean(v) :.4f} +- {np.std(v) :.4f}")

logging.info("******Non-absolute CNR results******")
for k, v in nonabs_cnrs.items():
    logging.info(f"{k}: {np.mean(v) :.4f} +- {np.std(v) :.4f}")

logging.info("******AUC-ROC results******")
for k, v in aucrocs.items():
    logging.info(f"{k}: {np.mean(v) :.4f} +- {np.std(v) :.4f}")
