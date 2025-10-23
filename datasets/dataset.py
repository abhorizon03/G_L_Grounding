import os, csv, json, random, math, hashlib, re
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader
from monai import transforms
from torchvision.transforms import Resize

def _extract_dicom_id(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def _to_num_or_none(v: Any):
    if v is None: return None
    s = str(v).strip()
    if s == "" or s.lower() in {"na","nan","null"}: return None
    try:
        x = float(s)
        if x != x: return None
        return x
    except: return None

def load_csv1_dicom_view(csv_path: str, dicom_col: str, view_col: str) -> Dict[str, str]:
    out = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            did = str(row.get(dicom_col, "")).strip()
            vp  = str(row.get(view_col, "")).strip()
            if did: out[did] = vp
    print(f"[csv1] dicom→view: {len(out)}")
    return out

def load_csv2_last14(csv_path: str, study_col: str) -> Tuple[Dict[str, List[Optional[float]]], List[str]]:
    out = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if len(cols) < 15: raise ValueError(f"[csv2] 列不足: {len(cols)}")
        tail_cols = cols[-14:]
        for row in reader:
            sid = str(row.get(study_col, "")).strip()
            if not sid: continue
            vals = [_to_num_or_none(row.get(c, "")) for c in tail_cols]
            out[sid] = vals
    print(f"[csv2] study→last14: {len(out)} | tail_cols={tail_cols}")
    return out, tail_cols

def ddpm_image_transforms(crop_size: int = 512):
    return transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"], reader="pilreader"),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.Lambdad(keys=["image"], func=lambda x: x[0, :, :][None, ]),
            transforms.Rotate90d(keys=["image"], k=-1, spatial_axes=(0, 1)),
            transforms.Flipd(keys=["image"], spatial_axis=1),
            transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
            transforms.Lambdad(keys=["image"], func=lambda x: Resize(crop_size, antialias=None)(x)),
            transforms.CenterSpatialCropd(keys=["image"], roi_size=(crop_size, crop_size)),
            transforms.ToTensord(keys=["image"]),
        ],
        map_items=True
    )

def valid_processed_image(img: torch.Tensor, expect_shape) -> Tuple[bool, str]:
    if not isinstance(img, torch.Tensor): return False, "not_tensor"
    if img.shape != expect_shape: return False, f"bad_shape:{tuple(img.shape)}"
    if not torch.is_floating_point(img): return False, f"bad_dtype:{img.dtype}"
    if not torch.isfinite(img).all(): return False, "non_finite"
    vmin, vmax = float(img.min()), float(img.max())
    if vmin < -1e-6 or vmax > 1.0 + 1e-6: return False, f"out_of_range:[{vmin:.4f},{vmax:.4f}]"
    if (vmax - vmin) <= 1e-6: return False, f"near_constant:{vmin:.6f}"
    return True, ""

# ---- patch 与掩蔽 ----
def image_to_patch_sequence(img: torch.Tensor, patch_num: int) -> torch.Tensor:
    _, H, W = img.shape
    assert H == W and H % patch_num == 0
    ps = H // patch_num
    seq = img.unfold(1, ps, ps).unfold(2, ps, ps)  # [1, pn, pn, ps, ps]
    seq = seq.permute(1,2,0,3,4).contiguous().view(patch_num*patch_num, 1, ps, ps)
    return seq

def mae_like_random_masking(L: int, mask_ratio: float, device, g: torch.Generator):
    len_keep = int(L * (1.0 - mask_ratio))
    noise = torch.rand(L, generator=g, device=device)
    ids_shuffle = torch.argsort(noise)
    ids_restore = torch.argsort(ids_shuffle)
    ids_keep = ids_shuffle[:len_keep]
    mask = torch.ones(L, device=device)
    mask[:len_keep] = 0
    mask = mask[ids_restore]
    return mask, ids_restore, ids_keep

def gather_sequence_by_index(seq: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
    return seq.index_select(0, ids_keep)

def patch_sequence_to_mask2d(mask_seq: torch.Tensor, patch_num: int, H: int, W: int) -> torch.Tensor:
    ps = H // patch_num
    mask_grid = mask_seq.view(patch_num, patch_num)
    mask = mask_grid.repeat_interleave(ps, 0).repeat_interleave(ps, 1)
    return mask.unsqueeze(0)

def make_vae_zero_masked(img: torch.Tensor, patch_num: int, mask_ratio: float, g: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    _, H, W = img.shape
    L = patch_num * patch_num
    mask_seq, _, _ = mae_like_random_masking(L, mask_ratio, img.device, g)
    mask2d = patch_sequence_to_mask2d(mask_seq, patch_num, H, W).to(img.dtype)
    img_masked = img * (1.0 - mask2d)
    return img_masked, mask2d

# ---- 文本：分句 + tokenizer 适配（tokenizer 外部传入）----
import re
from typing import List

_SECTION_PREFIX_RE = re.compile(
    r'\b(impression|findings)\s*:\s*',
    re.IGNORECASE
)
_SENT_SPLIT_RE = re.compile(r'[\.。！？!?;；]+\s*|\n+')

_NUMERIC_OR_PUNCT_RE = re.compile(r'^[\d\W_]+$')

def split_into_sentences(report: str) -> List[str]:
    if not report:
        return []

    text = _SECTION_PREFIX_RE.sub(lambda m: m.group(0).strip() + "\n", report)
    parts = _SENT_SPLIT_RE.split(text)
    sents = []
    for s in parts:
        s = re.sub(r'\s+', ' ', s).strip()
        if not s:
            continue
        if re.fullmatch(r'(impression|findings)\s*:?', s, re.IGNORECASE):
            continue
        if _NUMERIC_OR_PUNCT_RE.fullmatch(s):
            continue
        sents.append(s)
    return sents


class TokenizerAdapter:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.has_type_ids = ("token_type_ids" in getattr(self.tok, "model_input_names", []))
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self.cls_id = getattr(self.tok, "cls_token_id", None)
        self.sep_id = getattr(self.tok, "sep_token_id", None)
        self.bos_id = getattr(self.tok, "bos_token_id", None)
        self.eos_id = getattr(self.tok, "eos_token_id", None)

    def encode_no_specials(self, s: str) -> List[int]:
        return self.tok.encode(s, add_special_tokens=False)

    def pack_sentences(self, sentences: List[str], max_len: Optional[int], count_special_in_sent: bool=False):
        ids: List[int] = []
        smask: List[int] = []
        type_ids: List[int] = []

        def add_special(token_id: Optional[int]):
            if token_id is not None:
                ids.append(token_id)
                smask.append(1 if count_special_in_sent else 0)
                if self.has_type_ids: type_ids.append(0)

        add_special(self.cls_id or self.bos_id)
        for j, s in enumerate(sentences, start=1):
            tok = self.encode_no_specials(s)
            if self.has_type_ids: type_ids.extend([0] * len(tok))
            ids.extend(tok)
            smask.extend([j] * len(tok))
            add_special(self.sep_id or self.eos_id)

        if max_len is not None:
            ids = ids[:max_len]
            smask = smask[:max_len]
            if self.has_type_ids: type_ids = type_ids[:max_len]
            attn = [1] * len(ids)
            pad = max_len - len(ids)
            if pad > 0:
                ids   += [self.pad_id] * pad
                smask += [0] * pad
                attn  += [0] * pad
                if self.has_type_ids: type_ids += [0] * pad
        else:
            attn = [1] * len(ids)

        out = {
            "ids_f": torch.tensor(ids, dtype=torch.long),
            "attention_mask_f": torch.tensor(attn, dtype=torch.long),
            "sentence_mask_f": torch.tensor(smask, dtype=torch.long),
        }
        if self.has_type_ids:
            out["type_ids_f"] = torch.tensor(type_ids, dtype=torch.long)
        out["pad_token_id"] = int(self.pad_id)
        return out


# =======================
# Dataset
# =======================
class mimiccxrDDPM(Dataset):
    def __init__(self, cfg: Any, tokenizer):
        self.cfg = cfg
        self.mode = cfg.MODE.lower()
        random.seed(cfg.SEED)
        self.tfs = ddpm_image_transforms(cfg.CROP_SIZE)
        self.tok = TokenizerAdapter(tokenizer)

        dicom_view = load_csv1_dicom_view(cfg.CSV1_PATH, cfg.CSV1_DICOM_COL, cfg.CSV1_VIEW_COL)
        study_last14, tail_cols = load_csv2_last14(cfg.CSV2_PATH, cfg.CSV2_STUDY_COL)
        self.last14_names = tail_cols

        with open(cfg.JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if cfg.SPLIT not in data:
            raise KeyError(f"split='{cfg.SPLIT}' not in {list(data.keys())}")

        items = []
        total_seen = kept = miss_view = miss_study = 0
        for e in data[cfg.SPLIT]:
            report = e.get("report", "") or ""
            sentences = split_into_sentences(report)
            sid = str(e.get("study_id", "")).strip()
            last14 = study_last14.get(sid, None)
            if last14 is None: miss_study += 1
            for rel in e.get("image_path", []):
                abs_path = os.path.join(cfg.IMAGE_PREFIX, rel)
                did = _extract_dicom_id(abs_path)
                vp = dicom_view.get(did, None)
                total_seen += 1
                if vp != cfg.CSV1_KEEP_VIEW:
                    miss_view += 1
                    continue
                items.append((abs_path, report, sentences, sid, last14 if last14 is not None else [None]*14))
                kept += 1

        if cfg.LIMIT_FIRST_N is not None:
            items = items[: int(cfg.LIMIT_FIRST_N)]
        if cfg.SHUFFLE:
            rng = random.Random(cfg.SEED); rng.shuffle(items)

        self.items = items
        self.stats = dict(total_seen=total_seen, kept=kept, miss_view=miss_view, miss_study=miss_study)

    def __len__(self): return len(self.items)

    def _per_sample_generator(self, key: str) -> torch.Generator:
        h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) & ((1<<30)-1)
        g = torch.Generator()
        g.manual_seed(self.cfg.SEED ^ h)
        return g

    def __getitem__(self, idx: int):
        p, report, sentences, sid, last14 = self.items[idx]
        if not os.path.exists(p): raise FileNotFoundError(p)
        try:
            with Image.open(p) as im: im.verify()
        except (UnidentifiedImageError, OSError):
            raise RuntimeError(f"cannot_open:{p}")

        out_img = self.tfs({"image": p})
        img: torch.Tensor = out_img["image"]  # [1,H,W] in [0,1]
        ok, why = valid_processed_image(img, (1, self.cfg.CROP_SIZE, self.cfg.CROP_SIZE))
        if not ok: raise RuntimeError(f"bad_img:{why}:{p}")

        g = self._per_sample_generator(p)

        # 文本编码（pad/trunc 由 MAX_TEXT_LEN 控制；None 则动态）
        tok = self.tok.pack_sentences(
            sentences,
            max_len=self.cfg.MAX_TEXT_LEN,
            count_special_in_sent=self.cfg.COUNT_SPECIAL_IN_SENT
        )

        sample: Dict[str, Any] = {
            "report": report,
            "sentences": sentences,
            "study_id": sid,
            "last14": last14,
            "path": p,
            **tok,
        }

        if self.mode == "vae":
            img_masked, mask2d = make_vae_zero_masked(img, self.cfg.PATCH_NUM, self.cfg.MASK_LEVEL, g)
            sample["img"] = img
            sample["img_masked"] = img_masked
            sample["mask2d"] = mask2d  # [1,H,W], 1=mask, 0=keep
        elif self.mode == "vit":
            seq_orig = image_to_patch_sequence(img, self.cfg.PATCH_NUM)
            L = self.cfg.PATCH_NUM * self.cfg.PATCH_NUM
            mask_seq, ids_restore, ids_keep = mae_like_random_masking(L, self.cfg.MASK_LEVEL, img.device, g)
            seq_kept = gather_sequence_by_index(seq_orig, ids_keep)
            sample.update({
                "seq_orig": seq_orig,         # [L,1,ps,ps]
                "seq_kept": seq_kept,         # [len_keep,1,ps,ps]
                "mask_seq": mask_seq,         # [L]  1=mask
                "ids_keep": ids_keep,         # [len_keep]
                "ids_restore": ids_restore,   # [L]
            })
        else:
            raise ValueError(f"Unknown MODE: {self.mode}")

        return sample


# =======================
# collate（统一：文本动态对齐，图像按键存在性处理）
# =======================
def _pad_to_len(x: torch.Tensor, L: int, pad_val: int) -> torch.Tensor:
    L0 = x.numel()
    if L0 == L: return x
    pad = torch.full((L - L0,), pad_val, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], 0)

def collate_dynamic(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    pad_id = batch[0]["pad_token_id"]
    Lmax = max(it["ids_f"].numel() for it in batch)

    ids = []; attn = []; smask = []; typeids = []
    has_type = ("type_ids_f" in batch[0])

    for s in batch:
        ids.append(_pad_to_len(s["ids_f"],            Lmax, pad_id))
        attn.append(_pad_to_len(s["attention_mask_f"],Lmax, 0))
        smask.append(_pad_to_len(s["sentence_mask_f"],Lmax, 0))
        if has_type:
            typeids.append(_pad_to_len(s["type_ids_f"], Lmax, 0))

    out: Dict[str, Any] = {
        "ids_f": torch.stack(ids, 0),
        "attention_mask_f": torch.stack(attn, 0),
        "sentence_mask_f": torch.stack(smask, 0),
    }
    if has_type:
        out["type_ids_f"] = torch.stack(typeids, 0)

    if "img" in batch[0]:
        out["img"] = torch.stack([s["img"] for s in batch], 0)
        out["img_masked"] = torch.stack([s["img_masked"] for s in batch], 0)
        if "mask2d" in batch[0]:
            out["mask2d"] = torch.stack([s["mask2d"] for s in batch], 0)
    if "seq_orig" in batch[0]:
        for k in ["seq_orig","seq_kept","ids_keep","ids_restore","mask_seq"]:
            out[k] = [s[k] for s in batch]

    for k in ["report","sentences","study_id","last14","path"]:
        out[k] = [s[k] for s in batch]
    return out


# =======================
# 批打印
# =======================
def debug_print_batch(batch: Dict[str, Any], cfg: Any, batch_idx: int):
    ids = batch["ids_f"]; attn = batch["attention_mask_f"]; smask = batch["sentence_mask_f"]
    print(f"\n=== Batch #{batch_idx} (mode={cfg.MODE}) ===")
    print(f"ids_f      : {tuple(ids.shape)}   attention_mask_f: {tuple(attn.shape)}   sentence_mask_f: {tuple(smask.shape)}")
    if "img" in batch:
        print(f"img        : {tuple(batch['img'].shape)}   img_masked: {tuple(batch['img_masked'].shape)}   mask2d: {tuple(batch['mask2d'].shape)}")
    if "seq_orig" in batch:
        print(f"seq_orig   : {[tuple(x.shape) for x in batch['seq_orig']]}  (per-sample L may vary)")
        print(f"seq_kept   : {[tuple(x.shape) for x in batch['seq_kept']]}")

    head = cfg.PRINT_HEAD_N
    B = len(batch["report"])
    for i in range(min(B, 2)):
        print(f"--- sample {i} ---")
        print(f"  path: {batch['path'][i]}")
        print(f"  sentences({len(batch['sentences'][i])}): {batch['sentences'][i]}")
        li = ids.shape[1]
        print(f"  ids_f[:{li}]    :", ids[i, :li].tolist())
        print(f"  attn[:{li}]     :", attn[i, :li].tolist())
        print(f"  smask[:{li}]    :", smask[i, :li].tolist())
        print(f"  max sentence id : {int(smask[i].max().item())}")