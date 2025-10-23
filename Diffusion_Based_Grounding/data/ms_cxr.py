from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils import data

from .utils import input_transformations, checkCoord


class MSCXR(data.Dataset):
    def __init__(self, transform_name=None, crop_size=448) -> None:
        base_dir = Path('/media/yuganlab/blackstone')
        images_dir = '/media/yuganlab/blackstone/xinlong/raw_data/MIMICCXR'
        ms_dir = '/media/yuganlab/ezstore1/Zongye/MS_CXR/MS_CXR_Local_Alignment_v1.1.0.csv'

        df = pd.read_csv(ms_dir)

        # merge boxes in the same img that correspond to the same text
        temp_df = df.groupby(["path", "label_text"]).head(1)
        p1 = temp_df.path.values.tolist()
        l1 = temp_df.label_text.values.tolist()
        i1 = temp_df.index.tolist()
        df = df.astype({"x": "object", "y": "object", "w": "object", "h": "object"})
        for idx, fname, lbl_txt in zip(i1, p1, l1):
            r = df[(df.path == fname) & (df.label_text == lbl_txt)]
            if len(r) == 1:
                continue
            x, y, w, h = r.x.values.tolist(), r.y.values.tolist(), r.w.values.tolist(), r.h.values.tolist()
            df.at[idx, "x"], df.at[idx, "y"], df.at[idx, "w"], df.at[idx, "h"] = x, y, w, h
        df.drop_duplicates(subset=["path", "label_text"], inplace=True, keep="first", ignore_index=True)

        self.df = df
        self.base_dir = base_dir
        self.images_dir = images_dir
        self.transforms = input_transformations(transform_name, crop_size=crop_size)
        self.transform_name = transform_name
    
    def get_class_names(self):
        return self.df.category_name.unique()
    
    def __len__(self):
        return len(self.df)
    
    def get_report_sentences(self, idx):
        reports_dir = self.base_dir / 'data/MIMIC_CXR_report_sentences'
        study_id = self.df.path[idx].split('/')[-2]
        report = reports_dir / f"{study_id}.json"

        try:
            with open(report, "r") as f:
                reportData = json.load(f)
        except FileNotFoundError:
            reportData = {"sentences": []}

        return reportData["sentences"]
    
    def __getitem__(self, idx):
        """
        Returns:
            transformed image (C, H, W) --> torch.Tensor
            transformed bounding box(es) in (x, y, w, h) format --> List[List[int]]
            original bounding box(es) coordinates --> List[List[int]]
            text prompt --> str
            original image path --> str
        """

        image_id = self.df.path[idx]
        image_id = self.base_dir / self.images_dir / image_id
        im_w, im_h = self.df.image_width[idx], self.df.image_height[idx]
        prompt = self.df.label_text[idx]
        x, y, w, h = self.df.x[idx], self.df.y[idx], self.df.w[idx], self.df.h[idx]

        if isinstance(x, list):
            original_bbox = [[e1, e2, e3, e4] for e1, e2, e3, e4 in zip(x, y, w, h)]
        else:
            original_bbox =[[x, y, w, h]]

        # sanity check for bbox coordinates
        # for i, bb in enumerate(original_bbox):
        #     x, y, w, h = bb
        #     x, x_end, y, y_end = checkCoord(x, im_w), checkCoord(x+w, im_w), checkCoord(y, im_h), checkCoord(y+h, im_h)
        #     original_bbox[i] = [x, y, x_end - x, y_end - y]

        if self.transforms is not None:
            bbox = torch.zeros((1, im_h, im_w), dtype=torch.int64)
            for v, bb in enumerate(original_bbox, 1):
                x, y, w, h = bb
                bbox[:, y:y+h, x:x+w] = v * 5  # multiply by a larger number to avoid interpolation issues!
        img_dict = self.transforms({'image': image_id, "bbox": bbox, "mask": torch.ones((im_h, im_w))})  # mask only for 'medrpg'
        img = img_dict["image"]
        bbox = img_dict["bbox"]
        if self.transforms is not None:
            final_bbox, remove_idx = [], 0
            for v in range(1, len(original_bbox) + 1):
                _, y_inds, x_inds = torch.nonzero(bbox == v * 5, as_tuple=True)
                try:
                    temp_bbox = [x_inds[0].item(), y_inds[0].item(), (x_inds[-1] - x_inds[0]).item(), (y_inds[-1] - y_inds[0]).item()]
                    final_bbox.append(temp_bbox)
                except IndexError:
                    # avoid these boxes!
                    print(f"Index {idx} has a [-1,] * 4 bbox", flush=True)
                    original_bbox.pop(v - 1 - remove_idx)
                    remove_idx += 1
                    # final_bbox.append([-1,] * 4)
                    continue
        else:
            final_bbox = bbox

        if self.transform_name == "medrpg":
            return img, final_bbox, original_bbox, img_dict["mask"], prompt, str(image_id), self.df.category_name[idx]
        return img, final_bbox, original_bbox, prompt, str(image_id), self.df.category_name[idx]
    
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import json
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils import data

# 复用你现有的工具与变换
from .utils import input_transformations, checkCoord


class IU_Xray(data.Dataset):
    def __init__(
        self,
        *,
        csv1_path: str = "/media/yuganlab/ezstore1/Zongye/IU_Xray/positive_impressions_terms_filtered.csv",
        csv2_path: str = "/media/yuganlab/ezstore1/Zongye/IU_Xray/indiana_reports.csv",
        csv3_path: str = "/media/yuganlab/ezstore1/Zongye/IU_Xray/indiana_projections.csv",
        image_prefix: str = "/media/yuganlab/ezstore1/Zongye/IU_Xray/images/images_normalized",
        csv1_sent_col: str = "kept_sentence",
        csv2_uid_col: str = "uid",
        csv3_uid_col: str = "uid",
        csv3_proj_col: str = "projection",
        csv3_fname_col: str = "filename",
        transform_name: Optional[str] = None,
        crop_size: int = 448,
        min_words: int = 2,
    ) -> None:

        self.image_prefix = Path(image_prefix)
        self.transform_name = transform_name
        self.transforms = input_transformations(transform_name, crop_size=crop_size)

        df1 = pd.read_csv(csv1_path)
        df2 = pd.read_csv(csv2_path)
        df3 = pd.read_csv(csv3_path)

        if len(df1) != len(df2):
            raise ValueError(
                f"csv1 与 csv2 行数不一致: len(csv1)={len(df1)}, len(csv2)={len(df2)}"
            )

        def _valid_sentence(x: Any) -> bool:
            if pd.isna(x):
                return False
            s = str(x).strip()
            if len(s) == 0:
                return False
            tokens = s.split()
            return len(tokens) >= min_words

        valid_mask = df1[csv1_sent_col].apply(_valid_sentence)
        valid_indices = np.flatnonzero(valid_mask.values)

        # Step 2: 用 valid_indices 去 csv2 取 uid
        uids = df2.iloc[valid_indices][csv2_uid_col].astype(str).tolist()
        prompts = df1.iloc[valid_indices][csv1_sent_col].astype(str).tolist()

        records: List[Dict[str, Any]] = []
        df3_proj = df3[df3[csv3_proj_col].astype(str).str.lower() == "frontal"]

        for uid, prompt in zip(uids, prompts):
            hits = df3_proj[df3_proj[csv3_uid_col].astype(str) == uid]
            if hits.empty:
                continue
            for _, row in hits.iterrows():
                fname = str(row[csv3_fname_col]).strip()
                if not fname:
                    continue
                rec = {
                    "uid": uid,
                    "prompt": prompt,
                    "filename": fname,
                }
                records.append(rec)

        if len(records) == 0:
            raise RuntimeError("没有匹配到任何 frontal 图像，请检查 CSV 内容与列名配置。")

        self.df = pd.DataFrame.from_records(records)

        self.default_category = "frontal"

    def get_class_names(self):
        return np.array([self.default_category])

    def __len__(self):
        return len(self.df)

    def _open_image_and_get_size(self, image_path: Path) -> Tuple[int, int]:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            return im.size  # (W, H)

    @staticmethod
    def _make_center_dummy_bbox(im_w: int, im_h: int) -> List[List[int]]:
        side = max(8, min(im_w, im_h) // 8)
        cx, cy = im_w // 2, im_h // 2
        x = max(0, cx - side // 2)
        y = max(0, cy - side // 2)
        if x + side > im_w:
            x = max(0, im_w - side)
        if y + side > im_h:
            y = max(0, im_h - side)
        return [[int(x), int(y), int(side), int(side)]]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt: str = row["prompt"]
        filename: str = row["filename"]

        # 拼接为完整路径
        image_path = (self.image_prefix / filename).resolve()

        # 打开图像获取尺寸（W, H）
        im_w, im_h = self._open_image_and_get_size(image_path)

        # 生成 dummy 原始 bbox
        original_bbox = self._make_center_dummy_bbox(im_w, im_h)

        if self.transforms is not None:
            bbox_mask = torch.zeros((1, im_h, im_w), dtype=torch.int64)
            for v, bb in enumerate(original_bbox, 1):
                x, y, w, h = bb
                x_end = x + w
                y_end = y + h
                x, y = max(0, x), max(0, y)
                x_end, y_end = min(im_w, x_end), min(im_h, y_end)
                if x_end > x and y_end > y:
                    bbox_mask[:, y:y_end, x:x_end] = v * 5

        img_dict = self.transforms(
            {
                "image": str(image_path),
                "bbox": bbox_mask if self.transforms is not None else None,
                "mask": torch.ones((im_h, im_w), dtype=torch.float32),
            }
        )

        img = img_dict["image"]
        bbox_after = img_dict["bbox"]  # (1, H', W'), 其中目标值为 v*5
        final_bbox: List[List[int]] = []

        if self.transforms is not None:
            remove_idx = 0
            for v in range(1, len(original_bbox) + 1):
                _, y_inds, x_inds = torch.nonzero(bbox_after == v * 5, as_tuple=True)
                try:
                    x0, x1 = x_inds[0].item(), x_inds[-1].item()
                    y0, y1 = y_inds[0].item(), y_inds[-1].item()
                    final_bbox.append([x0, y0, (x1 - x0), (y1 - y0)])
                except IndexError:
                    original_bbox.pop(v - 1 - remove_idx)
                    remove_idx += 1
                    continue
        else:
            final_bbox = original_bbox

        if self.transform_name == "medrpg":
            return (
                img,
                final_bbox,
                original_bbox,
                img_dict["mask"],
                prompt,
                str(image_path),
                self.default_category,
            )

        return (
            img,
            final_bbox,
            original_bbox,
            prompt,
            str(image_path),
            self.default_category,
        )
