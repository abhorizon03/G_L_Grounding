from pathlib import Path
import pydicom
from PIL import Image
from itertools import combinations
import numpy as np
import pandas as pd
import torch
from torch.utils import data

from .utils import input_transformations


PATHOLOGIES = [
    "Lung Opacity",
    "Cardiomegaly",
    "Pleural effusion",
    "Pneumothorax",
    "Consolidation",
    "Atelectasis",
    "Edema",
    "Pneumonia"
]

# random samples from the training set to be transferred to the test set
# EDEMA_IDS_FOR_TEST = [
#     'b5fa2fc3b3e2a3bbf93884481fb08b74',
#     'f89143595274fa6016f6eec550442af9',
#     'c70dce909198abf8b39a7e0d41c9a895'
# ]


class VinDr_CXR(data.Dataset):
    def __init__(self, split="train", edema_samples=None, transform_name="vindr"):
        # if split == "test":
        #     assert edema_samples is not None, "Samples for Edema class were not provided"
        
        # base_dir = Path('/remote/rds/users/s2501010/Causal_TxtImg/vindr_cxr_jpg')
        base_dir = Path("/remote/rds/groups/idcom_imaging/data/VinDR_CXR/physionet.org/files/")
        vindr_dir = base_dir / f"vindr-cxr/1.0.0/"
        # img_dir = vindr_dir / f"{split}_resized"
        img_dir = vindr_dir / f"{split}_jpg"
        annot_dir = vindr_dir / "annotations"

        # df = pd.read_csv(annot_dir / f"annotations_percentage_coordinate_{split}.csv", index_col=0)  # local labels (Findings)
        df = pd.read_csv(annot_dir / f"annotations_{split}.csv")
        lbl_df = pd.read_csv(annot_dir / f"image_labels_{split}.csv")  # global labels -- only Pneumonia is useful here

        lbl_df = lbl_df.loc[:, lbl_df.columns.isin(["image_id"] + PATHOLOGIES)]
        lbl_df = lbl_df.loc[(lbl_df[PATHOLOGIES] != 0).any(axis=1)]  # filtering
        lbl_df["s"] = lbl_df.sum(axis=1, numeric_only=True)
        # keep samples for Pneumonia that are not overlapping with the rest of the 7 pathologies
        # lbl_df = lbl_df[(lbl_df.s == 1) & (lbl_df.Pneumonia == 1)]
        # lbl_df.reset_index(drop=True, inplace=True)

        df = df.dropna()
        df = df[(df.class_name.isin(PATHOLOGIES)) | (df.image_id.isin(lbl_df.image_id.unique().tolist()))]
        # for Pneumonia: keep only ILD and Infiltration
        df = df[df.class_name.isin(PATHOLOGIES + ["ILD", "Infiltration"])]
        df.loc[df.class_name.isin(["ILD", "Infiltration"]), "class_name"] = "Pneumonia"  # rename class
        df.reset_index(drop=True, inplace=True)

        self.df = df
        self.lbl_df = lbl_df

        if split == "train":
            indices_to_drop = self._filter_bboxes()
            self.df.drop(index=indices_to_drop, inplace=True)
            self.df.reset_index(drop=True, inplace=True)

        # Edema samples are only for test set
        # if split == "train":
        #     # self.edema_df = self.df[self.df.image_id.isin(EDEMA_IDS_FOR_TEST)].copy()
        #     # self.df = self.df[~self.df.image_id.isin(EDEMA_IDS_FOR_TEST)]
        #     # self.edema_df = self.df[self.df.class_name == "Edema"].copy()
        #     self.df = self.df[self.df.class_name.isin([p for p in PATHOLOGIES if p != "Edema"])]
        # else:
        #     temp_df = pd.read_csv(annot_dir / "annotations_train.csv", index_col=0)  # _percentage_coordinate
        #     edema_samples = temp_df[temp_df.class_name == "Edema"].copy()
        #     self.df = pd.concat([self.df, edema_samples], ignore_index=True)
        # self.df.reset_index(drop=True, inplace=True)

        self._merge_bbox_coords()

        self.img_dir = img_dir
        self.edema_dir = vindr_dir / "train"  # train_resized

        self.transforms = input_transformations(transform_name, split=split)
        self.transform_name = transform_name
    
    def __len__(self):
        return len(self.df)
    
    @staticmethod
    def get_class_names():
        return PATHOLOGIES
    
    def _filter_bboxes(self):
        '''
        filter bboxes with same class_name and IoU >= threshold
        '''

        iou_thresh = 0.55
        indices_to_drop = []
        for p in PATHOLOGIES:
            # keep unique ids corresponding to given pathology
            ## temp_ids = self.lbl_df[(self.lbl_df.s == 1) & (self.lbl_df[p] == 1)].image_id.unique().tolist()
            temp_ids = self.df[self.df.class_name == p].image_id.unique().tolist()

            # get bbox(es) per image id
            for temp_id in temp_ids:
                temp_df = self.df[(self.df.image_id == temp_id) & (self.df.class_name == p)].copy()
                x_min, y_min, x_max, y_max = temp_df.x_min.values, temp_df.y_min.values, temp_df.x_max.values, temp_df.y_max.values
                if len(x_min) < 2:
                    continue
                temp_df_inds = temp_df.index.values
                # pairwise evaluation of bboxes
                for inds_pairs in combinations(np.arange(len(x_min)), 2):
                    inds_pairs = list(inds_pairs)
                    intersection = np.maximum(x_max[inds_pairs].min() - x_min[inds_pairs].max(), 0) * np.maximum(y_max[inds_pairs].min() - y_min[inds_pairs].max(), 0)
                    areas = (x_max[inds_pairs] - x_min[inds_pairs]) * (y_max[inds_pairs] - y_min[inds_pairs])
                    # temp_df.loc[:, "areas"] = areas
                    union = areas.sum() - intersection
                    iou = intersection / union
                    if iou >= iou_thresh:
                        # keep only the bbox with largest area
                        res = inds_pairs[areas.argmin()]
                        indices_to_drop.append(temp_df_inds[res])
        
        return np.unique(indices_to_drop)
    
    def _merge_bbox_coords(self):
        """
        Merge samples with more than one bbox per pathology to a single entry
        """

        temp_df = self.df.groupby(["image_id", "class_name"]).head(1)
        p1 = temp_df.image_id.values.tolist()
        c1 = temp_df.class_name.values.tolist()
        i1 = temp_df.index.tolist()
        self.df = self.df.astype({"x_min": "object", "y_min": "object", "x_max": "object", "y_max": "object"})
        for idx, p_id, cl in zip(i1, p1, c1):
            r = self.df[(self.df.image_id == p_id) & (self.df.class_name == cl)]
            if len(r) == 1:
                continue
            x_min, y_min, x_max, y_max = r.x_min.values.tolist(), r.y_min.values.tolist(), r.x_max.values.tolist(), r.y_max.values.tolist()
            self.df.at[idx, "x_min"], self.df.at[idx, "y_min"], self.df.at[idx, "x_max"], self.df.at[idx, "y_max"] = x_min, y_min, x_max, y_max
        self.df.drop_duplicates(subset=["image_id", "class_name"], inplace=True, keep="first", ignore_index=True)
    
    def _num_samples_per_pathology(self):
        res = dict()
        for pathology in PATHOLOGIES:
            temp = self.df[self.df.class_name == pathology]
            samples = len(temp.image_id.unique().tolist())
            res[pathology] = {"samples": samples, "total_bboxes": len(temp)}
        
        return res
    
    def __getitem__(self, idx):
        """
        Returns:
            transformed image (C, H, W) --> torch.Tensor
            transformed bounding box(es) in (x, y, w, h) format --> List[List[int]]
            original bounding box(es) coordinates --> List[List[int]]
            text prompt --> str
            original image path --> str
        """

        image_id = self.df.image_id[idx]
        if self.df.class_name[idx] == "Edema":
            image_id = self.edema_dir / f"{image_id}.jpg"
        else:
            image_id = self.img_dir / f"{image_id}.jpg"
        
        im_w, im_h = Image.open(Path(image_id)).size

        prompt = f"The patient has {self.df.class_name[idx]}"
        # prep = "a"
        # if self.df.class_name[idx].lower()[0] in "aeiou":
        #     prep += "n"
        # prompt = f"There is {prep} {self.df.class_name[idx]}."

        if isinstance(self.df.x_min[idx], list):
            # x_min, y_min, x_max, y_max = [int(c1 * im_w) for c1 in self.df.x_min[idx]], [int(c2 * im_h) for c2 in self.df.y_min[idx]], [int(c3 * im_w) for c3 in self.df.x_max[idx]], [int(c4 * im_h) for c4 in self.df.y_max[idx]]
            x_min, y_min, x_max, y_max = [int(c1) for c1 in self.df.x_min[idx]], [int(c2) for c2 in self.df.y_min[idx]], [int(c3) for c3 in self.df.x_max[idx]], [int(c4) for c4 in self.df.y_max[idx]]
            x, y = x_min, y_min
            w, h = [c2 - c1 for (c1, c2) in zip(x_min, x_max)], [c4 - c3 for (c3, c4) in zip(y_min, y_max)]
            original_bbox = [[e1, e2, e3, e4] for e1, e2, e3, e4 in zip(x, y, w, h)]
        else:
            # x_min, y_min, x_max, y_max = int(self.df.x_min[idx] * im_w), int(self.df.y_min[idx] * im_h), int(self.df.x_max[idx] * im_w), int(self.df.y_max[idx] * im_h)
            x_min, y_min, x_max, y_max = int(self.df.x_min[idx]), int(self.df.y_min[idx]), int(self.df.x_max[idx]), int(self.df.y_max[idx])
            x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min
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
                bbox[:, y:y+h, x:x+w] = v * 150  # multiply by a larger number to avoid interpolation issues!
        img_dict = self.transforms({'image': image_id, "bbox": bbox})
        img = img_dict["image"]
        bbox = img_dict["bbox"]
        if self.transforms is not None:
            final_bbox, remove_idx = [], 0
            for v in range(1, len(original_bbox) + 1):
                _, y_inds, x_inds = torch.nonzero(bbox == v * 150, as_tuple=True)
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

        return img, final_bbox, original_bbox, prompt, str(image_id), self.df.class_name[idx]