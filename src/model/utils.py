import logging

import numpy as np
import torch
import torchvision
from torchvision.ops.boxes import box_area

from segment_anything.utils.amg import mask_to_rle_pytorch
from src.utils.bbox_utils import xyxy_to_xywh, xywh_to_xyxy
from src.utils.inout import save_npz

lmo_object_ids = np.array(
    [
        1,
        5,
        6,
        8,
        9,
        10,
        11,
        12,
    ]
)  # object ID of occlusionLINEMOD is different


def mask_to_rle(binary_mask):
    rle = {"counts": [], "size": list(binary_mask.shape)}
    counts = rle.get("counts")

    last_elem = 0
    running_length = 0

    for i, elem in enumerate(binary_mask.ravel(order="F")):
        if elem == last_elem:
            pass
        else:
            counts.append(running_length)
            running_length = 0
            last_elem = elem
        running_length += 1

    counts.append(running_length)

    return rle


class BatchedData:
    """
    A structure for storing data in batched format.
    Implements basic filtering and concatenation.
    """

    def __init__(self, batch_size, data=None, **kwargs) -> None:
        self.batch_size = batch_size
        if data is not None:
            self.data = data
        else:
            self.data = []

    def __len__(self):
        assert self.batch_size is not None, "batch_size is not defined"
        return np.ceil(len(self.data) / self.batch_size).astype(int)

    def __getitem__(self, idx):
        assert self.batch_size is not None, "batch_size is not defined"
        return self.data[idx * self.batch_size: (idx + 1) * self.batch_size]

    def cat(self, data, dim=0):
        if len(self.data) == 0:
            self.data = data
        else:
            self.data = torch.cat([self.data, data], dim=dim)

    def append(self, data):
        self.data.append(data)

    def stack(self, dim=0):
        self.data = torch.stack(self.data, dim=dim)


class Detections:
    """
    A structure for storing detections.
    """

    def __init__(self, data) -> None:
        if isinstance(data, str):
            data = self.load_from_file(data)
        for key, value in data.items():
            setattr(self, key, value)
        self.keys = list(data.keys()) if isinstance(data, dict) else data._stats.keys()
        if "boxes" in self.keys:
            if isinstance(self.boxes, np.ndarray):
                self.to_torch()
            self.boxes = self.boxes.long()

    def remove_very_small_detections(self, config):
        img_area = self.masks.shape[1] * self.masks.shape[2]
        box_areas = box_area(self.boxes) / img_area
        mask_areas = self.masks.sum(dim=(1, 2)) / img_area
        keep_idxs = torch.logical_and(
            box_areas > config.min_box_size ** 2, mask_areas > config.min_mask_size
        )
        # logging.info(f"Removing {len(keep_idxs) - keep_idxs.sum()} detections")
        for key in self.keys:
            setattr(self, key, getattr(self, key)[keep_idxs])

    def apply_nms_per_object_id(self, nms_thresh=0.5):
        keep_idxs = BatchedData(None)
        all_indexes = torch.arange(len(self.object_ids), device=self.boxes.device)
        for object_id in torch.unique(self.object_ids):
            this_object_detections_mask = self.object_ids == object_id

            this_object_detections_indices = all_indexes[this_object_detections_mask]

            this_obj_keep_indices = torchvision.ops.nms(self.boxes[this_object_detections_mask].float(),
                                                        self.scores[this_object_detections_mask].float(), nms_thresh)

            keep_idxs.cat(this_object_detections_indices[this_obj_keep_indices])
        keep_idxs = keep_idxs.data
        for key in self.keys:
            setattr(self, key, getattr(self, key)[keep_idxs])

        return keep_idxs

    def apply_nms_for_masks_inside_masks(self):
        all_masks = self.masks
        keep_detections = torch.ones(len(all_masks), dtype=torch.bool, device=self.boxes.device)

        for i in range(len(all_masks)):
            current_mask = all_masks[i:i + 1]  # Keep batch dimension
            current_mask_intersection = current_mask & all_masks

            # Compute IoU between current mask and all masks
            intersection = (current_mask & current_mask_intersection).sum(dim=(1, 2))
            union = (current_mask | current_mask_intersection).sum(dim=(1, 2))
            ious = intersection.float() / union.float()

            # Find masks with high overlap (>90% IoU)
            high_overlap_masks = ious > 0.9
            high_overlap_masks[i] = False  # Exclude self-comparison

            # Find which of these overlapping masks have lower scores
            lower_score_masks = self.scores < self.scores[i]
            masks_to_suppress = high_overlap_masks & lower_score_masks

            # Keep detections that are NOT suppressed by current mask
            keep_detections &= ~masks_to_suppress

        keep_detections_indices = torch.nonzero(keep_detections).squeeze(1)
        for key in self.keys:
            setattr(self, key, getattr(self, key)[keep_detections_indices])

        return keep_detections_indices

    def apply_nms(self, nms_thresh=0.5):
        keep_idx = torchvision.ops.nms(
            self.boxes.float(), self.scores.float(), nms_thresh
        )
        for key in self.keys:
            setattr(self, key, getattr(self, key)[keep_idx])

    def add_attribute(self, key, value):
        setattr(self, key, value)
        self.keys.append(key)

    def __len__(self):
        return len(self.boxes)

    def check_size(self):
        mask_size = len(self.masks)
        box_size = len(self.boxes)
        score_size = len(self.scores)
        object_id_size = len(self.object_ids)
        assert (
                mask_size == box_size == score_size == object_id_size
        ), f"Size mismatch {mask_size} {box_size} {score_size} {object_id_size}"

    def to_numpy(self):
        for key in self.keys:
            setattr(self, key, getattr(self, key).cpu().numpy())

    def to_torch(self):
        for key in self.keys:
            a = getattr(self, key)
            setattr(self, key, torch.from_numpy(getattr(self, key)))

    def save_to_file(
            self, scene_id, frame_id, runtime, file_path, dataset_name, return_results=False, save_mask=True,
            save_score_distribution=False
    ):
        """
        scene_id, image_id, category_id, bbox, time
        """
        boxes = xyxy_to_xywh(self.boxes)
        results = {
            "scene_id": scene_id,
            "image_id": frame_id,
            "category_id": self.object_ids + 1
            if dataset_name != "lmo"
            else lmo_object_ids[self.object_ids],
            "score": self.scores,
            "bbox": boxes,
            "time": runtime,
        }
        if save_mask:
            results["segmentation"] = self.masks
        if save_score_distribution:
            assert hasattr(self, "score_distribution"), "score_distribution is not defined"
            results["score_distribution"] = self.score_distribution
        save_npz(file_path, results)
        if return_results:
            return results

    def load_from_file(self, file_path):
        data = np.load(file_path)
        boxes = xywh_to_xyxy(np.array(data["bbox"]))
        output = {
            "object_ids": data["category_id"] - 1,
            "bbox": boxes,
            "scores": data["score"],
        }
        if "segmentation" in data.keys():
            output["masks"] = data["segmentation"]
        if "score_distribution" in data.keys():
            output["score_distribution"] = data["score_distribution"]
        logging.info(f"Loaded {file_path}")
        return data

    def filter(self, idxs):
        for key in self.keys:
            setattr(self, key, getattr(self, key)[idxs])

    def clone(self):
        """
        Clone the current object
        """
        return Detections(self.__dict__.copy())


def convert_npz_to_json(idx, list_npz_paths):
    npz_path = list_npz_paths[idx]

    detections = {}
    with np.load(npz_path) as npz_file:
        detections = {key: npz_file[key] for key in npz_file.keys()}

    results = []
    results_with_score_distribution = []
    for idx_det in range(len(detections["bbox"])):
        result = {
            "scene_id": int(detections["scene_id"]),
            "image_id": int(detections["image_id"]),
            "category_id": int(detections["category_id"][idx_det]),
            "bbox": detections["bbox"][idx_det].tolist(),
            "score": float(detections["score"][idx_det]),
            "time": float(detections["time"]),
        }
        if "segmentation" in detections.keys():
            result["segmentation"] = mask_to_rle_pytorch(torch.tensor(detections["segmentation"][idx_det][None]) > 0)
        results.append(result)

        if "score_distribution" in detections.keys():
            result_with_score_distribution = result.copy()
            result_with_score_distribution["score_distribution"] = detections["score_distribution"][idx_det].tolist()
            results_with_score_distribution.append(result_with_score_distribution)
    return results, results_with_score_distribution
