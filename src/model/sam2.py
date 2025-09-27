# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import logging
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def load_sam2(
        model_type: str,
        checkpoint_dir: Union[str, Path],
        checkpoint_file: Optional[str] = None,
        config_file: Optional[str] = None,
        device: str = "cuda",
        **kwargs
) -> torch.nn.Module:
    """Load SAM 2 model."""
    from hydra.core.global_hydra import GlobalHydra
    import sam2

    checkpoint_dir = Path(checkpoint_dir)

    # Auto-resolve checkpoint and config files
    if checkpoint_file is None or config_file is None:
        auto_checkpoint, auto_config = _resolve_sam2_files(model_type)
        if checkpoint_file is None:
            checkpoint_file = auto_checkpoint
        if config_file is None:
            config_file = auto_config

    checkpoint_path = checkpoint_dir / checkpoint_file

    # Verify files exist
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM 2 checkpoint not found: {checkpoint_path}")

    logging.info(f"Loading SAM 2 model from {checkpoint_path}")

    cfg_dir = Path(sam2.__file__).parent / "configs"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    from hydra import initialize_config_dir
    with initialize_config_dir(config_dir=str(cfg_dir.resolve()), version_base=None, job_name="sam2"):
        sam2_model = build_sam2(config_file, str(checkpoint_path), device=device)

    return sam2_model


def _resolve_sam2_files(model_type: str) -> tuple[str, str]:
    """Resolve SAM2 checkpoint and config file names from model type."""
    sam2_files = {
        'sam2.1_hiera_tiny': ('sam2.1_hiera_tiny.pt', 'sam2.1/sam2.1_hiera_t.yaml'),
        'sam2.1_hiera_small': ('sam2.1_hiera_small.pt', 'sam2.1/sam2.1_hiera_s.yaml'),
        'sam2.1_hiera_base_plus': ('sam2.1_hiera_base_plus.pt', 'sam2.1/sam2.1_hiera_b+.yaml'),
        'sam2.1_hiera_large': ('sam2.1_hiera_large.pt', 'sam2.1/sam2.1_hiera_l.yaml'),
        'sam2_hiera_tiny': ('sam2_hiera_tiny.pt', 'sam2_hiera_t.yaml'),
        'sam2_hiera_small': ('sam2_hiera_small.pt', 'sam2_hiera_s.yaml'),
        'sam2_hiera_base_plus': ('sam2_hiera_base_plus.pt', 'sam2_hiera_b+.yaml'),
        'sam2_hiera_large': ('sam2_hiera_large.pt', 'sam2_hiera_l.yaml'),
    }

    if model_type in sam2_files:
        return sam2_files[model_type]
    else:
        checkpoint_file = f"{model_type}.pt"
        if 'tiny' in model_type:
            config_file = f"{model_type.split('_')[0]}/sam2.1_hiera_t.yaml"
        elif 'small' in model_type:
            config_file = f"{model_type.split('_')[0]}/sam2.1_hiera_s.yaml"
        elif 'base' in model_type:
            config_file = f"{model_type.split('_')[0]}/sam2.1_hiera_b+.yaml"
        elif 'large' in model_type:
            config_file = f"{model_type.split('_')[0]}/sam2.1_hiera_l.yaml"
        else:
            config_file = f"{model_type}.yaml"
        return checkpoint_file, config_file


class CustomSamAutomaticMaskGenerator:
    def __init__(
            self,
            sam,
            min_mask_region_area: int = 0,
            points_per_batch: int = 64,
            stability_score_thresh: float = 0.95,
            box_nms_thresh: float = 0.7,
            crop_overlap_ratio: float = 512 / 1500,
            segmentor_width_size=None,
    ):
        self.segmentor_width_size = segmentor_width_size
        self.sam = sam

        # SAM2 parameters mapping
        sam2_params = {
            'model': sam,
            'points_per_batch': points_per_batch,
            'pred_iou_thresh': stability_score_thresh,
            'stability_score_thresh': stability_score_thresh,
            'box_nms_thresh': box_nms_thresh,
            'min_mask_region_area': min_mask_region_area,
        }
        if crop_overlap_ratio != 512 / 1500:  # only add if different from default
            sam2_params['crop_overlap_ratio'] = crop_overlap_ratio

        self._sam2_generator = SAM2AutomaticMaskGenerator(**sam2_params)
        logging.info(f"Init CustomSamAutomaticMaskGenerator with SAM2 done!")

    @property
    def model(self):
        """Provide access to the underlying SAM model for compatibility."""
        return self.sam

    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Generate masks - unified interface for SAM1 and SAM2."""
        if self.is_sam2:
            return self._sam2_generator.generate(image)
        else:
            return self.generate_masks(image)

    def preprocess_resize(self, image: np.ndarray):
        orig_size = image.shape[:2]
        height_size = int(self.segmentor_width_size * orig_size[0] / orig_size[1])
        resized_image = cv2.resize(
            image.copy(), (self.segmentor_width_size, height_size)  # (width, height)
        )
        return resized_image

    def postprocess_resize(self, detections, orig_size):
        detections["masks"] = F.interpolate(
            detections["masks"].unsqueeze(1).float(),
            size=(orig_size[0], orig_size[1]),
            mode="bilinear",
            align_corners=False,
        )[:, 0, :, :]
        scale = orig_size[1] / self.segmentor_width_size
        detections["boxes"] = detections["boxes"].float() * scale
        detections["boxes"][:, [0, 2]] = torch.clamp(
            detections["boxes"][:, [0, 2]], 0, orig_size[1] - 1
        )
        detections["boxes"][:, [1, 3]] = torch.clamp(
            detections["boxes"][:, [1, 3]], 0, orig_size[0] - 1
        )
        return detections

    @torch.no_grad()
    def generate_masks(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if self.segmentor_width_size is not None:
            orig_size = image.shape[:2]
            image = self.preprocess_resize(image)
        # Generate masks
        mask_data = self._generate_masks(image)

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )
        if self.segmentor_width_size is not None:
            mask_data = self.postprocess_resize(mask_data, orig_size)
        return mask_data

    def _generate_masks(self, image: np.ndarray) -> MaskData:
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # Iterate over image crops
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            data.cat(crop_data)

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        data["masks"] = [torch.from_numpy(rle_to_mask(rle)) for rle in data["rles"]]
        data["masks"] = torch.stack(data["masks"])
        data["masks"] = data["masks"].to(data["boxes"].device)
        # return data
        return {"masks": data["masks"], "boxes": data["boxes"]}

    def remove_small_detections(self, mask_data: MaskData, img_size: List) -> MaskData:
        # calculate area and number of pixels in each mask
        area = box_area(mask_data["boxes"]) / (img_size[0] * img_size[1])
        idx_selected = area >= self.mask_post_processing.min_box_size
        mask_data.filter(idx_selected)
        return mask_data