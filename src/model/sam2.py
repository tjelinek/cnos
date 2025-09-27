# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import logging
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import logging
sam2_logger = logging.getLogger('sam2')
sam2_logger.setLevel(logging.WARNING)  # Only show warnings and errors

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
        """Generate masks using SAM2."""
        return self._sam2_generator.generate(image)

    def generate_masks(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        """Generate masks and return in format expected by Detections class."""
        # Get SAM2 results
        sam2_results = self._sam2_generator.generate(image)

        if not sam2_results:
            # Return empty tensors if no detections
            h, w, c = image.shape
            return {
                "masks": torch.empty(0, h, w, dtype=torch.bool).to(self.sam.device),
                "boxes": torch.empty(0, 4, dtype=torch.float32).to(self.sam.device)
            }

        # Convert SAM2 format to expected format
        masks = []
        boxes = []

        for result in sam2_results:
            # Extract mask
            if isinstance(result["segmentation"], dict):
                # RLE format - need to decode
                from pycocotools import mask as mask_utils
                mask = mask_utils.decode(result["segmentation"])
            else:
                # Binary mask format
                mask = result["segmentation"]
            masks.append(torch.from_numpy(mask.astype(bool)).to(self.sam.device))

            # Extract box (convert from XYWH to XYXY format)
            bbox = result["bbox"]  # [x, y, w, h]
            x, y, w, h = bbox
            xyxy_box = [x, y, x + w, y + h]  # [x1, y1, x2, y2]
            boxes.append(xyxy_box)

        # Stack into tensors
        masks_tensor = torch.stack(masks, dim=0)
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).to(self.sam.device)
        breakpoint()

        return {
            "masks": masks_tensor,
            "boxes": boxes_tensor
        }