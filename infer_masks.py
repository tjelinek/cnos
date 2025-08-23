import sys
from pathlib import Path

import torch

from repositories.cnos.src.model.fast_sam import FastSAM

sys.path.append(str((Path(__file__).parent).resolve()))
import numpy as np
from PIL import Image
import cv2
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from utils.image_utils import overlay_mask
from repositories.cnos.src.model.detector import CNOS


def infer_masks_for_folder(folder: Path, cfg: DictConfig):
    cnos_model: CNOS = instantiate(cfg.model).to('cuda')
    cnos_model.move_to_device()
    if hasattr(cnos_model.segmentor_model, "predictor"):
        cnos_model.segmentor_model.predictor.model = (
            cnos_model.segmentor_model.predictor.model.to('cuda')
        )
    else:
        cnos_model.segmentor_model.model.setup_model(device=torch.device('cuda'), verbose=True)
    folder = folder.resolve()

    for sequence in tqdm(sorted(folder.iterdir()), desc=f"[{folder}] Sequences"):
        image_folder = sequence / 'rgb'
        if not image_folder.exists():
            image_folder = sequence / 'grayscale'
        if not image_folder.exists():
            continue

        segment_model_name = 'fastsam' if isinstance(cnos_model.segmentor_model, FastSAM) else 'sam'
        proposals_dir = sequence / f"cnos_{segment_model_name}_proposals"
        visual_dir = sequence / f"cnos_{segment_model_name}_visual"
        proposals_dir.mkdir(parents=True, exist_ok=True)
        visual_dir.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(sorted(image_folder.iterdir()), leave=False, desc=f"Images in {sequence.name}"):
            img_name = img_path.stem
            img = np.array(Image.open(img_path).convert("RGB"))
            detections = cnos_model.get_filtered_detections(img)

            masks = detections.masks
            detections_descriptors = cnos_model.descriptor_model(img, detections)
            for i, m in enumerate(masks):
                mask_uint8 = (m["segmentation"].astype(np.uint8) * 255)
                vis = overlay_mask(img, m["segmentation"].astype(np.float32))
                cv2.imwrite(str(proposals_dir / f"{img_name}_{i:06d}.png"), mask_uint8)
                cv2.imwrite(str(visual_dir / f"{img_name}_{i:06d}.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    model = 'cnos_fast'  # either 'cnos' or 'cnos_fast'

    sys.path.append('/repositories/cnos')
    cfg_dir = (Path(__file__).parent / "configs").resolve()
    with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
        cfg = compose(
            config_name="run_inference",
            overrides=[
                f"model={model}",
            ],
        )

    base_path = Path('/mnt/personal/jelint19/data/')
    bop_path = base_path / 'bop'
    folder_paths = [
        bop_path / 'lmo' / 'test',
        bop_path / 'tless' / 'test_primesense',
        bop_path / 'tudl' / 'test',
        bop_path / 'icbin' / 'test',
        bop_path / 'itodd' / 'test',
        bop_path / 'hb' / 'test_kinect',
        bop_path / 'hb' / 'test_primesense',
        bop_path / 'ycbv' / 'test',
        bop_path / 'handal' / 'test',
    ]

    for folder_path in tqdm(folder_paths, desc="Datasets"):
        infer_masks_for_folder(folder_path, cfg)
