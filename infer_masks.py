from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from tqdm import tqdm

from repositories.cnos.src.model.sam import CustomSamAutomaticMaskGenerator, load_sam
from utils.image_utils import overlay_mask


def build_generator(cfg: DictConfig):
    seg_cfg = cfg.model.segmentor_model
    sam = load_sam(
        model_type=seg_cfg.sam.model_type,
        checkpoint_dir=seg_cfg.sam.checkpoint_dir
    )
    return CustomSamAutomaticMaskGenerator(
        sam=sam,
        points_per_batch=seg_cfg.points_per_batch,
        min_mask_region_area=seg_cfg.min_mask_region_area,
        stability_score_thresh=seg_cfg.stability_score_thresh,
        box_nms_thresh=seg_cfg.box_nms_thresh,
        segmentor_width_size=seg_cfg.segmentor_width_size,
    )


def infer_masks_for_folder(folder: Path, cfg: DictConfig):
    gen = build_generator(cfg)
    folder = folder.resolve()
    base_dir = folder.parent
    proposals_dir = base_dir / "cnos_sam_proposals"
    visual_dir = base_dir / "cnos_sam_visual"
    proposals_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    for sequence in tqdm(list(folder.iterdir()), desc=f"[{folder.name}] Sequences"):
        image_folder = sequence / 'rgb'
        if not image_folder.exists():
            image_folder = sequence / 'grayscale'
        if not image_folder.exists():
            continue

        for img_path in tqdm(list(image_folder.iterdir()), leave=False, desc=f"Images in {sequence.name}"):
            img_name = img_path.stem
            img = np.array(Image.open(img_path).convert("RGB"))
            masks = gen.generate(img)

            for i, m in enumerate(masks):
                mask_uint8 = (m["segmentation"].astype(np.uint8) * 255)
                vis = overlay_mask(img, m["segmentation"].astype(np.float32))
                cv2.imwrite(str(proposals_dir / f"{img_name}_{i:06d}.png"), mask_uint8)
                cv2.imwrite(str(visual_dir / f"{img_name}_{i:06d}.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    cfg_dir = (Path(__file__).parent / "configs").resolve()
    with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
        cfg = compose(config_name="run_inference")

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
