import argparse
import pickle
import sys
from pathlib import Path
from time import time

import torch
from segment_anything.utils.amg import mask_to_rle_pytorch

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
from repositories.cnos.src.model.dinov2 import descriptor_from_hydra


def infer_masks_for_folder(folder: Path, base_cache_folder: Path, dataset: str, split: str, cfg: DictConfig,
                           cnos_model_name: str):
    cnos_model: CNOS = instantiate(cfg.model).to('cuda')
    cnos_model.move_to_device()
    folder = folder.resolve()

    descriptor_dinov2 = descriptor_from_hydra('dinov2')
    descriptor_dinov3 = descriptor_from_hydra('dinov3')

    all_sequences = sorted(folder.iterdir())
    for sequence in tqdm(all_sequences, desc=f"[{folder}] Sequences", total=len(all_sequences)):
        image_folder = sequence / 'rgb'
        if not image_folder.exists():
            image_folder = sequence / 'grayscale'
        if not image_folder.exists():
            continue

        segment_model_name = 'fastsam' if cnos_model_name == 'cnos_fast' else 'sam'

        # Create directories for both DINOv2 and DINOv3
        cache_sequence_dir = base_cache_folder / dataset / split / sequence.name
        proposals_dir_dinov2 = cache_sequence_dir / f"cnos_{segment_model_name}_detections_dinov2"
        detections_visual_dir = cache_sequence_dir / f"cnos_{segment_model_name}_visual"
        proposals_dir_dinov3 = cache_sequence_dir / f"cnos_{segment_model_name}_detections_dinov3"

        proposals_dir_dinov2.mkdir(parents=True, exist_ok=True)
        detections_visual_dir.mkdir(parents=True, exist_ok=True)
        proposals_dir_dinov3.mkdir(parents=True, exist_ok=True)

        all_images = sorted(image_folder.iterdir())
        for img_idx, img_path in tqdm(enumerate(all_images), total=len(all_images),
                                      leave=False, desc=f"Images in {dataset}/{split}/{sequence.name}"):
            img_name = img_path.stem

            pickle_path_dinov2 = Path(f"{proposals_dir_dinov2}/{img_name}.pkl")
            pickle_path_dinov3 = Path(f"{proposals_dir_dinov3}/{img_name}.pkl")

            # Skip if both files already exist
            if pickle_path_dinov2.exists() and pickle_path_dinov3.exists():
                continue

            img = np.array(Image.open(img_path).convert("RGB"))
            start_time = time()
            torch.cuda.synchronize()
            detections = cnos_model.get_filtered_detections(img)
            torch.cuda.synchronize()
            detections_time = time() - start_time

            masks = detections.masks
            masks_rle = mask_to_rle_pytorch((masks > 0).to(torch.long))

            # Process both DINOv2 and DINOv3 descriptors
            for descriptor_func, pickle_path in [(descriptor_dinov2, pickle_path_dinov2),
                                                 (descriptor_dinov3, pickle_path_dinov3)]:
                if not pickle_path.exists():
                    start_time = time()
                    torch.cuda.synchronize()
                    detections_cls_descriptors, detections_patch_descriptors = descriptor_func(img, detections)
                    torch.cuda.synchronize()
                    description_time = time() - start_time

                    detection_dict = {
                        "masks": masks_rle,
                        "descriptors": detections_cls_descriptors.numpy(force=True),
                        # "patch_descriptors": detections_patch_descriptors.numpy(force=True),
                        "detection_time": detections_time,
                        "description_time": description_time,
                    }

                    with open(pickle_path, "wb") as pickle_file:
                        pickle.dump(detection_dict, pickle_file)

            # Save visualizations (only for DINOv2 to avoid duplication since masks are the same)
            all_images_div_10 = len(all_images) // 10
            all_images_div_10 = round(all_images_div_10, -1)
            if img_idx % max(10, all_images_div_10) == 0:
                for i, m in enumerate(masks):
                    mask_uint8 = (m.numpy(force=True).astype(np.uint8) * 255)
                    vis = overlay_mask(img, m.numpy(force=True).astype(np.float32))
                    # cv2.imwrite(str(proposals_dir_dinov2 / f"{img_name}_{i:06d}.png"), mask_uint8)
                    cv2.imwrite(str(detections_visual_dir / f"{img_name}_{i:06d}.jpg"),
                                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def main():
    model = 'cnos'

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        help="Dataset shortcut to run on. If not provided, runs on all datasets."
    )
    args = parser.parse_args()

    sys.path.append('/repositories/cnos')
    cfg_dir = (Path(__file__).parent / "configs").resolve()
    with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
        cfg = compose(config_name="run_inference", overrides=[f"model={model}"])

    base_path = Path('/mnt/personal/jelint19/data/')
    bop_path = base_path / 'bop'
    folders = {
        "lmo": bop_path / 'lmo' / 'test',
        "tless": bop_path / 'tless' / 'test_primesense',
        "tudl": bop_path / 'tudl' / 'test',
        "icbin": bop_path / 'icbin' / 'test',
        "itodd": bop_path / 'itodd' / 'test',
        "hb_kinect": bop_path / 'hb' / 'test_kinect',
        "hb_primesense": bop_path / 'hb' / 'test_primesense',
        "ycbv": bop_path / 'ycbv' / 'test',
        "handal": bop_path / 'handal' / 'test',
        "hope": bop_path / 'hope' / 'test',
        "handal-val": bop_path / 'handal' / 'val',
        "hope-val": bop_path / 'hope' / 'val',
    }

    if args.dataset:
        targets = [(args.dataset, folders[args.dataset])]
    else:
        targets = list(folders.items())

    base_cache_path = Path('/mnt/personal/jelint19/cache/detections_cache/')
    for dataset_name, folder_path in tqdm(targets, desc="Datasets"):

        split = folder_path.name
        dataset = folder_path.parent.name

        infer_masks_for_folder(folder_path, base_cache_path, dataset, split, cfg, model)


if __name__ == "__main__":
    main()
