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


def infer_masks_for_folder(folder: Path, cfg: DictConfig, cnos_model: str):
    cnos_model: CNOS = instantiate(cfg.model).to('cuda')
    cnos_model.move_to_device()
    folder = folder.resolve()

    for sequence in tqdm(sorted(folder.iterdir()), desc=f"[{folder}] Sequences"):
        image_folder = sequence / 'rgb'
        if not image_folder.exists():
            image_folder = sequence / 'grayscale'
        if not image_folder.exists():
            continue

        segment_model_name = 'fastsam' if cnos_model == 'cnos_fast' else 'sam'
        proposals_dir = sequence / f"cnos_{segment_model_name}_detections"
        visual_dir = sequence / f"cnos_{segment_model_name}_visual"
        proposals_dir.mkdir(parents=True, exist_ok=True)
        visual_dir.mkdir(parents=True, exist_ok=True)

        all_images = sorted(image_folder.iterdir())
        for img_idx, img_path in tqdm(enumerate(all_images),
                                      leave=False, desc=f"Images in {sequence.name}"):
            img_name = img_path.stem
            img = np.array(Image.open(img_path).convert("RGB"))
            start_time = time()
            torch.cuda.synchronize()
            detections = cnos_model.get_filtered_detections(img)
            torch.cuda.synchronize()
            detections_time = time() - start_time

            masks = detections.masks

            start_time = time()
            torch.cuda.synchronize()
            detections_cls_descriptors, detections_patch_desriptors = cnos_model.descriptor_model(img, detections)
            torch.cuda.synchronize()
            description_time = time() - start_time

            masks_rle = mask_to_rle_pytorch((masks > 0).to(torch.long))

            detection_dict = {
                "masks": masks_rle,
                "descriptors": detections_cls_descriptors.numpy(force=True),
                "patch_descriptors": detections_patch_desriptors.numpy(force=True),
                "detection_time": detections_time,
                "description_time": description_time,
            }

            pickle_path = f"{proposals_dir}/{img_name}.pkl"
            with open(pickle_path, "wb") as pickle_file:
                pickle.dump(detection_dict, pickle_file)

            all_images_div_10 = len(all_images) // 10
            all_images_div_10 = round(all_images_div_10, -1)
            if img_idx % max(10, all_images_div_10) == 0:
                for i, m in enumerate(masks):
                    mask_uint8 = (m.numpy(force=True).astype(np.uint8) * 255)
                    vis = overlay_mask(img, m.numpy(force=True).astype(np.float32))
                    # cv2.imwrite(str(proposals_dir / f"{img_name}_{i:06d}.png"), mask_uint8)
                    cv2.imwrite(str(visual_dir / f"{img_name}_{i:06d}.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    model = 'cnos_fast'

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["lmo", "tless", "tudl", "icbin", "itodd", "hb_kinect", "hb_primesense", "ycbv", "handal"],
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
    }

    if args.dataset:
        targets = [folders[args.dataset]]
    else:
        targets = list(folders.values())

    for folder_path in tqdm(targets, desc="Datasets"):
        infer_masks_for_folder(folder_path, cfg, model)
