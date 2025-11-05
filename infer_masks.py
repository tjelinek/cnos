import argparse
import json
import logging
import pickle
import shutil
import sys
from pathlib import Path
from time import time
from typing import Any

import torch
import torchvision

from segment_anything.utils.amg import mask_to_rle_pytorch

sys.path.append(str((Path(__file__).parent).resolve()))
import numpy as np
from PIL import Image
import cv2
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from utils.image_utils import overlay_mask, compute_overlap_ratio, compute_target_coverage
from repositories.cnos.src.model.detector import CNOS
from repositories.cnos.src.model.dinov2 import descriptor_from_hydra


def infer_masks_for_folder(folder: Path, base_cache_folder: Path, dataset: str, split: str, cfg: DictConfig,
                           detector_model_name: str, min_gt_overlap: float = 0.9, min_coverage_of_gt: float = 0.25,
                           training_sequences_mode: bool = False, images_subsample: int = 1):

    # Silence SAM2 logs
    logging.getLogger().setLevel(logging.WARNING)

    cnos_model: CNOS = instantiate(cfg.model).to('cuda')
    cnos_model.move_to_device()
    folder = folder.resolve()

    descriptor_dinov2 = descriptor_from_hydra('dinov2')
    descriptor_dinov3 = descriptor_from_hydra('dinov3')

    all_sequences = sorted(folder.iterdir())
    import random
    random.shuffle(all_sequences)
    for sequence in tqdm(all_sequences, desc=f"{detector_model_name}: [{folder}] Sequences", total=len(all_sequences)):
        source_channels = ['rgb']
        scene_gt_path = sequence / 'scene_gt.json'
        segmentations_path = sequence / 'mask_visib'

        if 'quest3' in split:
            source_channels = ['gray1']
            scene_gt_path = sequence / 'scene_gt_gray1.json'
            segmentations_path = sequence / 'mask_visib_gray1'
        elif 'aria' in split:
            source_channels = ['rgb']
            scene_gt_path = sequence / 'scene_gt_rgb.json'
            segmentations_path = sequence / 'mask_visib_rgb'
        elif 'itodd' in dataset:
            source_channels = ['gray']
            scene_gt_path = sequence / 'scene_gt.json'
            segmentations_path = sequence / 'mask_visib'

        if scene_gt_path.exists():
            with open(scene_gt_path, 'r') as scene_gt_f:
                scene_gt = json.load(scene_gt_f)
        else:
            scene_gt = None

        for channel in tqdm(source_channels, desc=f'Channel in {dataset}/{split}/{sequence.name}',
                            total=len(source_channels)):
            image_folder = sequence / channel
            if not image_folder.exists():
                image_folder = sequence / 'grayscale'
            if not image_folder.exists():
                continue

            # Create directories for both DINOv2 and DINOv3
            cache_sequence_dir = base_cache_folder / dataset / split / sequence.name

            if channel == 'rgb':
                proposals_dir_dinov2 = cache_sequence_dir / f"cnos_{detector_model_name}_detections_dinov2"
                detections_visual_dir = cache_sequence_dir / f"cnos_{detector_model_name}_visual"
                proposals_dir_dinov3 = cache_sequence_dir / f"cnos_{detector_model_name}_detections_dinov3"
            else:
                proposals_dir_dinov2 = cache_sequence_dir / f"{channel}_cnos_{detector_model_name}_detections_dinov2"
                detections_visual_dir = cache_sequence_dir / f"{channel}_cnos_{detector_model_name}_visual"
                proposals_dir_dinov3 = cache_sequence_dir / f"{channel}_cnos_{detector_model_name}_detections_dinov3"

            if detections_visual_dir.exists():
                shutil.rmtree(detections_visual_dir)
            proposals_dir_dinov2.mkdir(parents=True, exist_ok=True)
            detections_visual_dir.mkdir(parents=True, exist_ok=True)
            proposals_dir_dinov3.mkdir(parents=True, exist_ok=True)

            all_images = sorted(image_folder.iterdir())
            rng = random.Random(42)
            rng.shuffle(all_images)
            all_images = all_images[::images_subsample]

            for img_idx, img_path in tqdm(enumerate(all_images), total=len(all_images),
                                          leave=False, desc=f"Images in {dataset}/{split}/{sequence.name}",
                                          disable=True):
                img_name = img_path.stem

                img_id_int = int(img_name)
                if scene_gt is not None and training_sequences_mode:
                    if 'onboarding' in folder.stem:
                        gt_obj_ids = [int(sequence.stem.split('_')[1])]
                    else:
                        try:
                            image_gt_annotations = scene_gt[str(img_id_int)]
                        except KeyError:  # Mainly Aria sequences do not have GT to all the images
                            continue
                        gt_obj_ids = [obj_data['obj_id'] for obj_data in image_gt_annotations]
                    try:
                        gt_obj_segmentations = get_gt_segmentations_for_image(img_name, segmentations_path, gt_obj_ids)
                    except RuntimeError:  # For some images I get RuntimeError: Expected a non-empty file
                        continue
                else:
                    gt_obj_ids = None
                    gt_obj_segmentations = None

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

                if gt_obj_ids is not None and gt_obj_segmentations is not None:
                    overlap = compute_overlap_ratio(masks, gt_obj_segmentations)

                    assigned_indices = overlap.argmax(dim=1)
                    max_overlap = overlap.max(dim=1).values
                    assigned_gt = gt_obj_segmentations[assigned_indices]

                    coverage_of_gt = compute_target_coverage(masks, assigned_gt)
                    valid_indices = torch.where((max_overlap >= min_gt_overlap) &
                                                (coverage_of_gt >= min_coverage_of_gt))[0]

                    valid_indices_list = valid_indices.tolist()
                    masks_rle = [masks_rle[i] for i in valid_indices_list]
                    masks = masks[valid_indices]

                    assigned_indices = assigned_indices[valid_indices]
                    detections_obj_ids = [gt_obj_ids[gt_index] for gt_index in assigned_indices.tolist()]
                else:
                    valid_indices = torch.arange(0, len(masks_rle), dtype=torch.long, device=masks.device)
                    detections_obj_ids = None

                # Process both DINOv2 and DINOv3 descriptors
                for descriptor_func, pickle_path in [(descriptor_dinov2, pickle_path_dinov2),
                                                     (descriptor_dinov3, pickle_path_dinov3)]:
                    if not pickle_path.exists():
                        start_time = time()
                        torch.cuda.synchronize()
                        detections_cls_descriptors, detections_patch_descriptors = descriptor_func(img, detections)
                        torch.cuda.synchronize()
                        description_time = time() - start_time

                        detections_cls_descriptors = detections_cls_descriptors[valid_indices]
                        detections_patch_descriptors = detections_patch_descriptors[valid_indices]

                        detection_dict = {
                            "masks": masks_rle,
                            "descriptors": detections_cls_descriptors.numpy(force=True),
                            "patch_descriptors": detections_patch_descriptors.numpy(force=True),
                            "detections_object_ids": detections_obj_ids,
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


def get_gt_segmentations_for_image(img_name: str, segmentations_path: Path, gt_obj_ids: list[int] | list[Any])\
        -> torch.Tensor:
    gt_obj_segmentations = []
    for i in range(len(gt_obj_ids)):
        obj_segmentation_path = segmentations_path / f"{img_name}_{i:06d}.png"
        segmentation = torchvision.io.read_image(str(obj_segmentation_path)).to('cuda')
        gt_obj_segmentations.append(segmentation.to(torch.float32) / 255.)

    gt_obj_segmentations = torch.cat(gt_obj_segmentations, dim=0)
    return gt_obj_segmentations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        help="Dataset shortcut to run on. If not provided, runs on all datasets."
    )
    parser.add_argument(
        "--detector",
        default="sam",
        choices=["sam", "fastsam", "sam2"],
        help="Detector type to use. Default: sam"
    )
    args = parser.parse_args()

    detector = args.detector

    sys.path.append('/repositories/cnos')
    cfg_dir = (Path(__file__).parent / "configs").resolve()
    with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
        cfg = compose(config_name="run_inference", overrides=[f"model/segmentor_model={detector}"])

    base_path = Path('/mnt/data/vrg/public_datasets/')
    bop_path = base_path / 'bop'
    folders = {
        "tless-train_pbr": bop_path / 'tless' / 'train_pbr',
        "tless-train_primesense": bop_path / 'tless' / 'train_primesense',
        "lmo-train": bop_path / 'lmo' / 'train',
        "icbin-train_pbr": bop_path / 'icbin' / 'train_pbr',
        "icbin-train": bop_path / 'icbin' / 'train',
        "handal-onboarding_static": bop_path / 'handal' / 'onboarding_static',
        "handal-onboarding_dynamic": bop_path / 'handal' / 'onboarding_dynamic',
        "handal-train_pbr": bop_path / 'handal' / 'train_pbr',
        "hope-onboarding_static": bop_path / 'hope' / 'onboarding_static',
        "hope-onboarding_dynamic": bop_path / 'hope' / 'onboarding_dynamic',
        "hope-train_pbr": bop_path / 'hope' / 'train_pbr',
        "hot3d-object_ref_aria_dynamic_scenewise": bop_path / 'hot3d' / 'object_ref_aria_dynamic_scenewise',
        "hot3d-object_ref_aria_static_scenewise": bop_path / 'hot3d' / 'object_ref_aria_static_scenewise',
        "hot3d-train_pbr": bop_path / 'hot3d' / 'train_pbr',
        "hot3d-object_ref_quest3_dynamic_scenewise": bop_path / 'hot3d' / 'object_ref_quest3_dynamic_scenewise',
        "hot3d-object_ref_quest3_static_scenewise": bop_path / 'hot3d' / 'object_ref_quest3_static_scenewise',
        "hot3d-aria": bop_path / 'hot3d' / 'test_aria_scenewise',
        "hot3d-quest3": bop_path / 'hot3d' / 'test_quest3_scenewise',
        "hot3d-aria-train": bop_path / 'hot3d' / 'train_aria_scenewise',
        "hot3d-quest3-train": bop_path / 'hot3d' / 'train_quest3_scenewise',
        "lmo": bop_path / 'lmo' / 'test',
        "tless": bop_path / 'tless' / 'test_primesense',
        "icbin": bop_path / 'icbin' / 'test',
        "tudl": bop_path / 'tudl' / 'test',
        "tudl-train_pbr": bop_path / 'tudl' / 'train_pbr',
        "tudl-train_real": bop_path / 'tudl' / 'train_real',
        "tudl-train_renderer": bop_path / 'tudl' / 'train_renderer',
        "itodd": bop_path / 'itodd' / 'test',
        "itodd-val": bop_path / 'itodd' / 'val',
        "itodd-train_pbr": bop_path / 'itodd' / 'train_pbr',
        "itodd-templates_pyrenderer": bop_path / 'itodd' / 'templates_pyrenderer',
        "hb-train_pbr": bop_path / 'hb' / 'train_pbr',
        "hb-templates_pyrenderer": bop_path / 'hb' / 'templates_pyrenderer',
        "hb_kinect": bop_path / 'hb' / 'test_kinect',
        "hb_primesense": bop_path / 'hb' / 'test_primesense',
        "ycbv": bop_path / 'ycbv' / 'test',
        "ycbv-train_pbr": bop_path / 'ycbv' / 'train_pbr',
        "ycbv-train_real": bop_path / 'ycbv' / 'train_real',
        "ycbv-templates_pyrenderer": bop_path / 'ycbv' / 'templates_pyrenderer',
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

        training_sequences_mode = split not in ['test', 'test_primesense', 'test_kinect', 'val']
        images_subsample = 10 if training_sequences_mode else 1

        infer_masks_for_folder(folder_path, base_cache_path, dataset, split, cfg, detector,
                               training_sequences_mode=training_sequences_mode,
                               images_subsample=images_subsample)


if __name__ == "__main__":
    main()
