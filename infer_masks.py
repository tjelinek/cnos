import os, glob, json
import numpy as np
from PIL import Image
import torch
import hydra
from omegaconf import DictConfig

from src.model.sam import CustomSamAutomaticMaskGenerator, load_sam

@hydra.main(config_path="configs", config_name="run_inference", version_base=None)
def main(cfg: DictConfig):
    seg_cfg = cfg.model.segmentor_model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = load_sam(
        model_type=seg_cfg.sam.model_type,
        checkpoint_dir=seg_cfg.sam.checkpoint_dir,
    )

    gen = CustomSamAutomaticMaskGenerator(
        sam=sam,
        points_per_batch=seg_cfg.points_per_batch,
        min_mask_region_area=seg_cfg.min_mask_region_area,
        stability_score_thresh=seg_cfg.stability_score_thresh,
        box_nms_thresh=seg_cfg.box_nms_thresh,
        segmentor_width_size=seg_cfg.segmentor_width_size,
    )

    IMG_DIR = "/mnt/personal/jelint19/data/bop/"
    OUT_DIR = "/mnt/personal/jelint19/results/cnos/sam_masks/"
    os.makedirs(OUT_DIR, exist_ok=True)

    for p in sorted(glob.glob(os.path.join(IMG_DIR, "*.png"))):
        img = np.array(Image.open(p).convert("RGB"))
        masks = gen.generate(img)
        with open(os.path.join(OUT_DIR, os.path.splitext(os.path.basename(p))[0] + ".json"), "w") as f:
            json.dump(masks, f)

if __name__ == "__main__":
    main()
