import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchvision.transforms as T
import numpy as np

from PIL import Image
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from torchvision.ops import masks_to_boxes

from src.model.utils import BatchedData, Detections
from src.utils.bbox_utils import CropResizePad, CustomResizeLongestSide

descriptor_size = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


class CustomDINOv2(pl.LightningModule):
    def __init__(
        self,
        model_name,
        model,
        token_name,
        image_size,
        chunk_size,
        descriptor_width_size,
        patch_size=14,
    ):
        super().__init__()
        self.model_name = model_name
        self.model = model
        self.token_name = token_name
        self.chunk_size = chunk_size
        self.patch_size = patch_size
        self.proposal_size = image_size
        self.descriptor_width_size = descriptor_width_size
        logging.info(f"Init CustomDINOv2 done!")
        self.rgb_normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        # use for global feature
        self.rgb_proposal_processor = CropResizePad(self.proposal_size)
        self.rgb_resize = CustomResizeLongestSide(
            descriptor_width_size, dividable_size=self.patch_size
        )
        logging.info(
            f"Init CustomDINOv2 with full size={descriptor_width_size} and proposal size={self.proposal_size} done!"
        )

    def process_rgb_proposals(self, image_np, masks, boxes):
        """
        1. Normalize image with DINOv2 transfom
        2. Mask and crop each proposals
        3. Resize each proposals to predefined longest image size
        """
        num_proposals = len(masks)
        rgb = self.rgb_normalize(image_np).to(masks.device).float()
        rgbs = rgb.unsqueeze(0).repeat(num_proposals, 1, 1, 1)
        masked_rgbs = rgbs * masks.unsqueeze(1)
        processed_masked_rgbs = self.rgb_proposal_processor(
            masked_rgbs, boxes
        )  # [N, 3, target_size, target_size]
        return processed_masked_rgbs

    @torch.no_grad()
    def compute_features(self, images, token_name):
        if token_name == "x_norm_clstoken":
            if images.shape[0] > self.chunk_size:
                cls_features, spatial_patch_features = self.forward_by_chunk(images)
            else:
                model_output = self.model.forward_features(images)
                cls_features = model_output['x_norm_clstoken']
                patch_features = model_output['x_norm_patchtokens']

                batch_size, num_patches, hidden_dim = patch_features.shape
                grid_size = int(num_patches ** 0.5)
                spatial_patch_features = patch_features.reshape(batch_size, grid_size, grid_size, hidden_dim)

        else:  # get both features
            raise NotImplementedError
        return cls_features, spatial_patch_features

    @torch.no_grad()
    def forward_by_chunk(self, processed_rgbs):
        batch_rgbs = BatchedData(batch_size=self.chunk_size, data=processed_rgbs)
        del processed_rgbs  # free memory
        cls_features = BatchedData(batch_size=self.chunk_size)
        spatial_patch_features = BatchedData(batch_size=self.chunk_size)
        for idx_batch in range(len(batch_rgbs)):
            cls_feats, spatial_patch_feats = self.compute_features(batch_rgbs[idx_batch], token_name="x_norm_clstoken")
            cls_features.cat(cls_feats)
            spatial_patch_features.cat(spatial_patch_feats)

        return cls_features.data, spatial_patch_features.data


    @torch.no_grad()
    def forward_cls_token(self, image_np, proposals):
        processed_rgbs = self.process_rgb_proposals(
            image_np, proposals.masks, proposals.boxes
        )
        return self.forward_by_chunk(processed_rgbs)

    @torch.no_grad()
    def forward(self, image_np, proposals):
        return self.forward_cls_token(image_np, proposals)

    def get_detections_from_files(self, image_path: Path, segmentation_path: Path, black_background: bool = False):
        image = Image.open(image_path).convert('RGB')
        image_tensor = torch.from_numpy(np.asarray(image)).to(self.device)
        segmentation = Image.open(segmentation_path)
        segmentation_mask = torch.from_numpy(np.array(segmentation)).unsqueeze(0).to(self.device)
        segmentation_bbox = masks_to_boxes(segmentation_mask)
        if black_background:
            image_tensor *= segmentation_mask.squeeze().unsqueeze(-1)
        image_np = image_tensor.to(torch.uint8).numpy(force=True)
        detections = Detections({'masks': segmentation_mask, 'boxes': segmentation_bbox})
        dino_cls_descriptor, dino_dense_descriptor = self.forward(image_np, detections)

        return dino_cls_descriptor, dino_dense_descriptor


def descriptor_from_hydra(device='cuda') -> CustomDINOv2:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    cfg_dir = (Path(__file__).parent.parent.parent / 'configs').resolve()
    with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
        cnos_cfg = compose(config_name="run_inference")

    dino_descriptor: CustomDINOv2 = instantiate(cnos_cfg.model.descriptor_model).to(device)
    dino_descriptor.model.device = device

    return dino_descriptor
