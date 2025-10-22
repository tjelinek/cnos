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
    # DINOv2 models
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
    # DINOv3 ViT models (web images - LVD-1689M)
    "dinov3_vits16": 384,
    "dinov3_vits16plus": 384,
    "dinov3_vitb16": 768,
    "dinov3_vitl16": 1024,
    "dinov3_vith16plus": 1280,
    "dinov3_vit7b16": 1536,
    # DINOv3 ConvNeXt models
    "dinov3_convnext_tiny": 768,
    "dinov3_convnext_small": 768,
    "dinov3_convnext_base": 1024,
    "dinov3_convnext_large": 1536,
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
            model_type="dinov2",
            normalization=None,
            apply_image_mask=True
    ):
        super().__init__()
        self.model_name = model_name
        self.model = model
        self.token_name = token_name
        self.chunk_size = chunk_size
        self.model_type = model_type
        self.apply_image_mask: bool = apply_image_mask

        # Determine patch size based on model type
        if model_type == "dinov3":
            self.patch_size = 16 if "16" in model_name else 14
        else:
            self.patch_size = patch_size

        self.proposal_size = image_size
        self.descriptor_width_size = descriptor_width_size

        logging.info(f"Init CustomDINOv2 wrapper for {model_type} model: {model_name}")

        # Setup normalization parameters
        if normalization is None:
            # Default normalization (ImageNet/LVD-1689M for web images)
            norm_mean = (0.485, 0.456, 0.406)
            norm_std = (0.229, 0.224, 0.225)
        else:
            norm_mean = tuple(normalization.get("mean", [0.485, 0.456, 0.406]))
            norm_std = tuple(normalization.get("std", [0.229, 0.224, 0.225]))

        self.rgb_normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=norm_mean, std=norm_std),
            ]
        )

        # use for global feature
        self.rgb_proposal_processor = CropResizePad(self.proposal_size)
        self.rgb_resize = CustomResizeLongestSide(
            descriptor_width_size, dividable_size=self.patch_size
        )

        logging.info(
            f"Init {model_type} wrapper with full size={descriptor_width_size}, "
            f"proposal size={self.proposal_size}, patch size={self.patch_size} done!"
        )

    def process_rgb_proposals(self, image_np, masks, boxes):
        """
        1. Normalize image with DINOv2/DINOv3 transform
        2. Mask and crop each proposals
        3. Resize each proposals to predefined longest image size
        """
        num_proposals = len(masks)
        rgb = self.rgb_normalize(image_np).to(masks.device).float()
        rgbs = rgb.unsqueeze(0).repeat(num_proposals, 1, 1, 1)
        if self.apply_image_mask:
            masked_rgbs = rgbs * masks.unsqueeze(1)
        else:
            masked_rgbs = rgbs
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

    def get_detections_from_files(self, image_path: Path, segmentation_path: Path):
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        image_tensor = torch.from_numpy(image_array).to(self.device)
        segmentation = Image.open(segmentation_path).convert('L')
        segmentation_np = np.array(segmentation)
        segmentation_mask = torch.from_numpy(segmentation_np).unsqueeze(0).to(self.device)
        segmentation_mask = segmentation_mask.to(torch.float32).clamp(0, 1)  # From 0-255 to binary
        segmentation_bbox = masks_to_boxes(segmentation_mask)
        image_np = image_tensor.to(torch.uint8).numpy(force=True)
        detections = Detections({'masks': segmentation_mask, 'boxes': segmentation_bbox})
        dino_cls_descriptor, dino_dense_descriptor = self.forward(image_np, detections)

        return dino_cls_descriptor, dino_dense_descriptor


def descriptor_from_hydra(model='dinov3', mask_detections=True, device='cuda') -> CustomDINOv2:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    cfg_dir = (Path(__file__).parent.parent.parent / 'configs').resolve()
    with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
        overrides = [
            f'model/descriptor_model={model}',
            f'model.descriptor_model.apply_image_mask={mask_detections}',
        ]
        cnos_cfg = compose(config_name="run_inference", overrides=overrides)

    model_cfg = cnos_cfg.model.descriptor_model

    if _is_dinov3_with_local_weights(model_cfg):
        return _load_dinov3_with_local_weights(model_cfg, device)
    else:
        # Standard loading for DINOv2 or DINOv3 with URLs
        dino_descriptor: CustomDINOv2 = instantiate(model_cfg).to(device)
        dino_descriptor.model.device = device
        return dino_descriptor


def _is_dinov3_with_local_weights(model_cfg) -> bool:
    """Check if this is DINOv3 with local .pth file."""
    return (
            model_cfg.get('model_type') == 'dinov3' and
            'weights' in model_cfg and
            model_cfg.weights.endswith('.pth')
    )


def _load_dinov3_with_local_weights(model_cfg, device):
    """Load DINOv3: first load model without weights, then load state dict."""
    weights_path = model_cfg.weights
    logging.info(f"Loading DINOv3 with local weights: {weights_path}")

    # Remove weights from config and load model
    import copy
    cfg_no_weights = copy.deepcopy(model_cfg)
    del cfg_no_weights.weights
    dino_descriptor: CustomDINOv2 = instantiate(cfg_no_weights).to(device)

    # Load weights manually
    state_dict = torch.load(weights_path, map_location=device)
    dino_descriptor.model.load_state_dict(state_dict, strict=True)
    dino_descriptor.model.device = device

    return dino_descriptor
