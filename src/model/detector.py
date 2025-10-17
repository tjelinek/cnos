import glob
import logging
import os
import os.path as osp
import time
from typing import Dict, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from tqdm import tqdm

from condensate_templates import TemplateBank
from src.model.loss import PairwiseSimilarity, compute_csls_terms
from src.model.utils import BatchedData, Detections, convert_npz_to_json
from src.utils.inout import save_json_bop23


def compute_templates_similarity_scores(template_data: TemplateBank, proposal_cls_descriptors: torch.Tensor,
                                        similarity_function: PairwiseSimilarity, aggregation_function: str,
                                        matching_confidence_thresh: float, matching_max_num_instances: int,
                                        use_per_template_confidence: bool = True, use_mahalanobis_dist: bool = False) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:

    db_descriptors = template_data.cls_desc
    sorted_obj_keys = sorted(db_descriptors.keys())

    similarities = {}

    rx, rt, splits = compute_csls_terms(proposal_cls_descriptors, db_descriptors)
    for i, obj_id in enumerate(sorted_obj_keys):
        obj_descriptor = db_descriptors[obj_id]

        rt_obj_id = rt[splits[i]:splits[i+1]]
        similarity = similarity_function(proposal_cls_descriptors, obj_descriptor, rx, rt_obj_id)
        similarities[obj_id] = similarity

    aggregated_similarities = {}
    aggregated_templates_ids = {}
    for obj_id in similarities.keys():
        if aggregation_function == "mean":
            # N_proposals x N_objects
            score_per_proposal = (torch.sum(similarities[obj_id], dim=-1) / similarities[obj_id].shape[-1])
            proposal_indices = torch.arange(similarities[obj_id].shape[1], device=similarities[obj_id].device)
        elif aggregation_function == "median":
            score_per_proposal, proposal_indices = torch.median(similarities[obj_id], dim=-1)
        elif aggregation_function == "max":
            score_per_proposal, proposal_indices = torch.max(similarities[obj_id], dim=-1)
        elif aggregation_function == "avg_5":
            k = min(similarities[obj_id].shape[-1], 5)
            score_per_proposal, proposal_indices = torch.topk(similarities[obj_id], k=k, dim=-1)
            score_per_proposal = torch.mean(score_per_proposal, dim=-1)
        else:
            raise ValueError("Unknown aggregation function")

        aggregated_similarities[obj_id] = score_per_proposal
        aggregated_templates_ids[obj_id] = proposal_indices

    score_per_proposal_and_object = torch.stack([aggregated_similarities[k] for k in sorted_obj_keys], dim=-1)
    aggregated_templates_ids = torch.stack([aggregated_templates_ids[k] for k in sorted_obj_keys], dim=-1)

    # assign each proposal to the object with the highest scores
    score_per_proposal, proposal_assigned_object_id = torch.max(score_per_proposal_and_object, dim=-1)

    #
    idx_selected_proposals = filter_proposals(aggregated_templates_ids, proposal_assigned_object_id, score_per_proposal,
                                              sorted_obj_keys, '', template_data, matching_confidence_thresh)
    # for bop challenge, we only keep top 100 instances
    if len(idx_selected_proposals) > matching_max_num_instances:
        logging.info(f"Selecting top {matching_max_num_instances} instances ...")
        _, idx = torch.topk(
            score_per_proposal[idx_selected_proposals], k=matching_max_num_instances
        )
        idx_selected_proposals = idx_selected_proposals[idx]
    pred_idx_objects = proposal_assigned_object_id[idx_selected_proposals]
    pred_scores = score_per_proposal[idx_selected_proposals]
    pred_score_distribution = score_per_proposal_and_object[idx_selected_proposals]

    filter_similarities_dict(similarities, idx_selected_proposals)

    sorted_db_keys_tensor = torch.tensor(sorted_obj_keys).to(pred_idx_objects.device)
    selected_objects = sorted_db_keys_tensor[pred_idx_objects]

    return idx_selected_proposals, selected_objects, pred_scores, pred_score_distribution, similarities


def filter_similarities_dict(similarities, idx_selected_proposals):
    for obj_id in similarities.keys():
        similarities[obj_id] = similarities[obj_id][idx_selected_proposals]


def filter_proposals(proposals_assigned_templates_ids: torch.Tensor, proposals_assigned_object_ids: torch.Tensor,
                     cosine_similarity_per_proposal: torch.Tensor, sorted_obj_keys: list[int],
                     ood_detection_method: str, template_data: TemplateBank = None,
                     global_similarity_threshold: float = None) -> torch.Tensor:
    device = cosine_similarity_per_proposal.device
    idx_proposals = torch.arange(len(cosine_similarity_per_proposal), device=device)

    if ood_detection_method == 'cosine_similarity_quantiles':
        num_proposals = proposals_assigned_object_ids.shape[0]
        assigned_template_id = \
            proposals_assigned_templates_ids[torch.arange(num_proposals, device=device), proposals_assigned_object_ids]

        template_thresholds = template_data.template_thresholds
        thresholds_for_selected_objs = []
        for det_id, obj_id in enumerate(proposals_assigned_object_ids):
            threshold = template_thresholds[sorted_obj_keys[obj_id]][assigned_template_id[det_id]]
            thresholds_for_selected_objs.append(threshold)

        thresholds_for_selected_objs = torch.stack(thresholds_for_selected_objs)

        idx_selected_proposals = idx_proposals[cosine_similarity_per_proposal > thresholds_for_selected_objs]
    elif ood_detection_method == 'global_threshold':
        assert global_similarity_threshold is not None
        idx_selected_proposals = idx_proposals[cosine_similarity_per_proposal > global_similarity_threshold]
    elif ood_detection_method == 'lowe_test':
        raise NotImplementedError()
    elif ood_detection_method == 'mahalanobis_ood_detection':
        raise NotImplementedError()
    elif ood_detection_method == 'none':
        idx_selected_proposals = idx_proposals  # Keep them as they are
    else:
        raise ValueError(f'Unknown OOD detection method {ood_detection_method}')

    return idx_selected_proposals


class CNOS(pl.LightningModule):
    def __init__(
        self,
        segmentor_model,
        descriptor_model,
        onboarding_config,
        matching_config,
        post_processing_config,
        log_interval,
        log_dir,
        save_mask,
        **kwargs,
    ):
        # define the network
        super().__init__()
        self.segmentor_model = segmentor_model
        self.descriptor_model = descriptor_model

        self.onboarding_config = onboarding_config
        self.matching_config = matching_config
        self.post_processing_config = post_processing_config
        self.log_interval = log_interval
        self.log_dir = log_dir
        self.save_mask = save_mask

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(osp.join(self.log_dir, "predictions"), exist_ok=True)
        self.inv_rgb_transform = T.Compose(
            [
                T.Normalize(
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                ),
            ]
        )
        logging.info(f"Init CNOS done!")

    def set_reference_objects(self):
        os.makedirs(
            osp.join(self.log_dir, f"predictions/{self.dataset_name}"), exist_ok=True
        )
        logging.info("Initializing reference objects ...")

        start_time = time.time()
        self.ref_data = {"descriptors": BatchedData(None)}
        descriptors_path = osp.join(self.ref_dataset.template_dir, "descriptors.pth")
        if self.onboarding_config.rendering_type == "pbr":
            descriptors_path = descriptors_path.replace(".pth", "_pbr.pth")
        if (
            os.path.exists(descriptors_path)
            and not self.onboarding_config.reset_descriptors
        ):
            self.ref_data["descriptors"] = torch.load(descriptors_path).to(self.device)
        else:
            for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Computing descriptors ...",
            ):
                ref_imgs = self.ref_dataset[idx]["templates"].to(self.device)
                ref_feats, patch_feats = self.descriptor_model.compute_features(
                    ref_imgs, token_name="x_norm_clstoken"
                )
                self.ref_data["descriptors"].append(ref_feats)

            self.ref_data["descriptors"].stack()  # N_objects x descriptor_size
            self.ref_data["descriptors"] = self.ref_data["descriptors"].data

            # save the precomputed features for future use
            torch.save(self.ref_data["descriptors"], descriptors_path)

        end_time = time.time()
        logging.info(
            f"Runtime: {end_time-start_time:.02f}s, Descriptors shape: {self.ref_data['descriptors'].shape}"
        )

    def move_to_device(self):
        self.descriptor_model.model = self.descriptor_model.model.to(self.device)
        self.descriptor_model.model.device = self.device

        # Move segmentor model to device - handle both SAM1 and SAM2
        if hasattr(self.segmentor_model, "predictor"):
            # SAM1 path - has predictor attribute
            self.segmentor_model.predictor.model = (
                self.segmentor_model.predictor.model.to(self.device)
            )
        elif hasattr(self.segmentor_model.model, "setup_model"):
            # SAM1 path - has setup_model method
            self.segmentor_model.model.setup_model(device=self.device, verbose=True)
        else:
            # SAM2 path - move the underlying sam model to device
            self.segmentor_model.sam = self.segmentor_model.sam.to(self.device)
            logging.info("Loaded checkpoint sucessfully")

        logging.info(f"Moving models to {self.device} done!")

    def find_matched_proposals(self, proposal_decriptors):
        aggregation_function = self.matching_config.aggregation_function
        matching_max_num_instances = self.matching_config.max_num_instances
        matching_confidence_thresh = self.matching_config.confidence_thresh
        descriptors = self.ref_data["descriptors"]
        similarity_function: PairwiseSimilarity = self.matching_config.metric

        # compute matching scores for each proposals
        scores = similarity_function(proposal_decriptors, descriptors)  # N_proposals x N_objects x N_templates
        if aggregation_function == "mean":
            score_per_proposal_and_object = (
                torch.sum(scores, dim=-1) / scores.shape[-1]
            )  # N_proposals x N_objects
        elif aggregation_function == "median":
            score_per_proposal_and_object = torch.median(scores, dim=-1)[0]
        elif aggregation_function == "max":
            score_per_proposal_and_object = torch.max(scores, dim=-1)[0]
        elif aggregation_function == "avg_5":
            score_per_proposal_and_object = torch.topk(scores, k=5, dim=-1)[0]
            score_per_proposal_and_object = torch.mean(
                score_per_proposal_and_object, dim=-1
            )
        else:
            raise NotImplementedError

        # assign each proposal to the object with the highest scores
        score_per_proposal, assigned_idx_object = torch.max(
            score_per_proposal_and_object, dim=-1
        )  # N_query

        idx_selected_proposals = torch.arange(
            len(score_per_proposal), device=score_per_proposal.device
        )[score_per_proposal > self.matching_config.confidence_thresh]
        # for bop challenge, we only keep top 100 instances
        if len(idx_selected_proposals) > self.matching_config.max_num_instances:
            logging.info(f"Selecting top {self.matching_config.max_num_instances} instances ...")
            _, idx = torch.topk(
                score_per_proposal[idx_selected_proposals], k=self.matching_config.max_num_instances
            )
            idx_selected_proposals = idx_selected_proposals[idx]
        pred_idx_objects = assigned_idx_object[idx_selected_proposals]
        pred_scores = score_per_proposal[idx_selected_proposals]
        pred_score_distribution = score_per_proposal_and_object[idx_selected_proposals]
        return idx_selected_proposals, pred_idx_objects, pred_scores, pred_score_distribution

    def test_step(self, batch, idx):
        if idx == 0:
            os.makedirs(
                osp.join(
                    self.log_dir,
                    f"predictions/{self.dataset_name}/{self.name_prediction_file}",
                ),
                exist_ok=True,
            )
            self.set_reference_objects()
            self.move_to_device()
        assert batch["image"].shape[0] == 1, "Batch size must be 1"

        image_np = (
            self.inv_rgb_transform(batch["image"][0])
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )
        image_np = np.uint8(image_np.clip(0, 1) * 255)

        # run propoals
        proposal_stage_start_time = time.time()

        detections = self.get_filtered_detections(image_np)

        # compute descriptors
        query_decriptors = self.descriptor_model(image_np, detections)[0]
        proposal_stage_end_time = time.time()

        # matching descriptors
        matching_stage_start_time = time.time()
        (
            idx_selected_proposals,
            pred_idx_objects,
            pred_scores,
            pred_score_distribution,
        ) = self.find_matched_proposals(query_decriptors)

        # update detections
        detections.filter(idx_selected_proposals)
        detections.add_attribute("scores", pred_scores)
        detections.add_attribute("score_distribution", pred_score_distribution)
        detections.add_attribute("object_ids", pred_idx_objects)
        detections.apply_nms_per_object_id(
            nms_thresh=self.post_processing_config.nms_thresh
        )
        matching_stage_end_time = time.time()

        runtime = (
            proposal_stage_end_time
            - proposal_stage_start_time
            + matching_stage_end_time
            - matching_stage_start_time
        )
        detections.to_numpy()

        scene_id = batch["scene_id"][0]
        frame_id = batch["frame_id"][0]
        file_path = osp.join(
            self.log_dir,
            f"predictions/{self.dataset_name}/{self.name_prediction_file}/scene{scene_id}_frame{frame_id}",
        )

        # save detections to file
        results = detections.save_to_file(
            scene_id=int(scene_id),
            frame_id=int(frame_id),
            runtime=runtime,
            file_path=file_path,
            dataset_name=self.dataset_name,
            return_results=True,
            save_mask=self.save_mask,
            save_score_distribution=True,
        )
        # save runtime to file
        np.savez(
            file_path + "_runtime",
            proposal_stage=proposal_stage_end_time - proposal_stage_start_time,
            matching_stage=matching_stage_end_time - matching_stage_start_time,
        )
        return 0

    def get_filtered_detections(self, image_np: np.ndarray) -> Detections:
        proposals = self.segmentor_model.generate_masks(image_np)
        # init detections with masks and boxes
        detections = Detections(proposals)
        detections.remove_very_small_detections(
            config=self.post_processing_config.mask_post_processing
        )
        return detections

    def on_test_epoch_end(self):
        if self.global_rank == 0:  # only rank 0 process
            # can use self.all_gather to gather results from all processes
            # but it is simpler just load the results from files so no file is missing
            result_paths = sorted(
                glob.glob(
                    osp.join(
                        self.log_dir,
                        f"predictions/{self.dataset_name}/{self.name_prediction_file}/*.npz",
                    )
                )
            )
            result_paths = sorted(
                [path for path in result_paths if "runtime" not in path]
            )

            logging.info(f"Converting {len(result_paths)} npz files to json ...")
            detections = []
            for idx in tqdm(range(len(result_paths)), desc="Converting npz to json"):
                results = convert_npz_to_json(idx, result_paths)
                detections.append(results)
            formatted_detections = []

            formatted_detections_with_score_distribution = []
            for detection in tqdm(detections, desc="Loading results ..."):
                formatted_detections.extend(detection[0])
                formatted_detections_with_score_distribution.extend(detection[1])

            detections_path = f"{self.log_dir}/{self.name_prediction_file}.json"
            save_json_bop23(detections_path, formatted_detections)
            logging.info(f"Saved predictions (BOP format) to {detections_path}")

            detections_path = f"{self.log_dir}/{self.name_prediction_file}_with_score_distribution.json"
            save_json_bop23(detections_path, formatted_detections_with_score_distribution)
            logging.info(f"Saved predictions (BOP format + score distribution) to {detections_path} ")
