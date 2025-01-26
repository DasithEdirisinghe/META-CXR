"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
from time import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop, transforms

from model.lavis.common.registry import registry
from model.lavis.models.base_model import all_gather_with_grad, concat_all_gather
from model.lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from model.lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures

from mhcac.mhcac_12 import AbnormalityClassificationModel


from vision_encoders.pubmedclip.pubmed_clip import Pubmedclip
from vision_encoders.medclip.medclip import Medclip

from mhcac.utils import compute_metrics_for_tasks
from mhcac.loss import ClassificationLoss
from mhcac.aggregator import Aggregator

chexpert_cols = ["No Finding", "Enlarged Cardiomediastinum",
                              "Cardiomegaly", "Lung Opacity",
                              "Lung Lesion", "Edema",
                              "Consolidation", "Pneumonia",
                              "Atelectasis", "Pneumothorax",
                              "Pleural Effusion", "Pleural Other",
                              "Fracture", "Support Devices"]

@registry.register_model("blip2")
@registry.register_model("blip2_feature_extractor")
class Blip2Qformer(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision, vis_num_feat = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        self.vit_model = vit_model
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, vis_num_feat, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len
        
        self.vis_augs = Compose([transforms.RandomAffine(degrees=30, shear=15),
                                        transforms.ColorJitter(brightness=0.2, contrast=0.2)])
        
        self.vis_transforms = Compose([CenterCrop(448)])
        
        
        self.pubmedclip = Pubmedclip(aug = self.vis_augs).eval()
        
        self.medclip = Medclip().eval()
        
        self.mhcac = AbnormalityClassificationModel(embed_dim=768, num_abnormalities=14, num_classes=3, num_layers=6, num_commmon_tokens = 14,initial_expert_tokens = None)
        
        class_weights = [
            torch.tensor([1.0, 1.0, 0.000], dtype=torch.float),  # Class weights for no finding
            torch.tensor([1.0, 10.0, 10.0], dtype=torch.float),  # Class weights for Enlarged Cardiomediastinum
            torch.tensor([1.0, 5.0, 10.0], dtype=torch.float),  # Class weights for Cardiomegaly
            torch.tensor([1.0, 4.0, 10.0], dtype=torch.float),  # Class weights for Lung Opacity
            torch.tensor([1.0, 5.0, 10.0], dtype=torch.float),  # Class weights for Lung Lesion
            torch.tensor([1.0, 5.0, 10.0], dtype=torch.float),   # Edema
            torch.tensor([1.0, 5.0, 10.0], dtype=torch.float),  # Consolidation
            torch.tensor([1.0, 10.0, 10.0], dtype=torch.float),   # Class weights for Pneumonia
            torch.tensor([1.0, 4.0, 10.0], dtype=torch.float),   # Class weights for Atelectasis 
            torch.tensor([1.0, 5.0, 10.0], dtype=torch.float),  #  Pneumothorax 
            torch.tensor([1.0, 4.0, 10.0], dtype=torch.float),  # Class weights for Pleural Effusion 
            torch.tensor([1.0, 10.0, 10.0], dtype=torch.float), # Class weights for Pleural Other 
            torch.tensor([1.0, 10.0, 10.0], dtype=torch.float),  # Fracture 
            torch.tensor([1.0, 3.0, 0.0], dtype=torch.float)  # Class weights for Support Devices
        ]  #negative(0),positive(1),uncertain(2)
        
        # class_weights = [
        #     torch.tensor([10.0, 10.0], dtype=torch.float),  # Class weights for no finding
        #     torch.tensor([1.0, 10.0], dtype=torch.float),  # Class weights for Enlarged Cardiomediastinum
        #     torch.tensor([2.0, 10.0], dtype=torch.float),  # Class weights for Cardiomegaly
        #     torch.tensor([4.0, 10.0], dtype=torch.float),  # Class weights for Lung Opacity
        #     torch.tensor([4.0, 10.0], dtype=torch.float),  # Class weights for Lung Lesion
        #     torch.tensor([1.0, 10.0], dtype=torch.float),   # Edema
        #     torch.tensor([1.0, 10.0], dtype=torch.float),  # Consolidation
        #     torch.tensor([1.0, 10.0], dtype=torch.float),   # Class weights for Pneumonia
        #     torch.tensor([5.0, 10.0], dtype=torch.float),   # Class weights for Atelectasis 
        #     torch.tensor([1.0, 10.0], dtype=torch.float),  #  Pneumothorax 
        #     torch.tensor([2.0, 10.0], dtype=torch.float),  # Class weights for Pleural Effusion 
        #     torch.tensor([5.0, 10.0], dtype=torch.float), # Class weights for Pleural Other 
        #     torch.tensor([2.0, 10.0], dtype=torch.float),  # Fracture 
        #     torch.tensor([5.0, 10.0], dtype=torch.float)  # Class weights for Support Devices
        # ]
        
        print(f"class weights are {class_weights}")

        """
        chexpert_cols = ["No Finding", "Enlarged Cardiomediastinum",
                              "Cardiomegaly", "Lung Opacity",
                              "Lung Lesion", "Edema",
                              "Consolidation", "Pneumonia",
                              "Atelectasis", "Pneumothorax",
                              "Pleural Effusion", "Pleural Other",
                              "Fracture", "Support Devices"]
                             
        """
        # Instantiate the loss function with abnormality-specific class weights
        self.cls_loss_fn = ClassificationLoss(penalty_weight=0.1, class_weights=class_weights, num_abnormalities=14)  #negative(0),positive(1),uncertain(2)
        

    def _create_mask(self, embeddings, mask_ratio=0.1):
        num_patches = embeddings.size(1)
        num_masked = int(mask_ratio * num_patches)
        
        # Create a mask of ones, then set a subset to zero
        mask = torch.ones(num_patches, device=embeddings.device)
        mask[:num_masked] = 0
        mask = mask[torch.randperm(num_patches)]  # Shuffle to randomize masked positions
        
        # Expand mask to match embeddings' dimensions and apply
        mask = mask.unsqueeze(0).expand(embeddings.size(0), -1)
        mask = mask.unsqueeze(-1)  # Add dimension for broadcasting
        return embeddings * mask  # Apply mask by element-wise multiplication
    
    def initialize_expert_tokens(self, chexpert_cols, embed_dim):
        # Initialize expert tokens based on text embeddings of abnormality names
        expert_embeddings = []
        for abnormality in chexpert_cols:
            # Get the text embedding (CLS token) for each abnormality
            text_tokens = self.tokenizer(
                abnormality,
                padding="max_length",
                truncation=True,
                max_length=20,  # Adjust max_length if necessary
                return_tensors="pt",
            ).to(next(self.parameters()).device)  # Move to device of the model
            
            text_output = self.Qformer.bert(
                text_tokens.input_ids,
                attention_mask=text_tokens.attention_mask,
                return_dict=True,
            )
            cls_embedding = text_output.last_hidden_state[:, 0, :]  # CLS token embedding
            expert_embeddings.append(cls_embedding)

        # Stack embeddings and return as initialized expert tokens
        embeddings = torch.cat(expert_embeddings, dim=0).reshape(len(chexpert_cols), embed_dim)
        torch.save(embeddings, "weights/expert_embeddings.pt")
        return embeddings
        
    
    def forward(self, samples):
        start_time = time()
        image = samples["image"]
        text = samples["text_output"]

        if self.vit_model == "biovil":
            image_bio = image
            # image_bio = self.vis_augs(image_bio)
            image_embeds = self.ln_vision(self.visual_encoder(image_bio).projected_patch_embeddings.reshape(image.shape[0], -1, 1408))
            # image_embeds = self.visual_encoder(image_bio).projected_patch_embeddings.reshape(image.shape[0], -1, 1408)
            # image_embeds = self._create_mask(image_embeds, mask_ratio=0.2)  # Mask 20% of patches

        else:
            image_bio = image
            # image_bio = self.vis_augs(image_bio)
            image_embeds = self.ln_vision(self.visual_encoder(image_bio))
            # image_embeds = self.visual_encoder(image_bio)
            # image_embeds = self._create_mask(image_embeds)  # Mask 20% of patches

        image_embeds_2 = self.pubmedclip(image, apply_aug = False)
        # image_embeds_2 = self._create_mask(image_embeds_2, mask_ratio=0.1)  # Mask 10% of patches

        # image_embeds_3 = self.ln_vision(self.medclip(image))
        # image_embeds_3 = self._create_mask(image_embeds_3)  # Mask 20% of patches

        
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        
        # query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        

        # query_output = self.Qformer.bert(
        #     query_embeds=query_tokens,
        #     encoder_hidden_states=image_embeds,
        #     encoder_attention_mask=image_atts,
        #     use_cache=True,
        #     return_dict=True,
        # )

        # cls_image_feat = F.normalize(
        #     self.vision_proj(query_output.last_hidden_state[:, 0, :]), dim=-1
        # )
        
        # image_feats = F.normalize(
        #     self.vision_proj(query_output.last_hidden_state), dim=-1
        # )

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        # text_feat = F.normalize(
        #     self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        # )
        
        # image_patches = self.image_embed_proj_norm(self.image_embed_proj(image_embeds))
        # txt_cls_token = self.text_cls_proj_norm(self.text_cls_proj(text_output.last_hidden_state[:, 0, :]))
        
        # image_patches_norm =  F.normalize(image_patches, dim=-1)
        # txt_cls_token_norm =  F.normalize(txt_cls_token, dim=-1)
        
        # Hook to log gradient norms to a file
        def log_grad_to_file(grad, filename="gradient_log.txt"):
            with open(filename, 'a') as f:
                f.write(f"Expert token gradient norm: {grad.norm().item()}\n")

        # Register the hook
        self.mhcac.expert_tokens.register_hook(lambda grad: log_grad_to_file(grad))
        
        ### ---- classification loss ----###
        
        cls_labels = samples["classification_labels"]  # Ground truth labels for abnormalities
        classification_logits, attention, contrastive_loss, orth_loss, sparsity_loss = self.mhcac(cnn_patches = image_embeds, vit_patches = image_embeds_2, text_embeddings = text_output.last_hidden_state ,labels = cls_labels)  # Output from your MHCAC module

        # classification_logits,vit_attention, cnn_attention = self.mhcac(cnn_patches = image_embeds, vit_patches = image_embeds_2, labels = None)  # Output from your MHCAC module

        # Compute abnormality-specific loss
        cls_loss = self.cls_loss_fn(classification_logits, cls_labels)
        
        metrics = compute_metrics_for_tasks(classification_logits, cls_labels)
        
         ###============== Image-text Contrastive for Claasifcation ===================###
        # sim_q2t = torch.matmul(
        #     image_patches_norm.unsqueeze(1), txt_cls_token_norm.unsqueeze(-1)
        # ).squeeze()

        # # image-text similarity: aggregate across all query tokens
        # sim_i2t, _ = sim_q2t.max(-1)
        # sim_i2t = sim_i2t / self.temp

        # # text-query similarity
        # sim_t2q = torch.matmul(
        #     txt_cls_token_norm.unsqueeze(1).unsqueeze(1), image_patches_norm.permute(0, 2, 1)
        # ).squeeze()

        # # text-image similarity: aggregate across all query tokens
        # sim_t2i, _ = sim_t2q.max(-1)
        # sim_t2i = sim_t2i / self.temp

        # bs = image.size(0)
        # targets = torch.arange(bs, dtype=torch.long).to(image.device)

        # loss_itc = (
        #                    F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
        #                    + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
                #    ) / 2
        
        
        ###============== Image-text Contrastive ===================###
        
        # # Compute the similarity between CLS image and CLS text tokens
        # sim_q2t = torch.matmul(cls_image_feat, cls_text_feat.T)  # [batch_size, batch_size]

        # # Apply temperature scaling
        # sim_q2t = sim_q2t / self.temp

        # # sim_i2t represents the similarity from image to text, sim_t2i from text to image
        # # These matrices should be symmetrical, but we'll compute both for clarity
        # sim_i2t = sim_q2t  # Image-to-Text similarity
        # sim_t2i = sim_q2t.T  # Text-to-Image similarity

        # # Create target indices for positive pairs
        # bs = image.size(0)
        # targets = torch.arange(bs, dtype=torch.long).to(image.device)

        # # Calculate Image-Text Contrastive Loss (ITC)
        # loss_itc = (
        #     F.cross_entropy(sim_i2t, targets)  # Image-to-Text contrastive loss
        #     + F.cross_entropy(sim_t2i, targets)  # Text-to-Image contrastive loss
        # ) / 2
        
        ###============== Image-text Contrastive ===================###
        # sim_q2t = torch.matmul(
        #     image_feats.unsqueeze(1), text_feat.unsqueeze(-1)
        # ).squeeze()

        # # image-text similarity: aggregate across all query tokens
        # sim_i2t, _ = sim_q2t.max(-1)
        # sim_i2t = sim_i2t / self.temp

        # # text-query similarity
        # sim_t2q = torch.matmul(
        #     text_feat.unsqueeze(1).unsqueeze(1), image_feats.permute(0, 2, 1)
        # ).squeeze()

        # # text-image similarity: aggregate across all query tokens
        # sim_t2i, _ = sim_t2q.max(-1)
        # sim_t2i = sim_t2i / self.temp

        # bs = image.size(0)
        # targets = torch.arange(bs, dtype=torch.long).to(image.device)

        # loss_itc = (
        #                    F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
        #                    + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        #            ) / 2
        
        ###============== Image-text Matching ===================###
        # with torch.no_grad():
        #     weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-4
        #     weights_t2i.fill_diagonal_(0)
        #     weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-4
        #     weights_i2t.fill_diagonal_(0)

        # # select a negative image for each text
        # image_embeds_neg = []
        # for b in range(bs):
        #     clamped_weight = torch.clamp(weights_t2i[b], min=1e-6)
        #     neg_idx = torch.multinomial(clamped_weight, 1).item()
        #     image_embeds_neg.append(image_embeds[neg_idx])
        # image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # # select a negative text for each image
        # text_ids_neg = []
        # text_atts_neg = []
        # for b in range(bs):
        #     clamped_weight = torch.clamp(weights_i2t[b], min=1e-6)
        #     neg_idx = torch.multinomial(weights_i2t[b], 1).item()
        #     text_ids_neg.append(text_tokens.input_ids[neg_idx])
        #     text_atts_neg.append(text_tokens.attention_mask[neg_idx])

        # text_ids_neg = torch.stack(text_ids_neg, dim=0)
        # text_atts_neg = torch.stack(text_atts_neg, dim=0)

        # text_ids_all = torch.cat(
        #     [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        # )  # pos, pos, neg
        # text_atts_all = torch.cat(
        #     [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
        #     dim=0,
        # )

        # query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        # query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
        #     image.device
        # )
        # attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        # image_embeds_all = torch.cat(
        #     [image_embeds, image_embeds_neg, image_embeds], dim=0
        # )  # pos, neg, pos
        # image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
        #     image.device
        # )

        # output_itm = self.Qformer.bert(
        #     text_ids_all,
        #     query_embeds=query_tokens_itm,
        #     attention_mask=attention_mask_all,
        #     encoder_hidden_states=image_embeds_all,
        #     encoder_attention_mask=image_atts_all,
        #     return_dict=True,
        # )

        # vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        # vl_output = self.itm_head(vl_embeddings)
        # logits = vl_output.mean(dim=1)

        # itm_labels = torch.cat(
        #     [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
        #     dim=0,
        # ).to(image.device)
        # loss_itm = F.cross_entropy(logits, itm_labels)

        # ##================= Image Captioning ========================##
        # decoder_input_ids = text_tokens.input_ids.clone()
        # decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        # labels = decoder_input_ids.masked_fill(
        #     decoder_input_ids == self.tokenizer.pad_token_id, -100
        # )

        # query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
        #     image.device
        # )
        # attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        # lm_output = self.Qformer(
        #     decoder_input_ids,
        #     attention_mask=attention_mask,
        #     past_key_values=query_output.past_key_values,
        #     return_dict=True,
        #     labels=labels,
        # )

        # loss_lm = lm_output.loss
        # # print(self.tokenizer.decode(torch.argmax(lm_output.logits, dim=-1)[0]))
        # loss_lm = 0.0
        end_time = time()
        # print(f"forward function took {end_time - start_time:.4f} seconds")
        
        return BlipOutput(
            loss = cls_loss + contrastive_loss * 0.3 + orth_loss * 0.7 + sparsity_loss * 0.3,
            # loss = cls_loss,
            loss_cls=cls_loss,
            loss_contrastive = contrastive_loss,
            loss_orthagonal = orth_loss,
            loss_sparsity = sparsity_loss,
            average_precision = metrics['average']['precision'],
            average_recall = metrics['average']['recall'],
            average_accuracy = metrics['average']['accuracy'],
            average_f1_score = metrics['average']['f1_score'],
        )
        
        # return BlipOutput(
        #     loss=loss_itm + loss_itc*1.5 + loss_lm,
        #     loss_itc=loss_itc*1.5,
        #     loss_itm=loss_itm,
        #     loss_lm=loss_lm,
        # )

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=1,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"].cuda()
        if self.vit_model == "biovil":
            image_embeds = self.ln_vision(self.visual_encoder(image).projected_patch_embeddings.reshape(image.shape[0], -1, 1408))
        else:
            image_embeds = self.ln_vision(self.visual_encoder(image))

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_image(self, image):
        if self.vit_model == "biovil":
            image_bio = image
            image_embeds = self.ln_vision(self.visual_encoder(image_bio).projected_patch_embeddings.reshape(image.shape[0], -1, 1408))
            # image_embeds = self.visual_encoder(image_bio).projected_patch_embeddings.reshape(image.shape[0], -1, 1408)
            # image_embeds = self._create_mask(image_embeds)  # Mask 20% of patches

        else:
            # image_bio = self.vis_transforms(image)
            image_bio = image
            image_embeds = self.ln_vision(self.visual_encoder(image_bio))
            # image_embeds = self._create_mask(image_embeds)  # Mask 20% of patches

        image_embeds_2 = self.pubmedclip(image,  apply_aug = False)
        # image_embeds_2 = self._create_mask(image_embeds_2)  # Mask 20% of patches

        # image_embeds_3 = self.ln_vision(self.medclip(image))
        # image_embeds_3 = self._create_mask(image_embeds_3)  # Mask 20% of patches

        # Concatenate the masked outputs
        # concat_image_embeds = torch.cat((image_embeds, image_embeds_2), dim=1)
        # image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
        #     image.device
        # )

        # query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        # query_output = self.Qformer.bert(
        #     query_embeds=query_tokens,
        #     encoder_hidden_states=image_embeds,
        #     encoder_attention_mask=image_atts,
        #     output_attentions=True,
        #     return_dict=True,
        # )
    
        
        # image_patches = self.image_embed_proj_norm(self.image_embed_proj(image_embeds))
        classification_logits, attention, contrastive_loss, orth_loss, sparsity_loss = self.mhcac(cnn_patches = image_embeds, vit_patches = image_embeds_2, text_embeddings = None ,labels = None)
        # txt_cls_token = text_output.last_hidden_state[:, 0, :]
        
        return classification_logits, attention


    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_output")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
