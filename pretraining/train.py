"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random
import json

import numpy as np
import pickle
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import wandb
import pandas as pd

from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from med_cxr.utils import save_to_csv, compute_metrics_for_tasks, aggregate_results, query_attention_visualization, expert_atttention_visualization, visualize_images_with_labels
import model.lavis.tasks as tasks
from model.lavis.common.config import Config
from model.lavis.common.dist_utils import get_rank
from model.lavis.common.logger import setup_logger

from local_config import WANDB_ENTITY
from model.lavis.common.registry import registry
from model.lavis.common.utils import now

# imports modules for registration
from model.lavis.common.optims import (
   LinearWarmupCosineLRScheduler,
   LinearWarmupStepLRScheduler,
)
from model.lavis.datasets.builders import *
from model.lavis.models import *
from model.lavis.processors import *
from model.lavis.runners import *
from model.lavis.tasks import *
from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset, CheXpertDataset, IU_Xray_Dataset
from local_config import PATH_TO_MIMIC_CXR


# python -m pretraining.train --cfg-path pretraining/configs/blip2_pretrain_stage1_emb.yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    registry.mapping['paths']['cache_root'] = '.'
    cfg = Config(parse_args())

    job_id = now()

    # init_distributed_mode(cfg)
    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    wandb_run = wandb.init(
        project=cfg.run_cfg.project_name,
        entity=WANDB_ENTITY,
        name=cfg.run_cfg.run_name
    )

    cfg.pretty_print()

    task = tasks.setup_task(cfg)

    # my report dataset
    datasets = {}
    datasets['mimic_cxr'] = {}
    datasets['chexpert'] = {}
    datasets['iu_xray'] = {}

    datasets['mimic_cxr']['train'] = MIMIC_CXR_Dataset(vis_processor=None, text_processor=None, vis_root=f"{PATH_TO_MIMIC_CXR}/mimic-cxr-jpg/2.1.0",
                                                       split="train", cfg=cfg, truncate=None)
    # datasets['mimic_cxr']['train_val'] = MIMIC_CXR_Dataset(vis_processor=None, text_processor=None,
    #                                                        vis_root=f"{PATH_TO_MIMIC_CXR}/mimic-cxr-jpg/2.1.0", split="train", cfg=cfg,
    #                                                        truncate=1000)  # 1000
    # datasets['mimic_cxr']['val'] = MIMIC_CXR_Dataset(vis_processor=None, text_processor=None, vis_root=f"{PATH_TO_MIMIC_CXR}/mimic-cxr-jpg/2.1.0",
    #                                                  split="validate", cfg=cfg, truncate=None)
    # datasets['mimic_cxr']['test'] = MIMIC_CXR_Dataset(vis_processor=None, text_processor=None, vis_root=f"{PATH_TO_MIMIC_CXR}/mimic-cxr-jpg/2.1.0",
    #                                                  split="test", cfg=cfg, truncate=None)

    # datasets['chexpert']['val'] = CheXpertDataset(vis_processor=None, text_processor=None, vis_root="meta-cxr", split="val", cfg=cfg, truncate=None)

    # datasets['iu_xray']['test'] = IU_Xray_Dataset(vis_processor=None, text_processor=None, vis_root="iu-xray", split="test", cfg=cfg, truncate=None)

    model = task.build_model(cfg)
    # print(summary(model, input_size=None, device='cpu'))


    if not cfg.run_cfg.evaluate:
        ''' training code '''
        runner = RunnerBase(
            cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
        )

        runner.train(wandb_run)


    else:
        ''' precompute Q-Former output embeddings for all images '''
        model.cuda()
        model.eval()

        # dataloader = DataLoader(datasets['mimic_cxr']['train'], batch_size=2, shuffle=False, num_workers=cfg.run_cfg.num_workers)
        # embeddings = {}
        # for i, batch in enumerate(tqdm(dataloader)):
        #     qformer_embs, _, cls_logits = model.forward_image(batch['image'].cuda())
        #     cls_labels = batch['classification_labels']
        #     dicom_id = batch['dicom_id']
    
        #     save_to_csv(cls_logits, cls_labels, dicom_id, file_name = f"pretraining/cls/predictions_{cfg.run_cfg.run_name}_train.csv")

        # # save embeddings
        # with open(f"pretraining/embs/{cfg.run_cfg.run_name}_embeddings_test.pkl", "wb") as f:
        #     pickle.dump(embeddings, f)

        split = 'train'
        dataset = 'mimic_cxr'
        batch_size = 64
        dataloader = DataLoader(datasets[dataset][split], batch_size=batch_size, shuffle=False, num_workers=cfg.run_cfg.num_workers)
        embeddings = {}
        cls_logits_dict = {}

        metrics_data = {}
        precision = 0.0
        recall = 0.0
        f1_score = 0.0
        accuracy = 0.0
        dataloader_len = len(dataloader)
        for i, batch in enumerate(tqdm(dataloader)):
            print(batch)
            # cls_logits, vit_attention, cnn_attention = model.forward_image(batch['image'].cuda())
            qformer_embs, _, cls_logits, attention_weights = model.forward_image(batch['image'].cuda(), None)

            for j, id in enumerate(batch['image_id']):
                if dataset == 'mimic_cxr':
                    dicom = datasets['mimic_cxr'][split].id_to_dicom[id.item()]
                elif dataset == 'iu_xray':
                    dicom = datasets['iu_xray'][split].id_to_img[id.item()]
                embeddings[dicom] = qformer_embs[j].cpu().detach().numpy()
                cls_logits_dict[dicom] = cls_logits[j].cpu().detach().numpy()

        # save cls_logits
        with open(f"pretraining/cls/{cfg.run_cfg.run_name}_cls_logits_{dataset}_{split}.pkl", "wb") as f:
            pickle.dump(cls_logits_dict, f)

        # save embeddings
        with open(f"pretraining/embs/{cfg.run_cfg.run_name}_embeddings_{dataset}_{split}.pkl", "wb") as f:
            pickle.dump(embeddings, f)
        
        with open(f"pretraining/cls/{cfg.run_cfg.run_name}_meta_{dataset}_{split}.json", "w") as f:
            json.dump({"dataset": dataset, "split": split, "batch_size": batch_size, "model": cfg.run_cfg.run_name}, f)
            
            #######################################

            # generated_caption = model.generate(batch)
            # cls_labels = batch['classification_labels']
            # # dicom_id = batch['dicom_id']
            # image_path = batch['image_path']
            # # text_output = batch['text_output']
            
            # metrics = compute_metrics_for_tasks(cls_logits, cls_labels)
            # metrics_data[i] = metrics

            # batch_precision = metrics['average']['precision'].item()
            # batch_recall = metrics['average']['recall'].item()
            # batch_f1_score = metrics['average']['f1_score'].item()
            # batch_accuracy = metrics['average']['accuracy'].item()
            
            # precision += batch_precision
            # recall += batch_recall
            # f1_score += batch_f1_score
            # accuracy += batch_accuracy

            ################################
            
            # print(type(vit_attention))
            # print(len(vit_attention))
            # print(vit_attention[0].shape)
            
            ### this is only for mhcac layer ###
            # vit_attention = torch.stack(vit_attention, dim=1).squeeze(2).mean(dim=2)
            # cnn_attention = torch.stack(cnn_attention, dim=1).squeeze(2).mean(dim=2)
            
            # print(attention_weights[-1].shape)
            
            #print(cnn_attention.shape)
            # for i in range(len(qformer_attn)):
            #     print(type(qformer_attn[i]))
                # print(qformer_attn[i].shape)
            # print(qformer_attn)

            # query_attention_visualization(batch['image'], qformer_attn[-2])
            # expert_atttention_visualization(batch['image'], cls_labels, i,  attention_weights[-1])
            
            # save_to_csv(cls_logits, cls_labels, dicom_id, file_name = f"pretraining/cls/predictions_{cfg.run_cfg.run_name}_test.csv")
            
            # visualize_images_with_labels(batch['image'], cls_labels, image_path, dicom_id = dicom_id, prefix = str(i), text_output = text_output, generated_caption= generated_caption)
            
            # if i == 300:
            #     break

        # aggregate_results(metrics_data)
            
        
        # print(f"Average Precision: {precision/dataloader_len} | Average Recall: {recall/dataloader_len} | Average f1 score: {f1_score/dataloader_len} | Average Accuracy: {accuracy/dataloader_len}")
        # # save embeddings
        # with open(f"pretraining/embs/{cfg.run_cfg.run_name}_embeddings_val.pkl", "wb") as f:
        #     pickle.dump(embeddings, f)

        # dataloader = DataLoader(datasets['mimic_cxr']['train'], batch_size=2, shuffle=False, num_workers=cfg.run_cfg.num_workers)
        # embeddings = {}
        # for i, batch in enumerate(tqdm(dataloader)):
        #     qformer_embs, _, cls_logits = model.forward_image(batch['image'].cuda())
        #     for j, id in enumerate(batch['image_id']):
        #         dicom = datasets['mimic_cxr']['train'].id_to_dicom[id.item()]
        #         embeddings[dicom] = qformer_embs[j].cpu().detach().numpy()

        # # save embeddings
        # with open(f"pretraining/embs/{cfg.run_cfg.run_name}_embeddings_train.pkl", "wb") as f:
        #     pickle.dump(embeddings, f)
    

if __name__ == "__main__":
    main()
