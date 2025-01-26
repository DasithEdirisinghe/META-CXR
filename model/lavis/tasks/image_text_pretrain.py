"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from model.lavis.common.registry import registry
from model.lavis.tasks.base_task import BaseTask
from model.lavis.datasets.data_utils import move_to_cuda


@registry.register_task("image_text_pretrain_eval")
class ImageTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        loss = 0.0
        precision = 0.0
        recall = 0.0
        f1_score = 0.0
        accuracy = 0.0
        dataloader_len = len(data_loader)
        for batch in data_loader:
            if cuda_enabled:
                batch = move_to_cuda(batch)
            loss_dict = model(batch)
            loss += loss_dict["loss"].item()
            
            if "average_precision" in loss_dict:
                precision += loss_dict["average_precision"].item()
            if "average_recall" in loss_dict:
                recall += loss_dict["average_recall"].item()
            if "average_f1_score" in loss_dict:
                f1_score += loss_dict["average_f1_score"].item()
            if "average_accuracy" in loss_dict:
                accuracy += loss_dict["average_accuracy"].item()
        
            
        print(f"Average Precision: {precision/dataloader_len} | Average Recall: {recall/dataloader_len} | Average f1 score: {f1_score/dataloader_len} | Average Accuracy: {accuracy/dataloader_len}")
        return loss / dataloader_len
