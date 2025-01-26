import os
import json

import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, auc
import pandas as pd
import numpy as np
import textwrap


# def compute_metrics_for_tasks(logits, labels):
#     metrics = {}
#     total_precision = 0
#     total_recall = 0
#     total_f1 = 0
#     total_accuracy = 0
    
#     task_names = get_task_list()
#     num_tasks = len(task_names)
#     probabilities = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
#     predicted_labels = torch.argmax(probabilities, dim=-1)  # Get predicted classes

#     for i, task_name in enumerate(task_names):
#         task_preds = predicted_labels[:, i].detach().cpu().numpy()
#         task_labels = labels[:, i].detach().cpu().numpy()

#         precision = precision_score(task_labels, task_preds, average='weighted', zero_division=1)
#         recall = recall_score(task_labels, task_preds, average='weighted', zero_division=1)
#         f1 = f1_score(task_labels, task_preds, average='weighted', zero_division=1)
#         accuracy = accuracy_score(task_labels, task_preds)

#         precision_tensor = torch.tensor(precision, dtype=torch.float)
#         recall_tensor = torch.tensor(recall, dtype=torch.float)
#         f1_tensor = torch.tensor(f1, dtype=torch.float)
#         accuracy_tensor = torch.tensor(accuracy, dtype=torch.float)

#         total_precision += precision_tensor.item()
#         total_recall += recall_tensor.item()
#         total_f1 += f1_tensor.item()
#         total_accuracy += accuracy_tensor.item()
        
#         # Compute AUC-ROC and PR-AUC
#         n_classes = logits.size(-1)
#         auc_roc_dict = {}
#         pr_score_dict = {}
#         for class_idx in range(n_classes):
#             binary_labels = (task_labels == class_idx).astype(int)
#             class_probs = probabilities[:, i, class_idx].detach().cpu().numpy()
#             valid_mask = (~np.isnan(binary_labels) & ~np.isnan(class_probs) & np.isfinite(binary_labels) & np.isfinite(class_probs))
#             binary_labels = binary_labels[valid_mask]
#             class_probs = class_probs[valid_mask]

#             if len(set(binary_labels)) > 1:
#                 try:
#                     fpr, tpr, _ = roc_curve(binary_labels, class_probs)
#                     auc_roc = roc_auc_score(binary_labels, class_probs)
#                     precision_, recall_, _ = precision_recall_curve(binary_labels, class_probs)
#                     pr_auc = auc(recall_, precision_)
#                 except ValueError:
#                     fpr, tpr, precision_, recall_ = None, None, None, None
#                     auc_roc = None
#                     pr_auc = None
#             else:
#                 fpr, tpr, precision_, recall_ = None, None, None, None
#                 auc_roc = None
#                 pr_auc = None
            
            
#             auc_roc_dict[get_class_map(class_idx)] = torch.tensor(auc_roc if auc_roc is not None else 0.0, dtype=torch.float)
#             pr_score_dict[get_class_map(class_idx)] = torch.tensor(pr_auc if pr_auc is not None else 0.0, dtype=torch.float)
            

#         metrics[task_name] = {
#             'precision': precision_tensor,
#             'recall': recall_tensor,
#             'f1_score': f1_tensor,
#             'accuracy': accuracy_tensor,
#             'auc_roc': auc_roc_dict,
#             'pr_auc': pr_score_dict
#         }
    
#     average_precision = torch.tensor(total_precision / num_tasks, dtype=torch.float)
#     average_recall = torch.tensor(total_recall / num_tasks, dtype=torch.float)
#     average_f1 = torch.tensor(total_f1 / num_tasks, dtype=torch.float)
#     average_accuracy = torch.tensor(total_accuracy / num_tasks, dtype=torch.float)

#     metrics['average'] = {
#         'precision': average_precision,
#         'recall': average_recall,
#         'f1_score': average_f1,
#         'accuracy': average_accuracy
#     }

#     return metrics


# def aggregate_results(metrics_data):
#     precision = 0
#     recall = 0
#     f1_score = 0
#     accuracy = 0
    
#     task_data = {} 
#     data_len = len(metrics_data)
    
#     task_list = get_task_list()
    
#     print("aggregate result invoked")
    
#     for batch_id, metrics in metrics_data.items():
    
#         batch_precision = metrics['average']['precision'].item()
#         batch_recall = metrics['average']['recall'].item()
#         batch_f1_score = metrics['average']['f1_score'].item()
#         batch_accuracy = metrics['average']['accuracy'].item()
        
#         precision += batch_precision
#         recall += batch_recall
#         f1_score += batch_f1_score
#         accuracy += batch_accuracy
        
#         for task, data in metrics.items():
#             if task in task_list:
#                 if task not in task_data:
#                     task_data[task] = {}  # Initialize inner dict if not present
                
#                 for key, value in data.items():  # Corrected here
#                     if key not in task_data[task]:
#                         task_data[task][key] = value
#                     else:
#                         if key not in ['auc_roc', 'pr_auc']:
#                             task_data[task][key] += value
#                         else:
#                             if key not in task_data[task]:
#                                 task_data[task][key] = {}
#                             for class_name, auc in value.items():
#                                 # Initialize the class_name in the auc_roc dictionary
#                                 if class_name not in task_data[task][key]:
#                                     task_data[task][key][class_name] = auc.clone()  # Initialize with auc Tensor
#                                 else:
#                                     task_data[task][key][class_name] += auc
    
#     for task, data in task_data.items():  # Corrected here
#         for key, value in data.items():  # Corrected here
#             if  key not in ['auc_roc', 'pr_auc']:
#                 data[key] = value / data_len
#             else:
#                 for class_name, auc in value.items():  # Corrected here
#                     data[key][class_name] = auc / data_len
        
#         auc_roc_output = " | ".join([f"{class_name}: {data['auc_roc'][class_name]}" for class_name in data['auc_roc']])
#         pr_auc_output = " | ".join([f"{class_name}: {data['pr_auc'][class_name]}" for class_name in data['pr_auc']])
        
#         print(f"Task: {task} | Average Precision: {data['precision']} | Average Recall: {data['recall']} | Average f1 score: {data['f1_score']} \n")

#         print(f"Task: {task} | Average AUC-ROC | {auc_roc_output} \n")
#         print(f"Task: {task} | Average PR-AUC | {pr_auc_output}\n")
    
#     print(f"Average Precision: {precision / data_len} | Average Recall: {recall / data_len} | Average f1 score: {f1_score / data_len} | Average Accuracy: {accuracy / data_len}")


def compute_metrics_for_tasks(logits, labels):
    metrics = {}
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_accuracy = 0
    
    task_names = get_task_list()
    num_tasks = len(task_names)
    probabilities = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
    predicted_labels = torch.argmax(probabilities, dim=-1)  # Get predicted classes

    for i, task_name in enumerate(task_names):
        task_preds = predicted_labels[:, i].detach().cpu().numpy()
        task_labels = labels[:, i].detach().cpu().numpy()

        precision = precision_score(task_labels, task_preds, average='weighted', zero_division=1)
        recall = recall_score(task_labels, task_preds, average='weighted', zero_division=1)
        f1 = f1_score(task_labels, task_preds, average='weighted', zero_division=1)
        accuracy = accuracy_score(task_labels, task_preds)

        precision_tensor = torch.tensor(precision, dtype=torch.float)
        recall_tensor = torch.tensor(recall, dtype=torch.float)
        f1_tensor = torch.tensor(f1, dtype=torch.float)
        accuracy_tensor = torch.tensor(accuracy, dtype=torch.float)

        total_precision += precision_tensor.item()
        total_recall += recall_tensor.item()
        total_f1 += f1_tensor.item()
        total_accuracy += accuracy_tensor.item()
        
        # Compute probabilities and raw labels for each class
        n_classes = logits.size(-1)
        auc_roc_dict = {}
        pr_score_dict = {}
        for class_idx in range(n_classes):
            binary_labels = (task_labels == class_idx).astype(int)
            class_probs = probabilities[:, i, class_idx].detach().cpu().numpy()
            valid_mask = (~np.isnan(binary_labels) & ~np.isnan(class_probs) &
                          np.isfinite(binary_labels) & np.isfinite(class_probs))
            binary_labels = binary_labels[valid_mask]
            class_probs = class_probs[valid_mask]

            # Store raw probabilities and labels for aggregation
            auc_roc_dict[get_class_map(class_idx)] = {
                'probs': class_probs.tolist(),  # Store probabilities
                'labels': binary_labels.tolist()  # Store labels
            }
            pr_score_dict[get_class_map(class_idx)] = auc_roc_dict[get_class_map(class_idx)]  # Same for PR

        metrics[task_name] = {
            'precision': precision_tensor,
            'recall': recall_tensor,
            'f1_score': f1_tensor,
            'accuracy': accuracy_tensor,
            'auc_roc': auc_roc_dict
        }
    
    average_precision = torch.tensor(total_precision / num_tasks, dtype=torch.float)
    average_recall = torch.tensor(total_recall / num_tasks, dtype=torch.float)
    average_f1 = torch.tensor(total_f1 / num_tasks, dtype=torch.float)
    average_accuracy = torch.tensor(total_accuracy / num_tasks, dtype=torch.float)

    metrics['average'] = {
        'precision': average_precision,
        'recall': average_recall,
        'f1_score': average_f1,
        'accuracy': average_accuracy
    }

    return metrics



def aggregate_results(metrics_data):
    precision = 0.0  # Ensure scalars
    recall = 0.0
    f1_score = 0.0
    accuracy = 0.0
    
    task_data = {} 
    data_len = len(metrics_data)
    
    task_list = get_task_list()
    
    print("aggregate result invoked")
    
    for batch_id, metrics in metrics_data.items():
        batch_precision = float(metrics['average']['precision'].item())  # Ensure float
        batch_recall = float(metrics['average']['recall'].item())        # Ensure float
        batch_f1_score = float(metrics['average']['f1_score'].item())   # Ensure float
        batch_accuracy = float(metrics['average']['accuracy'].item())   # Ensure float
        
        precision += batch_precision
        recall += batch_recall
        f1_score += batch_f1_score
        accuracy += batch_accuracy
        
        for task, data in metrics.items():
            if task in task_list:
                if task not in task_data:
                    task_data[task] = {}  # Initialize inner dict if not present
                
                for key, value in data.items():
                    if key not in task_data[task]:
                        if key not in ['auc_roc', 'pr_auc']:
                            task_data[task][key] = value.clone()  # Copy scalar value
                        else:
                            task_data[task][key] = {
                                class_name: {'probs': [], 'labels': []} for class_name in value.keys()
                            }
                    else:
                        if key not in ['auc_roc', 'pr_auc']:
                            task_data[task][key] += value  # Accumulate scalar metrics
                        else:
                            for class_name, class_data in value.items():
                                task_data[task][key][class_name]['probs'].extend(class_data['probs'])
                                task_data[task][key][class_name]['labels'].extend(class_data['labels'])
    
    # Finalize aggregation
    for task, data in task_data.items():
        for key, value in data.items():
            if key not in ['auc_roc', 'pr_auc']:
                data[key] = value / data_len  # Average scalar metrics
            else:
                for class_name, class_data in value.items():
                    probs = class_data['probs']
                    labels = class_data['labels']

                    if len(set(labels)) > 1:
                        fpr, tpr, roc_thresh = roc_curve(labels, probs)
                        precision_, recall_, pr_thresh = precision_recall_curve(labels, probs)

                        roc_auc = auc(fpr, tpr)
                        pr_auc = auc(recall_, precision_)
                        
                        class_data['auc'] = roc_auc
                        class_data['pr_auc'] = pr_auc
                        class_data['fpr'] = fpr
                        class_data['tpr'] = tpr
                        class_data['precision'] = precision_
                        class_data['recall'] = recall_
                        class_data['roc_thresh'] = roc_thresh
                        class_data['pr_thresh'] = pr_thresh
                    else:
                        class_data['auc'] = 0.0
                        class_data['pr_auc'] = 0.0
                        class_data['fpr'] = []
                        class_data['tpr'] = []
                        class_data['precision'] = []
                        class_data['recall'] = []
                        class_data['roc_thresh'] = []
                        class_data['pr_thresh'] = []

        
        auc_roc_output = " | ".join(
            [f"{class_name}: {data['auc_roc'][class_name]['auc']:.2f}" for class_name in data['auc_roc']]
        )
        pr_auc_output = " | ".join(
            [f"{class_name}: {data['auc_roc'][class_name]['pr_auc']:.2f}" for class_name in data['auc_roc']]
        )
        
        print(f"Task: {task} | Average Precision: {data['precision']:.2f} | Average Recall: {data['recall']:.2f} | Average f1 score: {data['f1_score']:.2f}")
        print(f"Task: {task} | Average AUC-ROC | {auc_roc_output}")
        print(f"Task: {task} | Average PR-AUC | {pr_auc_output}")
    
    print(f"Average Precision: {precision / data_len:.2f} | Average Recall: {recall / data_len:.2f} | Average f1 score: {f1_score / data_len:.2f} | Average Accuracy: {accuracy / data_len:.2f}")
    
    upated_data = convert_to_serializable(task_data)
    
    with open('aggregated_metric.json', 'w') as file:
        json.dump(upated_data, file, indent=4)
                
    
    
def convert_to_serializable(data):
    if isinstance(data, torch.Tensor):  # Handle PyTorch tensors
        return data.tolist()
    elif isinstance(data, np.ndarray):  # Handle NumPy arrays
        return data.tolist()
    elif isinstance(data, dict):  # Recursively process dictionaries
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):  # Recursively process lists
        return [convert_to_serializable(item) for item in data]
    else:
        return data  # Return unchanged for serializable types

def save_to_csv(cls_logits, cls_labels, batch_ids, file_name="predictions.csv"):
    """
    Save predicted probabilities, predicted classes, and actual classes to a CSV file.
    Each row represents a batch item with three columns for each task: actual class, predicted class, and probability of predicted class.
    If the file already exists, append the new records to the same CSV file.

    Args:
        cls_logits (torch.Tensor): The logits tensor of shape [batch_size, num_tasks, num_classes].
        cls_labels (torch.Tensor): The actual labels tensor of shape [batch_size, num_tasks].
        task_names (list): List of task names corresponding to the number of tasks.
        batch_ids (list): List of IDs representing each batch item (same length as batch_size).
        file_name (str): The name of the output CSV file.
    """
    task_names = get_task_list()
    # Step 1: Apply softmax to logits to get probabilities
    probabilities = F.softmax(cls_logits, dim=-1)
    
    # Step 2: Get the predicted class using argmax
    predicted_classes = torch.argmax(probabilities, dim=-1)
    
    # Step 3: Convert probabilities, predicted classes, and actual classes to numpy arrays
    probabilities_np = probabilities.detach().cpu().numpy()  # [batch_size, num_tasks, num_classes]
    predicted_classes_np = predicted_classes.detach().cpu().numpy()  # [batch_size, num_tasks]
    cls_labels_np = cls_labels.detach().cpu().numpy()  # [batch_size, num_tasks]
    
    # Step 4: Prepare data for CSV
    combined_results = []
    for batch_idx in range(cls_labels_np.shape[0]):  # Iterate over batch
        row = {"ID": batch_ids[batch_idx]}  # Add ID for each batch item
        for task_idx, task_name in enumerate(task_names):  # Iterate over tasks
            row[f"{task_name}"] = cls_labels_np[batch_idx, task_idx]  # Actual class
            row[f"{task_name}_pred"] = predicted_classes_np[batch_idx, task_idx]  # Predicted class
            row[f"{task_name}_prob"] = probabilities_np[batch_idx, task_idx, predicted_classes_np[batch_idx, task_idx]]  # Probability of the predicted class
        combined_results.append(row)
    
    # Step 5: Create a Pandas DataFrame for saving
    df = pd.DataFrame(combined_results)
    
    # Step 6: Append to CSV file if it exists, otherwise create a new one
    if os.path.isfile(file_name):
        # If the file exists, append without writing the header
        df.to_csv(file_name, mode='a', header=False, index=False)
    else:
        # If the file does not exist, create a new one with the header
        df.to_csv(file_name, mode='w', header=True, index=False)

    print(f"Predictions saved to {file_name}")
    
def get_task_list():
    task_names = ["No Finding", "Enlarged Cardiomediastinum",
                              "Cardiomegaly", "Lung Opacity",
                              "Lung Lesion", "Edema",
                              "Consolidation", "Pneumonia",
                              "Atelectasis", "Pneumothorax",
                              "Pleural Effusion", "Pleural Other",
                              "Fracture", "Support Devices"]
    
    return task_names

def get_class_map(id):
    
    class_map = {
        '0': 'negative',
        '1': 'positive',
        '2': 'uncertain'
    }
    
    return class_map.get(str(id), f"class_{id}")


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def expert_atttention_visualization(images, class_labels, batch_id, attention_weights = None, vit_attention = None, cnn_attention = None, save_dir="med_cxr/expert_heatmap_attn_pool_3", vit_patches=50, resnet_patches=49, vit_grid=(7, 7), resnet_grid=(7, 7)):
    # replace expert attention with cnn_attention and vit_attention
    """
    Visualize attention maps for each expert token in a batch of images with attention scores split between ViT and ResNet patches.
    Each figure will contain 15 subplots: the original image and attention maps for 14 expert tokens.
    
    Args:
        images (torch.Tensor): Batch of images in shape [batch_size, C, H, W].
        expert_attention (torch.Tensor): Attention scores for each expert token in shape [batch_size, num_expert_tokens, total_patches].
        save_dir (str): Directory to save the attention map visualizations.
        vit_patches (int): Number of patches from ViT encoder (e.g., 50 for ViT with CLS token).
        resnet_patches (int): Number of patches from ResNet encoder (e.g., 196).
        vit_grid (tuple): Grid size for ViT patches (e.g., (7, 7)).
        resnet_grid (tuple): Grid size for ResNet patches (e.g., (14, 14)).
    """
    os.makedirs(save_dir, exist_ok=True)
    batch_size, _, H, W = images.shape
    if attention_weights is not None:
        num_expert_tokens = attention_weights.size(1)
    else:
        num_expert_tokens = vit_attention.size(1)
    class_names = get_task_list()
    
    for i in range(batch_size):
        img = images[i].permute(1, 2, 0).cpu().detach().numpy()  # Convert to HWC format for visualization
        img = (img * 255).astype(np.uint8)  # Rescale to 0-255
        
        labels = class_labels[i].cpu().detach().numpy()

        # Plot for ResNet patches
        fig_resnet, axes_resnet = plt.subplots(3, 5, figsize=(15, 9))
        fig_resnet.suptitle("ResNet Attention", fontsize=16)

        # Original image in the first subplot
        axes_resnet[0, 0].imshow(img)
        axes_resnet[0, 0].set_title("Original Image")
        axes_resnet[0, 0].axis("off")

        for j in range(num_expert_tokens):
            # Get ResNet attention for the j-th expert token
            if attention_weights is not None:
                expert_attn_resnet = attention_weights[i, j, :resnet_patches].cpu().detach().numpy().reshape(resnet_grid)
            else:
                expert_attn_resnet = cnn_attention[i, j, :].cpu().detach().numpy().reshape(resnet_grid)
            
            # Resize and normalize attention map
            heatmap_resnet = cv2.resize(expert_attn_resnet, (W, H))
            heatmap_resnet = np.uint8(255 * heatmap_resnet / heatmap_resnet.max())
            heatmap_resnet = cv2.applyColorMap(heatmap_resnet, cv2.COLORMAP_JET)
            # overlay_resnet = cv2.addWeighted(img, 0.6, heatmap_resnet, 0.4, 0)
            overlay_resnet = heatmap_resnet

            # Place attention map in the correct subplot
            row, col = divmod(j + 1, 5)
            axes_resnet[row, col].imshow(cv2.cvtColor(overlay_resnet, cv2.COLOR_BGR2RGB))
            axes_resnet[row, col].set_title(f"{class_names[j]} - {get_class_map(labels[j])}")
            axes_resnet[row, col].axis("off")
        
        # Save ResNet attention figure
        plt.savefig(os.path.join(save_dir, f"attention_resnet_{batch_id}_{i}.png"), dpi=300)
        plt.close(fig_resnet)

        # Plot for ViT patches
        fig_vit, axes_vit = plt.subplots(3, 5, figsize=(15, 9))
        fig_vit.suptitle("ViT Attention", fontsize=16)

        # Original image in the first subplot
        axes_vit[0, 0].imshow(img)
        axes_vit[0, 0].set_title("Original Image")
        axes_vit[0, 0].axis("off")

        for j in range(num_expert_tokens):
            # Get ViT attention for the j-th expert token (ignore CLS token)
            if attention_weights is not None:
                expert_attn_vit = attention_weights[i, j, resnet_patches:resnet_patches + vit_patches].cpu().detach().numpy().reshape(vit_grid)
            else:
                expert_attn_vit = vit_attention[i, j, :].cpu().detach().numpy()[1:].reshape(vit_grid)
            
            # Resize and normalize attention map
            heatmap_vit = cv2.resize(expert_attn_vit, (W, H))
            heatmap_vit = np.uint8(255 * heatmap_vit / heatmap_vit.max())
            heatmap_vit = cv2.applyColorMap(heatmap_vit, cv2.COLORMAP_JET)
            # overlay_vit = cv2.addWeighted(img, 0.6, heatmap_vit, 0.4, 0)
            overlay_vit = heatmap_vit

            # Place attention map in the correct subplot
            row, col = divmod(j + 1, 5)
            axes_vit[row, col].imshow(cv2.cvtColor(overlay_vit, cv2.COLOR_BGR2RGB))
            axes_vit[row, col].set_title(f"{class_names[j]} - {get_class_map(labels[j])}")
            axes_vit[row, col].axis("off")
        
        # Save ViT attention figure
        plt.savefig(os.path.join(save_dir, f"attention_vit_{batch_id}_{i}.png"), dpi=300)
        plt.close(fig_vit)

def query_attention_visualization(images, query_attention_scores, save_dir="med_cxr/query_attn", resnet_grid=(14, 14)):
    """
    Save attention maps for query attention on ResNet patches only, after aggregating attention scores across heads.
    
    Args:
        images (torch.Tensor): Batch of images in shape [batch_size, C, H, W].
        query_attention_scores (torch.Tensor): Attention scores for query tokens attended to ResNet patches, 
                                               shape [batch_size, num_heads, num_query_tokens, resnet_patches].
        save_dir (str): Directory to save the PNG files.
        resnet_grid (tuple): Grid size for ResNet patches (e.g., (14, 14)).
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    batch_size, num_heads, num_query_tokens, resnet_patches = query_attention_scores.shape
    H, W = images.shape[2], images.shape[3]
    
    for i in range(batch_size):
        img = images[i].permute(1, 2, 0).cpu().detach().numpy()  # Convert to HWC format for visualization
        img = (img * 255).astype(np.uint8)  # Rescale to 0-255

        for q in range(num_query_tokens):
            # Aggregate attention scores across heads by taking the mean
            attn_resnet = query_attention_scores[i, :, q].mean(dim=0).cpu().detach().numpy().reshape(resnet_grid)  # Reshape to [14, 14]

            # Resize attention map to the image size
            heatmap_resnet = cv2.resize(attn_resnet, (W, H))
            heatmap_resnet = np.uint8(255 * heatmap_resnet / heatmap_resnet.max())
            heatmap_resnet = cv2.applyColorMap(heatmap_resnet, cv2.COLORMAP_JET)

            # Blend the heatmap with the original image
            overlay_resnet = cv2.addWeighted(img, 0.6, heatmap_resnet, 0.4, 0)

            # Save the visualization as a PNG file
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(cv2.cvtColor(overlay_resnet, cv2.COLOR_BGR2RGB))
            ax.set_title(f"Batch {i} - Query {q} Mean Attention on ResNet Patches")
            ax.axis("off")
            plt.savefig(os.path.join(save_dir, f"batch_{i}_query_{q}_mean_attention.png"))
            plt.close(fig)


import matplotlib.pyplot as plt


def visualize_images_with_labels(images, class_labels, image_path, dicom_id, save_dir = "med_cxr/true_images", prefix="batch", text_output = None, generated_caption = None):
    """
    Visualizes a batch of images with their corresponding true labels and saves the plots to a directory.

    Args:
        images (torch.Tensor): Batch of images, shape [batch_size, C, H, W].
        class_labels (torch.Tensor): Batch of labels, shape [batch_size, num_abnormalities].
        save_dir (str): Directory to save the plots
        prefix (str): Prefix for saved file names.
    """
    os.makedirs(save_dir, exist_ok=True)  # Ensure save directory exists
    class_names = get_task_list()
    batch_size = images.size(0)
    num_abnormalities = len(class_names)

    # Create a figure
    fig, axes = plt.subplots(batch_size, 1, figsize=(12, batch_size * 4))  # One row per image

    # Adjust layout
    fig.subplots_adjust(hspace=0.5)

    for i in range(batch_size):
        # Get image and rescale to 0-255
        img = images[i].permute(1, 2, 0).cpu().detach().numpy()  # Convert to HWC format
        img = (img * 255).astype(np.uint8)  # Rescale to 0-255
        
        # Get labels for the current image
        labels = class_labels[i].cpu().detach().numpy()

        # Format labels for multi-line display
        formatted_labels = [
            f"{class_names[j]}: {get_class_map(labels[j])}" for j in range(num_abnormalities)
        ]
        # Split labels into multiple lines
        label_text = "\n".join(textwrap.wrap(", ".join(formatted_labels), width=5 * 20))

        # Plot the image
        ax = axes[i] if batch_size > 1 else axes  # Handle single-row case
        ax.imshow(img)
        ax.set_title(f"ID: {dicom_id[i]}\nPATH: {image_path[i]}\n{label_text}", fontsize=10)
        ax.axis("off")
        
        with open(os.path.join(save_dir, f"{prefix}_{i}_true_caption.txt"), "w") as file:
            # Write the text data to the file
            file.write(text_output[i])
    
        with open(os.path.join(save_dir, f"{prefix}_{i}_generated_caption.txt"), "w") as file:
            # Write the text data to the file
            file.write(generated_caption[i])
    
    # Save the plot
    file_path = os.path.join(save_dir, f"{prefix}_visualization.png")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close(fig)  # Close the figure to free memory
    
    

    print(f"Visualization saved to {file_path}")
