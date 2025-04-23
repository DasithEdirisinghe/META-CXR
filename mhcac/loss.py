import torch
import torch.nn as nn
import torch.nn.functional as F

# class ClassificationLoss(nn.Module):
#     def __init__(self, penalty_weight=0.5, class_weights=None):
#         super(ClassificationLoss, self).__init__()
#         if class_weights is not None:
#             self.cross_entropy_loss = nn.CrossEntropyLoss(weight=class_weights)  # Use class weights
#         else:
#             self.cross_entropy_loss = nn.CrossEntropyLoss()  # No class weights
#         self.penalty_weight = penalty_weight  # Weight of the penalty for incorrect classes

#     def forward(self, logits, true_labels):
#         # Compute weighted cross-entropy loss
#         ce_loss = self.cross_entropy_loss(logits, true_labels)

#         #Compute probabilities from logits
#         probs = torch.softmax(logits, dim=1)  # Shape: (batch_size, num_classes)

#         # Create a mask to ignore the correct class
#         batch_size = logits.shape[0]
#         correct_class_mask = torch.zeros_like(probs)
#         correct_class_mask[torch.arange(batch_size), true_labels] = 1

#         # Penalize the incorrect class probabilities
#         incorrect_probs = probs * (1 - correct_class_mask)  # Mask out correct class probabilities
#         penalty = incorrect_probs.sum(dim=1).mean()  # Mean of the summed incorrect probabilities

#         # Combine weighted cross-entropy loss with the penalty
#         total_loss = ce_loss + self.penalty_weight * penalty

#         return total_loss

class ClassificationLoss(nn.Module):
    def __init__(self, penalty_weight=0.2, class_weights=None, num_abnormalities=14):
        super(ClassificationLoss, self).__init__()
        self.penalty_weight = penalty_weight  # Weight for penalty for incorrect classes
        if class_weights is not None:
            assert isinstance(class_weights, list), "class_weights should be a list of tensors, one for each abnormality"
            self.cross_entropy_loss_list = nn.ModuleList(
                [nn.CrossEntropyLoss(weight=w, label_smoothing=0.1) for w in class_weights]
            )  # List of cross-entropy losses, one per abnormality
        else:
            self.cross_entropy_loss_list = nn.ModuleList(
                [nn.CrossEntropyLoss(label_smoothing=0.1) for _ in range(num_abnormalities)]
            )

    def forward(self, logits, true_labels):
        num_abnormalities = logits.shape[1]
        batch_size = logits.shape[0]
        total_loss = 0.0
        penalty_values = 0.0
        
        # Loop through each abnormality and compute the loss separately
        for i in range(num_abnormalities):
            logits_i = logits[:, i, :]  # Shape: [batch_size, num_classes]
            labels_i = true_labels[:, i]  # Shape: [batch_size]
            
            # Compute cross-entropy loss
            ce_loss = self.cross_entropy_loss_list[i](logits_i, labels_i)
            
            # Use log_softmax for numerical stability
            probs = torch.softmax(logits_i, dim=1).clamp(min=1e-7, max=1.0 - 1e-7)  # Avoid 0 and 1 values
            
            # Create a mask to ignore the correct class for abnormality i
            correct_class_mask = torch.zeros_like(probs)
            correct_class_mask[torch.arange(batch_size), labels_i] = 1
            
            # Penalize the incorrect class probabilities
            incorrect_probs = probs * (1 - correct_class_mask)
            penalty_values += incorrect_probs.sum(dim=1).mean()
            
            total_loss += ce_loss

        mean_loss = total_loss / num_abnormalities
        # final_loss = mean_loss + penalty_values * self.penalty_weight
        final_loss = mean_loss
        
        # Clipping the loss to avoid extremely high values
        # final_loss = torch.clamp(final_loss, min=-1e4, max=1e4)

        return final_loss

# class InfoNCELoss(nn.Module):
#     def __init__(self, temperature=0.07, margin=0.5):
#         """
#         Modified InfoNCE Loss to enforce relative positions of Positive, Negative, and Uncertain states.

#         Args:
#             temperature (float): Temperature parameter for scaling similarity logits.
#             margin (float): Margin to enforce separation between Positive and Negative states.
#         """
#         super(InfoNCELoss, self).__init__()
#         self.temperature = temperature
#         self.margin = margin

#     def forward(self, expert_tokens, labels):
#         B, N, D = expert_tokens.shape
#         expert_tokens = F.normalize(expert_tokens, dim=-1)

#         total_loss = 0.0
#         for i in range(N):
#             tokens = expert_tokens[:, i, :]
#             token_labels = labels[:, i]

#             # Masks
#             pos_mask = (token_labels == 1).float()
#             neg_mask = (token_labels == 0).float()
#             # unc_mask = (token_labels == 2).float()

#             pos_indices = pos_mask.nonzero(as_tuple=True)[0]
#             neg_indices = neg_mask.nonzero(as_tuple=True)[0]
#             # unc_indices = unc_mask.nonzero(as_tuple=True)[0]

#             similarity_matrix = torch.matmul(tokens, tokens.T)  # Pairwise similarities

#             # Positive-Negative Separation
#             if len(pos_indices) > 0 and len(neg_indices) > 0:
#                 pos_neg_similarity = similarity_matrix[pos_indices][:, neg_indices]
#                 pos_neg_loss = torch.relu(self.margin - (1 - pos_neg_similarity)).mean()
#             else:
#                 pos_neg_loss = 0.0

#             # # Uncertain Alignment
#             # if len(unc_indices) > 0 and len(pos_indices) > 0 and len(neg_indices) > 0:
#             #     pos_unc_similarity = similarity_matrix[unc_indices][:, pos_indices].mean(dim=1)
#             #     neg_unc_similarity = similarity_matrix[unc_indices][:, neg_indices].mean(dim=1)
#             #     unc_loss = torch.abs(pos_unc_similarity - neg_unc_similarity).mean()
#             # else:
#             #     unc_loss = 0.0

#             # Dynamically weight the contributions
#             # contribution_weight = len(pos_indices) + len(neg_indices) + len(unc_indices) + 1e-6
#             # total_loss += (pos_neg_loss + unc_loss) / contribution_weight

#             total_loss += (pos_neg_loss)
            
#         return total_loss / N


class AttentionPooling(nn.Module):
    def __init__(self, d_embedding, num_abnormalities):
        super().__init__()
        self.query_vectors = nn.Parameter(torch.randn(num_abnormalities, d_embedding))  # Learnable queries
        nn.init.xavier_uniform_(self.query_vectors)

    def forward(self, common_representations):
        """
        Args:
            common_representations: Tensor of shape [batch_size, num_tokens, d_embedding]
        
        Returns:
            pooled_representations: Tensor of shape [batch_size, num_abnormalities, d_embedding]
        """
        batch_size, num_tokens, d_embedding = common_representations.shape
        num_abnormalities = self.query_vectors.size(0)

        # Compute attention scores for each abnormality
        attention_scores = torch.einsum("ad,bnd->ban", self.query_vectors, common_representations)  # [batch_size, num_abnormalities, num_tokens]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_abnormalities, num_tokens]

        # Pool features using attention weights
        pooled_representations = torch.einsum("ban,bnd->bad", attention_weights, common_representations)  # [batch_size, num_abnormalities, d_embedding]

        return pooled_representations

class AbnormalitySpecificLoss(nn.Module):
    def __init__(self, temperature=0.07, margin=0.7, d_embedding = 768, num_abnormalities = 14):
        """
        Modified InfoNCE Loss for abnormality-specific tokens.

        Args:
            temperature (float): Temperature parameter for scaling similarity logits.
            margin (float): Margin to enforce separation between Positive and Negative states.
            inter_abnormality_weight (float): Weight for inter-abnormality dissimilarity.
        """
        super(AbnormalitySpecificLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        self.attention_pooling = AttentionPooling(d_embedding, num_abnormalities)
    
    def orthogonality_loss(self, common_representations):
        """
        Compute orthogonality loss for the common tokens.
        
        Args:
            common_representations: Tensor of shape [batch_size, num_tokens, d_embedding]

        Returns:
            orth_loss: Orthogonality loss
        """
        batch_size, num_tokens, d_embedding = common_representations.shape
        common_representations = F.normalize(common_representations, dim=-1)  # Normalize token embeddings

        # Compute pairwise similarity within tokens
        similarity_matrix = torch.einsum("bnd,bmd->bnm", common_representations, common_representations)  # [batch_size, num_tokens, num_tokens]

        # Compute Frobenius norm loss to enforce orthogonality
        off_diagonal_mask = 1 - torch.eye(num_tokens, device=common_representations.device).unsqueeze(0)  # [1, num_tokens, num_tokens]
        # Penalize only off-diagonal elements
        orth_loss = torch.mean((similarity_matrix * off_diagonal_mask) ** 2)
        return orth_loss
    
    def compute_weighted_sparsity_loss(self, attention_weights_list):
        """
        Compute sparsity loss across layers with layer-specific weighting.
        
        Args:
            attention_weights_list (list of torch.Tensor): List of attention weights for each layer.
            lambda_sparsity (float): Global weight for sparsity loss.
        
        Returns:
            sparsity_loss: Weighted sparsity loss across layers.
        """
        total_sparsity_loss = 0.0
        num_layers = len(attention_weights_list)
        
        # Assign higher weights to deeper layers
        layer_weights = torch.sigmoid(torch.linspace(-2, 2, steps=num_layers))  # Linearly increasing weights

        for i, layer_attention_weights in enumerate(attention_weights_list):
            # Compute sparsity loss for this layer
            sparsity_loss_layer = -torch.sum(
                layer_attention_weights * torch.log(layer_attention_weights + 1e-6)
            ) / layer_attention_weights.numel()
            
            # Apply layer-specific weight
            total_sparsity_loss += layer_weights[i] * sparsity_loss_layer

        # Scale by lambda_sparsity
        sparsity_loss = total_sparsity_loss / num_layers

        return sparsity_loss


    def forward(self, common_representations, attention_weights_list, labels = None):
        """
        Args:
            common_representations: Tensor of shape [batch_size, num_tokens, d_embedding]
            labels: Tensor of shape [batch_size, num_abnormalities] (binary labels per abnormality)

        Returns:
            total_loss: Combined loss across all abnormalities
        """
        pooled_representations_ = self.attention_pooling(common_representations)
        orth_loss = self.orthogonality_loss(common_representations)
        sparsity_loss = self.compute_weighted_sparsity_loss(attention_weights_list)
        
        if labels is None:
            return pooled_representations_, orth_loss, None, sparsity_loss
        
        batch_size, num_abnormalities, d_embedding = pooled_representations_.shape
        pooled_representations = F.normalize(pooled_representations_, dim=-1)  # Normalize embeddings

        contrastive_loss = 0.0

        # Loop over each abnormality
        for a in range(num_abnormalities):
            tokens = pooled_representations[:, a, :]  # [batch_size, d_embedding]
            token_labels = labels[:, a]  # [batch_size]

            # Masks
            pos_mask = (token_labels == 1).float()
            neg_mask = (token_labels == 0).float()
            unc_mask = (token_labels == 2).float()
            
            pos_indices = pos_mask.nonzero(as_tuple=True)[0]
            neg_indices = neg_mask.nonzero(as_tuple=True)[0]
            unc_indices = unc_mask.nonzero(as_tuple=True)[0]

            # Compute pairwise similarity matrix
            similarity_matrix = torch.matmul(tokens, tokens.T)  # [batch_size, batch_size]

            # Positive-Negative Separation
            if len(pos_indices) > 0 and len(neg_indices) > 0:
                pos_neg_similarity = similarity_matrix[pos_indices][:, neg_indices]  # [num_pos, num_neg]
                pos_neg_loss = torch.relu(self.margin - (1 - pos_neg_similarity)).mean()
            else:
                pos_neg_loss = 0.0
            
            # Uncertain Alignment
            if len(unc_indices) > 0 and len(pos_indices) > 0 and len(neg_indices) > 0:
                pos_unc_similarity = similarity_matrix[unc_indices][:, pos_indices].mean(dim=1)
                neg_unc_similarity = similarity_matrix[unc_indices][:, neg_indices].mean(dim=1)
                unc_loss = torch.abs(pos_unc_similarity - neg_unc_similarity).mean()
            else:
                unc_loss = 0.0

            contrastive_loss += (pos_neg_loss + unc_loss)

        contrastive_loss = contrastive_loss / num_abnormalities
        
        return pooled_representations_, orth_loss, contrastive_loss, sparsity_loss



class AttentionLoss:
    """
    Computes combined attention consistency loss and sparsity loss.
    This class allows modular computation of these losses for attention weights.

    Args:
        lambda_sparsity (float): Weighting factor for the sparsity loss component.
    """
    def __init__(self, lambda_sparsity=0.3):
        self.lambda_sparsity = lambda_sparsity

    def compute_consistency_loss(self, attention_weights_list):
        """
        Computes the attention consistency loss for each expert token across the batch.

        Args:
            attention_weights_list: List of attention weights [batch_size, num_expert_tokens, num_image_patches].

        Returns:
            consistency_loss: Scalar loss enforcing consistent attention for each expert token.
        """
        consistency_loss = 0.0
        num_layers = len(attention_weights_list)

        for attention_weights in attention_weights_list:  # Iterate over layers
            num_tokens = attention_weights.size(1)  # Number of expert tokens

            for token_idx in range(num_tokens):  # Iterate over expert tokens
                # Extract attention weights for this token: [batch_size, num_image_patches]
                token_attention = attention_weights[:, token_idx, :]

                # Consistency Loss: Mean Squared Deviation from Batch Mean
                mean_attention = token_attention.mean(dim=0, keepdim=True)  # [1, num_image_patches]
                deviation = token_attention - mean_attention
                consistency_loss += torch.mean(deviation ** 2)  # MSE loss for this token

        # Normalize by the number of layers and tokens
        return consistency_loss / (num_layers * attention_weights_list[0].size(1))

    def compute_sparsity_loss(self, attention_weights_list):
        """
        Computes sparsity loss to encourage focused attention maps.

        Args:
            attention_weights_list: List of attention weights [batch_size, num_expert_tokens, num_image_patches].

        Returns:
            sparsity_loss: Scalar loss encouraging sparse attention maps.
        """
        sparsity_loss = 0.0
        num_layers = len(attention_weights_list)

        for attention_weights in attention_weights_list:  # Iterate over layers
            num_tokens = attention_weights.size(1)  # Number of expert tokens

            for token_idx in range(num_tokens):  # Iterate over expert tokens
                # Extract attention weights for this token: [batch_size, num_image_patches]
                token_attention = attention_weights[:, token_idx, :]

                # Sparsity Loss: L1 Regularization of Attention Weights
                sparsity_loss += torch.sum(torch.abs(token_attention)) / token_attention.size(0)  # Average over batch

        # Normalize by the number of layers and tokens
        return sparsity_loss / (num_layers * attention_weights_list[0].size(1))

    def __call__(self, attention_weights_list):
        """
        Computes the total loss as the sum of consistency loss and sparsity loss.

        Args:
            attention_weights_list: List of attention weights [batch_size, num_expert_tokens, num_image_patches].

        Returns:
            total_loss: Combined loss (attention consistency + sparsity).
            consistency_loss: Attention consistency loss component.
            sparsity_loss: Sparsity loss component.
        """
        consistency_loss = self.compute_consistency_loss(attention_weights_list)
        sparsity_loss = self.compute_sparsity_loss(attention_weights_list)
        total_loss = consistency_loss + self.lambda_sparsity * sparsity_loss

        return total_loss






"""
# Example usage
logits = torch.tensor([[2.0, 1.0, 0.1], [0.1, 2.0, 0.9]])  # Logits from the model
true_labels = torch.tensor([0, 1])  # True labels (class 0 and 1)

# Define class weights (e.g., based on class imbalance in the dataset)
# In this case, we assign higher weight to class 2 (assumed to be the rare class)
class_weights = torch.tensor([1.0, 1.0, 2.0])  # Class 2 has double the weight

# Initialize and compute custom loss with class weights
loss_fn = ClassificationLoss(penalty_weight=0.5, class_weights=class_weights)
loss = loss_fn(logits, true_labels)

print(f"Loss with class weighting: {loss.item()}")
"""

