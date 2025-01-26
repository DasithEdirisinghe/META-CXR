import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import AbnormalitySpecificLoss, AttentionLoss

class DownsamplePatches(nn.Module):
    def __init__(self, input_patch_count, output_patch_count, embed_dim, method="conv"):
        """
        Downsamples patch embeddings to match the target patch count.
        
        Args:
            input_patch_count (int): Number of input patches (e.g., 196 for CNN).
            output_patch_count (int): Number of target patches (e.g., 49 for ViT).
            embed_dim (int): Dimensionality of the patch embeddings.
            method (str): Downsampling method ('conv' for convolution, 'pool' for pooling).
        """
        super(DownsamplePatches, self).__init__()
        self.input_patch_count = input_patch_count
        self.output_patch_count = output_patch_count
        self.embed_dim = embed_dim
        self.method = method
        
        # Define the method for downsampling
        if method == "conv":
            self.downsampler = nn.Conv2d(
                embed_dim, embed_dim, kernel_size=2, stride=2, padding=0
            )
        elif method == "pool":
            self.downsampler = None  # Will use F.adaptive_avg_pool2d
        else:
            raise ValueError("Unsupported method. Choose 'conv' or 'pool'.")
    
    def forward(self, patches):
        """
        Forward pass to downsample patches.
        
        Args:
            patches (Tensor): Patch embeddings of shape [batch_size, input_patch_count, embed_dim].
        
        Returns:
            Tensor: Downsampled patch embeddings of shape [batch_size, output_patch_count, embed_dim].
        """
        batch_size = patches.size(0)
        
        # Reshape patches into a grid (assumes square grid)
        grid_size = int(self.input_patch_count ** 0.5)  # e.g., 14x14
        target_size = int(self.output_patch_count ** 0.5)  # e.g., 7x7
        patches = patches.view(batch_size, grid_size, grid_size, self.embed_dim)
        patches = patches.permute(0, 3, 1, 2)  # Shape: [batch_size, embed_dim, grid_size, grid_size]
        
        if self.method == "conv":
            # Apply convolutional downsampling
            patches_downsampled = self.downsampler(patches)  # Shape: [batch_size, embed_dim, target_size, target_size]
        elif self.method == "pool":
            # Apply average pooling
            patches_downsampled = F.adaptive_avg_pool2d(patches, (target_size, target_size))
        
        # Reshape back to patch embedding format
        patches_downsampled = patches_downsampled.permute(0, 2, 3, 1)  # Shape: [batch_size, target_size, target_size, embed_dim]
        patches_downsampled = patches_downsampled.view(batch_size, self.output_patch_count, self.embed_dim)  # Flatten
        
        return patches_downsampled

# CrossModalEmbeddingAlignment to project image, text, and query embeddings into a common space
class CrossModalEmbeddingAlignment(nn.Module):
    def __init__(self, common_dim, cnn_dim=1408,vit_dim = 768, txt_dim=768, expert_dim=768):
        super(CrossModalEmbeddingAlignment, self).__init__()
        self.vit_proj = nn.Linear(vit_dim, common_dim)
        self.cnn_proj = nn.Linear(cnn_dim, common_dim)
        self.text_proj = nn.Linear(txt_dim, common_dim)
        self.expert_proj = nn.Linear(expert_dim, common_dim)
        
        self.cnn_norm = nn.LayerNorm(common_dim)  # For CNN patches
        self.vit_norm = nn.LayerNorm(common_dim)    # For ViT patches
        self.text_norm = nn.LayerNorm(common_dim)   # For text embeddings
        self.expert_norm = nn.LayerNorm(common_dim) # For expert tokens

    def forward(self, vit_patches = None, cnn_patches = None, text_embeddings = None, expert_tokens = None):
        # Project image and text embeddings
        vit_proj = F.normalize(self.vit_proj(vit_patches), dim=-1) if vit_patches is not None else None
        cnn_proj = F.normalize(self.cnn_proj(cnn_patches), dim=-1) if cnn_patches is not None else None
        txt_proj = F.normalize(self.text_proj(text_embeddings), dim=-1) if text_embeddings is not None else None
        expert_proj = self.expert_norm(self.expert_proj(expert_tokens)) if expert_tokens is not None else None
        return vit_proj, cnn_proj, txt_proj, expert_proj

# Define trainable positional encoding
class TrainablePositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(TrainablePositionalEncoding, self).__init__()
        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches, embed_dim))  # Learnable parameter

    def forward(self, x):
        return x + self.positional_encoding  # Add positional encoding to input
    
# ExpertTokenCrossAttention layer that performs both image and query cross-attention in a single pass
class ExpertTokenCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_abnormalities=14, dropout=0.1, text_dropout_rate = 0.0):
        super(ExpertTokenCrossAttention, self).__init__()
        
        # Expert-to-image cross-attention
        self.expert_to_image_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.expert_to_text_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Feed-forward networks for modality-specific features
        self.ffn_expert = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        # Self-attention among expert tokens for knowledge sharing
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Layer normalization for residual connections
        self.norm_expert_text = nn.LayerNorm(embed_dim)
        self.norm_self_attention = nn.LayerNorm(embed_dim)
        self.norm_expert_image = nn.LayerNorm(embed_dim)
        self.norm_ff = nn.LayerNorm(embed_dim)
        
        self.text_dropout_rate = text_dropout_rate

    def forward(self, expert_tokens, image_patches, text_embeddings = None):
        if text_embeddings is not None:
            # Apply dropout to text embeddings conditionally during training
            dropout_mask = (torch.rand(text_embeddings.size(0), 1, 1) > self.text_dropout_rate).float().to(text_embeddings.device)
            text_embeddings = text_embeddings * dropout_mask
            
            # Cross-attend image patches with text embeddings
            expert_text, _ = self.expert_to_text_attention(
                query=expert_tokens, key=text_embeddings, value=text_embeddings
            )
            expert_text = self.norm_expert_text(expert_text + expert_tokens)  # Residual connection
        
        else:
            # In inference mode or when text_embeddings is unavailable, rely on image patches alone
            expert_text = expert_tokens
        
        # Self-attention among expert tokens
        expert_refined, _ = self.self_attention(
            query=expert_text, key=expert_text, value=expert_text
        )
        expert_refined = self.norm_self_attention(expert_text + expert_refined)  # Residual connection and normalization
        
        # Expert-to-ViT cross-attention
        expert_image, attention_weights = self.expert_to_image_attention(query = expert_refined, key = image_patches, value = image_patches)
        expert_image = self.norm_expert_image(expert_image + expert_refined)  # Residual connection
        expert_image = self.norm_ff(self.ffn_expert(expert_image) + expert_image)  # Feed-forward layer

        return expert_image, attention_weights


# Main Abnormality Classification Model
class AbnormalityClassificationModel(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, num_abnormalities=14, num_classes=3, num_layers=2, num_commmon_tokens= 8, dropout=0.2, initial_expert_tokens = None):
        super(AbnormalityClassificationModel, self).__init__()

        self.embed_dim = embed_dim
        self.num_abnormalities = num_abnormalities
        self.num_layers = num_layers
        # Initial projection layer to align image, text, and query embeddings
        self.embedding_alignment = CrossModalEmbeddingAlignment(embed_dim)

        if initial_expert_tokens is not None:
            self.expert_tokens = nn.Parameter(initial_expert_tokens)
        else:
            self.expert_tokens = nn.Parameter(torch.randn(num_commmon_tokens, embed_dim))
            nn.init.xavier_uniform_(self.expert_tokens)

        # Stack multiple ExpertTokenCrossAttention layers
        self.attention_layers = nn.ModuleList([
            ExpertTokenCrossAttention(embed_dim, num_heads, num_abnormalities, dropout)
            for _ in range(num_layers)
        ])

        # Classification heads for each expert token
        # self.classifiers = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(embed_dim * (num_commmon_tokens + 1), embed_dim * 2),  # Expand feature space
        #         nn.ReLU(),  # Non-linearity
        #         nn.Dropout(0.2),  # Regularization
        #         nn.Linear(embed_dim * 2, num_classes)  # Final classification layer
        #     )
        #     for _ in range(num_abnormalities)
        # ])
        
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),  # Expand feature space
                nn.ReLU(),  # Non-linearity
                nn.Dropout(0.2),  # Regularization
                nn.Linear(embed_dim * 2, num_classes)  # Final classification layer
            )
            for _ in range(num_abnormalities)
        ])

        self.expert_loss = AbnormalitySpecificLoss(temperature=0.05, margin=0.5)
        # self.attention_loss = AttentionLoss(lambda_sparsity=0.3)
        
        self.norm_vit = nn.LayerNorm(embed_dim)
        self.expert_token_norm = nn.LayerNorm(embed_dim)
        
        # self.w_cnn = nn.Parameter(torch.tensor(1.0))
        # self.w_vit = nn.Parameter(torch.tensor(1.0))
        
        self.pos_enc = TrainablePositionalEncoding(num_patches=49, embed_dim=embed_dim)
        self.cnn_downsampler = DownsamplePatches(196, 49, 1408, method="conv")
        
        if isinstance(self.cnn_downsampler.downsampler, nn.Conv2d):  # Ensure it’s a Conv2d layer
            nn.init.xavier_uniform_(self.cnn_downsampler.downsampler.weight)
            nn.init.constant_(self.cnn_downsampler.downsampler.bias, 0)

    def forward(self, cnn_patches, vit_patches = None, text_embeddings = None, labels = None):
        batch_size = cnn_patches.size(0)
        
        # Initial projection of image, text, and query embeddings
        if vit_patches is not None:
            vit_patches = vit_patches[:, 1:, :] # here exclude the cls token- > result is 49 patches
        
        # Expand expert tokens to match batch size
        expert_tokens = self.expert_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N_expert, D]
        
        cnn_patches = self.cnn_downsampler(cnn_patches) # down sample cnn patches from 196 to 49 through a conv layer
        vit_proj, cnn_proj, txt_proj, expert_tokens = self.embedding_alignment(vit_patches, cnn_patches, text_embeddings, expert_tokens)
        
        cnn_proj = self.pos_enc(cnn_proj)
        
        # cnn_proj = self.w_cnn * cnn_proj
        if (vit_proj is not None):
            vit_proj = self.pos_enc(vit_proj)
            image_patches = torch.cat([cnn_proj, vit_proj], dim=1)  # [batch_size, num_cnn_patches + num_vit_patches, embed_dim]
        else:
            image_patches = cnn_proj

        # Pass through multiple attention layers
        attention_weights_list = []
        for i, layer in enumerate(self.attention_layers):
            if i in [0, 1]:
                expert_tokens, attention_weights = layer(expert_tokens, image_patches, txt_proj)
            elif i == self.num_layers - 2: #last before layer
                normalized_expert_tokens = self.expert_token_norm(self.expert_tokens)
                # Add normalized initial expert tokens back
                expert_tokens = expert_tokens + normalized_expert_tokens.unsqueeze(0).expand(batch_size, -1, -1)
                expert_tokens, attention_weights = layer(expert_tokens, image_patches, text_embeddings = None)
            else:
                expert_tokens, attention_weights = layer(expert_tokens, image_patches, text_embeddings = None)
                
            attention_weights_list.append(attention_weights)

        pooled_representations, orth_loss, contrastive_loss, sparsity_loss = self.expert_loss(expert_tokens, attention_weights_list, labels)
        
        # Classification for each abnormality
        logits = []
        for i in range(len(self.classifiers)):
            logits.append(self.classifiers[i](pooled_representations[:, i, :]))  # Shape: [batch_size, num_classes] for each abnormality
            # combined_features = torch.cat([expert_tokens.flatten(1), pooled_representations[:, i, :]], dim=1)  # Shape: [batch_size, embed_dim * (num_tokens + 1)]
            # logits.append(self.classifiers[i](combined_features))
        logits = torch.stack(logits, dim=1)  # Shape: [batch_size, num_abnormalities, num_classes]
        
            
        return logits, attention_weights_list, contrastive_loss, orth_loss, sparsity_loss
        
