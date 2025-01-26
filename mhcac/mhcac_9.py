import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import InfoNCELoss

# CrossModalEmbeddingAlignment to project image, text, and query embeddings into a common space
class CrossModalEmbeddingAlignment(nn.Module):
    def __init__(self, common_dim, cnn_dim=1408,vit_dim = 768, txt_dim=768, query_dim=768):
        super(CrossModalEmbeddingAlignment, self).__init__()
        self.vit_proj = nn.Linear(vit_dim, common_dim)
        self.cnn_proj = nn.Linear(cnn_dim, common_dim)
        self.text_proj = nn.Linear(txt_dim, common_dim)
        self.query_proj = nn.Linear(query_dim, common_dim)
        self.layer_norm = nn.LayerNorm(common_dim)

    def forward(self, vit_patches, cnn_patches, text_embeddings = None, query_embeddings= None):
        # Project image and text embeddings
        vit_proj = self.layer_norm(self.vit_proj(vit_patches))
        cnn_proj = self.layer_norm(self.cnn_proj(cnn_patches))
        txt_proj = self.text_proj(text_embeddings) if text_embeddings is not None else None
        query_proj = self.query_proj(query_embeddings) if query_embeddings is not None else None
        return vit_proj, cnn_proj


# ExpertTokenCrossAttention layer that performs both image and query cross-attention in a single pass
class ExpertTokenCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_abnormalities=14, dropout=0.1):
        super(ExpertTokenCrossAttention, self).__init__()
        
        # Expert-to-image cross-attention
        self.vit_cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cnn_cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Feed-forward networks for modality-specific features
        self.ffn_vit = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_cnn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        # Self-attention among expert tokens for knowledge sharing
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Feed-forward for fused features
        self.fusion_ffn = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

        # Layer normalization for residual connections
        self.norm_fusion = nn.LayerNorm(embed_dim)
        self.norm_self_attention = nn.LayerNorm(embed_dim)
        self.norm_vit = nn.LayerNorm(embed_dim)
        self.norm_cnn = nn.LayerNorm(embed_dim)

    def forward(self, cnn_expert_tokens, vit_expert_tokens, cnn_patches, vit_patches):
        # Expert-to-ViT cross-attention
        vit_attended, vit_attention_weights = self.vit_cross_attention(query = vit_expert_tokens, key = vit_patches, value = vit_patches)
        vit_attended = self.norm_vit(vit_attended + vit_expert_tokens)  # Residual connection
        vit_attended = self.ffn_vit(vit_attended)  # Feed-forward layer

        # Expert-to-CNN cross-attention
        cnn_attended, cnn_attention_weights = self.cnn_cross_attention(query = cnn_expert_tokens, key = cnn_patches, value = cnn_patches)
        cnn_attended = self.norm_cnn(cnn_attended + cnn_expert_tokens)  # Residual connection
        cnn_attended = self.ffn_cnn(cnn_attended)  # Feed-forward layer

        # Fuse modality-specific features
        fused_features = torch.cat([vit_attended, cnn_attended], dim=-1)  # [B, N_expert, 2 * D]
        fused_features = self.fusion_ffn(fused_features)  # Non-linear fusion
        expert_tokens = self.norm_fusion(fused_features)  # Residual connection and normalization

        # Self-attention among expert tokens
        expert_refined, _ = self.self_attention(
            query=expert_tokens, key=expert_tokens, value=expert_tokens
        )
        expert_tokens = self.norm_self_attention(expert_refined + expert_tokens)  # Residual connection and normalization

        return expert_tokens, vit_attended, cnn_attended, vit_attention_weights, cnn_attention_weights


# Main Abnormality Classification Model
class AbnormalityClassificationModel(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, num_abnormalities=14, num_classes=3, num_layers=2, dropout=0.2, initial_expert_tokens = None):
        super(AbnormalityClassificationModel, self).__init__()

        self.embed_dim = embed_dim
        self.num_abnormalities = num_abnormalities
        # Initial projection layer to align image, text, and query embeddings
        self.embedding_alignment = CrossModalEmbeddingAlignment(embed_dim)

        if initial_expert_tokens is not None:
            self.cnn_expert_tokens = nn.Parameter(initial_expert_tokens)
            self.vit_expert_tokens = nn.Parameter(initial_expert_tokens)
        else:
            self.cnn_expert_tokens = nn.Parameter(torch.randn(num_abnormalities, embed_dim))
            self.vit_expert_tokens = nn.Parameter(torch.randn(num_abnormalities, embed_dim))
            nn.init.xavier_uniform_(self.cnn_expert_tokens)
            nn.init.xavier_uniform_(self.vit_expert_tokens)

        # Stack multiple ExpertTokenCrossAttention layers
        self.attention_layers = nn.ModuleList([
            ExpertTokenCrossAttention(embed_dim, num_heads, num_abnormalities, dropout)
            for _ in range(num_layers)
        ])

        # Classification heads for each expert token
        self.classifiers = nn.ModuleList([nn.Linear(embed_dim, num_classes) for _ in range(num_abnormalities)])
        
        self.contrastive_loss = InfoNCELoss(temperature=0.05)
        self.norm_vit = nn.LayerNorm(embed_dim)

    def forward(self, cnn_patches, vit_patches, labels = None):
        batch_size = cnn_patches.size(0)
        
        # Initial projection of image, text, and query embeddings
        vit_proj, cnn_proj = self.embedding_alignment(vit_patches, cnn_patches)
        # print(img_proj.shape)

        # Expand CNN and ViT expert tokens to match batch size
        cnn_expert_tokens = self.cnn_expert_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N_expert, D]
        vit_expert_tokens = self.vit_expert_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N_expert, D]
        # img_proj = img_proj.unsqueeze(1).expand(-1, self.num_abnormalities, -1, -1).contiguous()
        # query_proj = query_proj.unsqueeze(1).expand(-1, self.num_abnormalities, -1, -1).contiguous() if query_proj is not None else None

        # Pass through multiple attention layers
        vit_attention_weights_list = []
        cnn_attention_weights_list = []
        for layer in self.attention_layers:
            expert_tokens, vit_expert_tokens, cnn_expert_tokens, vit_attention_weights, cnn_attention_weights = layer(cnn_expert_tokens, vit_expert_tokens, cnn_proj, vit_proj)
            vit_attention_weights_list.append(vit_attention_weights)
            cnn_attention_weights_list.append(cnn_attention_weights)

        # Classification for each abnormality
        logits = []
        for i in range(len(self.classifiers)):
            logits.append(self.classifiers[i](expert_tokens[:, i, :]))  # Shape: [batch_size, num_classes] for each abnormality
        logits = torch.stack(logits, dim=1)  # Shape: [batch_size, num_abnormalities, num_classes]
        
        if labels is not None:
            # Contrastive loss
            contrastive_loss = self.contrastive_loss(expert_tokens, labels)
            
            return logits, vit_attention_weights_list, cnn_attention_weights_list, contrastive_loss
        
        else:
            return logits, vit_attention_weights_list, cnn_attention_weights_list
