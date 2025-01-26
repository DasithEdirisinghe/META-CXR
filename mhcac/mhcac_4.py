import torch
import torch.nn as nn
import torch.nn.functional as F

# CrossModalEmbeddingAlignment to project image and text embeddings into a common space
class CrossModalEmbeddingAlignment(nn.Module):
    def __init__(self, common_dim, img_dim = 1408, txt_dim = 768):
        super(CrossModalEmbeddingAlignment, self).__init__()
        self.image_proj = nn.Linear(img_dim, common_dim)
        self.text_proj = nn.Linear(txt_dim, common_dim)
        self.layer_norm = nn.LayerNorm(common_dim)

    def forward(self, image_patches, text_embeddings):
        # Project and normalize image patches and text embeddings
        img_proj = self.layer_norm(self.image_proj(image_patches))
        txt_proj = None
        if text_embeddings is not None:
            txt_proj = self.layer_norm(self.text_proj(text_embeddings))
        
        return img_proj, txt_proj

class CrossModalAttentionLayerWithPartialTraining(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, text_dropout_rate=0.5):
        super(CrossModalAttentionLayerWithPartialTraining, self).__init__()
        
        # Cross-attention layers for image and text interactions
        self.image_to_text_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Fusion layer after cross-attending between modalities
        self.fusion_layer = nn.Linear(2 * embed_dim, embed_dim)
        
        # Cross-attention for expert tokens with fused representation
        self.expert_to_fused_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Self-attention for expert tokens
        self.expert_self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Layer norms and dropout
        self.norm_image_text = nn.LayerNorm(embed_dim)
        self.norm_fusion = nn.LayerNorm(embed_dim)
        self.norm_expert = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.text_dropout_rate = text_dropout_rate

    def forward(self, image_patches, text_embeddings=None, expert_tokens=None):
        if text_embeddings is not None:
            # Apply dropout to text embeddings conditionally during training
            dropout_mask = (torch.rand(text_embeddings.size(0), 1, 1) > self.text_dropout_rate).float().to(text_embeddings.device)
            text_embeddings = text_embeddings * dropout_mask
            
            # Cross-attend image patches with text embeddings
            attended_image, _ = self.image_to_text_attention(
                query=image_patches, key=text_embeddings, value=text_embeddings
            )
            attended_image = self.norm_image_text(attended_image + image_patches)  # Residual connection
            
            fused_representation = self.norm_fusion(attended_image)  # Layer normalization
        
        else:
            # In inference mode or when text_embeddings is unavailable, rely on image patches alone
            fused_representation = self.norm_fusion(image_patches)

        # Cross-attend expert tokens with the fused representation
        expert_to_fused, _ = self.expert_to_fused_attention(
            query=expert_tokens, key=fused_representation, value=fused_representation
        )
        expert_to_fused = self.norm_expert(expert_to_fused + expert_tokens)  # Residual connection
        
        # Self-attention among expert tokens
        expert_features, _ = self.expert_self_attention(
            query=expert_to_fused, key=expert_to_fused, value=expert_to_fused
        )
        expert_features = self.norm_expert(expert_features + expert_to_fused)  # Residual connection
        
        return expert_features


class AbnormalityClassifierWithPartialTraining(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, num_abnormalities=14, num_classes=3, dropout=0.1, text_dropout_rate=0.5, num_layers=3):
        super(AbnormalityClassifierWithPartialTraining, self).__init__()
        
        # Alignment layer to project image and text embeddings into a common space
        self.alignment_layer = CrossModalEmbeddingAlignment(embed_dim)
        
        # Stack multiple cross-modal attention layers for expert token refinement
        self.cross_modal_attention_layers = nn.ModuleList([
            CrossModalAttentionLayerWithPartialTraining(
                embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, text_dropout_rate=text_dropout_rate
            ) for _ in range(num_layers)
        ])
        
        # Learnable expert tokens, one for each abnormality
        self.expert_tokens = nn.Parameter(torch.randn(num_abnormalities, embed_dim))
        nn.init.xavier_uniform_(self.expert_tokens)
        
        # Classification heads for each expert token
        self.classifiers = nn.ModuleList([nn.Linear(embed_dim, num_classes) for _ in range(num_abnormalities)])

    def forward(self, image_patches, text_embeddings):
        batch_size = image_patches.size(0)
        
        # Align the image and text embeddings to a common space
        img_proj, txt_proj = self.alignment_layer(image_patches, text_embeddings)
        
        # Expand expert tokens to match batch size
        expert_tokens = self.expert_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Sequentially apply each cross-modal attention layer for expert feature refinement
        for layer in self.cross_modal_attention_layers:
            expert_tokens = layer(img_proj, txt_proj, expert_tokens)
        
        # Classification for each abnormality
        logits = []
        for i in range(len(self.classifiers)):
            logits.append(self.classifiers[i](expert_tokens[:, i, :]))  # Each expert token’s final representation is classified
        logits = torch.stack(logits, dim=1)  # Shape: [batch_size, num_expert_tokens, num_classes]

        return logits




