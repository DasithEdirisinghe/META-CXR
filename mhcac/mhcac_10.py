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

    def forward(self, vit_patches = None, cnn_patches = None, text_embeddings = None):
        # Project image and text embeddings
        vit_proj = self.layer_norm(self.vit_proj(vit_patches)) if vit_patches is not None else None
        cnn_proj = self.layer_norm(self.cnn_proj(cnn_patches)) if cnn_patches is not None else None
        txt_proj = self.layer_norm(self.text_proj(text_embeddings)) if text_embeddings is not None else None
        return vit_proj, cnn_proj, txt_proj

# Define trainable positional encoding
class TrainablePositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(TrainablePositionalEncoding, self).__init__()
        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches, embed_dim))  # Learnable parameter

    def forward(self, x):
        return x + self.positional_encoding  # Add positional encoding to input
    
# ExpertTokenCrossAttention layer that performs both image and query cross-attention in a single pass
class ExpertTokenCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_abnormalities=14, dropout=0.1, text_dropout_rate = 0.5):
        super(ExpertTokenCrossAttention, self).__init__()
        
        # Expert-to-image cross-attention
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.image_to_text_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

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
        self.norm_image_text = nn.LayerNorm(embed_dim)
        self.norm_self_attention = nn.LayerNorm(embed_dim)
        self.norm_cross_attention = nn.LayerNorm(embed_dim)
        self.norm_ff = nn.LayerNorm(embed_dim)
        
        self.text_dropout_rate = text_dropout_rate

    def forward(self, expert_tokens, image_patches, text_embeddings = None):
        if text_embeddings is not None:
            # Apply dropout to text embeddings conditionally during training
            dropout_mask = (torch.rand(text_embeddings.size(0), 1, 1) > self.text_dropout_rate).float().to(text_embeddings.device)
            text_embeddings = text_embeddings * dropout_mask
            
            # Cross-attend image patches with text embeddings
            attended_image, _ = self.image_to_text_attention(
                query=image_patches, key=text_embeddings, value=text_embeddings
            )
            attended_image = self.norm_image_text(attended_image + image_patches)  # Residual connection
        
        else:
            # In inference mode or when text_embeddings is unavailable, rely on image patches alone
            attended_image = self.norm_image_text(image_patches)
        
        # Expert-to-ViT cross-attention
        attended_expert, attention_weights = self.cross_attention(query = expert_tokens, key = attended_image, value = attended_image)
        attended_expert = self.norm_cross_attention(attended_expert + expert_tokens)  # Residual connection
        attended_expert = self.norm_ff(self.ffn_expert(attended_expert) + attended_expert)  # Feed-forward layer


        # Self-attention among expert tokens
        expert_refined, _ = self.self_attention(
            query=attended_expert, key=attended_expert, value=attended_expert
        )
        expert_refined = self.norm_self_attention(attended_expert + expert_refined)  # Residual connection and normalization

        return expert_refined, attention_weights


# Main Abnormality Classification Model
class AbnormalityClassificationModel(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, num_abnormalities=14, num_classes=3, num_layers=2, dropout=0.2, initial_expert_tokens = None):
        super(AbnormalityClassificationModel, self).__init__()

        self.embed_dim = embed_dim
        self.num_abnormalities = num_abnormalities
        # Initial projection layer to align image, text, and query embeddings
        self.embedding_alignment = CrossModalEmbeddingAlignment(embed_dim)

        if initial_expert_tokens is not None:
            self.expert_tokens = nn.Parameter(initial_expert_tokens)
        else:
            self.expert_tokens = nn.Parameter(torch.randn(num_abnormalities, embed_dim))
            nn.init.xavier_uniform_(self.expert_tokens)

        # Stack multiple ExpertTokenCrossAttention layers
        self.attention_layers = nn.ModuleList([
            ExpertTokenCrossAttention(embed_dim, num_heads, num_abnormalities, dropout)
            for _ in range(num_layers)
        ])

        # Classification heads for each expert token
        self.classifiers = nn.ModuleList([nn.Linear(embed_dim, num_classes) for _ in range(num_abnormalities)])
        
        self.contrastive_loss = InfoNCELoss(temperature=0.03)
        
        self.norm_vit = nn.LayerNorm(embed_dim)
        self.expert_token_norm = nn.LayerNorm(embed_dim)
        
        self.cnn_pos_enc = TrainablePositionalEncoding(num_patches=196, embed_dim=embed_dim)

    def forward(self, cnn_patches, vit_patches = None, text_embeddings = None, labels = None):
        batch_size = cnn_patches.size(0)
        
        # Initial projection of image, text, and query embeddings
        if vit_patches is not None:
            vit_patches = self.norm_vit(vit_patches)
        vit_proj, cnn_proj, txt_proj = self.embedding_alignment(vit_patches, cnn_patches, text_embeddings)
        cnn_proj = self.cnn_pos_enc(cnn_proj)
        
        if (vit_proj is not None):
            image_patches = torch.cat([cnn_proj, vit_proj], dim=1)  # [batch_size, num_cnn_patches + num_vit_patches, embed_dim]
        else:
            image_patches = cnn_proj

        # Expand CNN and ViT expert tokens to match batch size
        expert_tokens = self.expert_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N_expert, D]

        # Pass through multiple attention layers
        attention_weights_list = []
        for i, layer in enumerate(self.attention_layers):
            expert_tokens, attention_weights = layer(expert_tokens, image_patches, txt_proj)
            attention_weights_list.append(attention_weights)
        
        # Normalize initial expert tokens
        normalized_expert_tokens = self.expert_token_norm(self.expert_tokens)
        # Add normalized initial expert tokens back
        final_expert_tokens = expert_tokens + normalized_expert_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        # Classification for each abnormality
        logits = []
        for i in range(len(self.classifiers)):
            logits.append(self.classifiers[i](final_expert_tokens[:, i, :]))  # Shape: [batch_size, num_classes] for each abnormality
        logits = torch.stack(logits, dim=1)  # Shape: [batch_size, num_abnormalities, num_classes]
        
        if labels is not None:
            # Contrastive loss
            contrastive_loss = self.contrastive_loss(final_expert_tokens, labels)
            
            return logits, attention_weights_list, contrastive_loss
        
        else:
            return logits, attention_weights_list
