import torch
import torch.nn as nn
import torch.nn.functional as F

# List of abnormality names (from CheXpert dataset)
chexpert_cols = ["No Finding", "Enlarged Cardiomediastinum",
                 "Cardiomegaly", "Lung Opacity",
                 "Lung Lesion", "Edema",
                 "Consolidation", "Pneumonia",
                 "Atelectasis", "Pneumothorax",
                 "Pleural Effusion", "Pleural Other",
                 "Fracture", "Support Devices"]

# CrossModalEmbeddingAlignment to project image and text embeddings into a common space
class CrossModalEmbeddingAlignment(nn.Module):
    def __init__(self, common_dim, img_dim=1408, txt_dim=768, query_dim=768):
        super(CrossModalEmbeddingAlignment, self).__init__()
        self.image_proj = nn.Linear(img_dim, common_dim)
        self.text_proj = nn.Linear(txt_dim, common_dim)
        self.query_proj = nn.Linear(query_dim, common_dim)

    def forward(self, image_patches, text_embeddings, query_embeddings):
        # Project image and text embeddings
        img_proj = self.image_proj(image_patches)
        txt_proj = self.text_proj(text_embeddings) if text_embeddings is not None else None
        query_proj = self.query_proj(query_embeddings) if query_embeddings is not None else None
        return img_proj, txt_proj, query_proj

class CrossModalAttentionLayerWithPartialTraining(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossModalAttentionLayerWithPartialTraining, self).__init__()

        # Expert token-specific multi-head attention for image patches and query embeddings
        self.image_to_expert_attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True) 
            for _ in range(embed_dim)  # One attention head per expert token
        ])

        self.query_to_expert_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Self-attention for expert tokens
        self.expert_self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Feed-forward layer for expert refinement
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

        # LayerNorm for residual connections
        self.norm_after_image_attention = nn.LayerNorm(embed_dim)
        self.norm_after_query_attention = nn.LayerNorm(embed_dim)
        self.norm_after_self_attention = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_patches, expert_tokens, query_embeddings=None):
        # Cross-attend each expert token with image patches (individual attention for each expert token)
        attended_image = []
        image_attention_weights = [] 
        for i, expert_token in enumerate(expert_tokens.transpose(0, 1)):  # Iterating through expert tokens
            attention_output, attention_weights = self.image_to_expert_attention[i](
                query=expert_token.unsqueeze(0), key=image_patches, value=image_patches
            )
            attended_image.append(attention_output)
            image_attention_weights.append(attention_weights)

        # Stack attended image features
        attended_image = torch.stack(attended_image, dim=1)  # Shape: [batch_size, num_expert_tokens, embed_dim]

        # Residual connection and LayerNorm after attending to image patches
        attended_image = self.norm_after_image_attention(attended_image + expert_tokens)

        # Cross-attend expert tokens with query embeddings
        if query_embeddings is not None:
            attended_query, _ = self.query_to_expert_attention(
                query=attended_image, key=query_embeddings, value=query_embeddings
            )
            # Residual connection and LayerNorm after attending to query embeddings
            attended_query = self.norm_after_query_attention(attended_query + attended_image)
        else:
            attended_query = attended_image  # No query embeddings available, use image features only

        # Expert token refinement with feed-forward network
        expert_refined = self.feed_forward(attended_query)

        # Self-attention among expert tokens
        expert_features, _ = self.expert_self_attention(
            query=expert_refined, key=expert_refined, value=expert_refined
        )
        # Residual connection and LayerNorm after self-attention
        expert_features = self.norm_after_self_attention(expert_features + expert_refined)
        
        image_attention_weights = torch.stack(image_attention_weights, dim=1).transpose(0, 1)

        return expert_features, image_attention_weights

class AbnormalityClassifierWithPartialTraining(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, num_abnormalities=14, num_classes=3, dropout=0.1, num_layers=3, initial_expert_tokens = None):
        super(AbnormalityClassifierWithPartialTraining, self).__init__()

        # Alignment layer to project image and text embeddings into a common space
        self.alignment_layer = CrossModalEmbeddingAlignment(embed_dim)

        # Stack multiple cross-modal attention layers for expert token refinement
        self.cross_modal_attention_layers = nn.ModuleList([
            CrossModalAttentionLayerWithPartialTraining(
                embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
            ) for _ in range(num_layers)
        ])

        if initial_expert_tokens is not None:
            self.expert_tokens = nn.Parameter(initial_expert_tokens)
        else:
            self.expert_tokens = nn.Parameter(torch.randn(num_abnormalities, embed_dim))
            nn.init.xavier_uniform_(self.expert_tokens)
            
        # Classification heads for each expert token
        self.classifiers = nn.ModuleList([nn.Linear(embed_dim, num_classes) for _ in range(num_abnormalities)])


    def forward(self, image_patches, text_embeddings=None, query_embeddings=None):
        batch_size = image_patches.size(0)

        # Align the image and text embeddings to a common space
        img_proj, txt_proj, query_proj = self.alignment_layer(image_patches, text_embeddings, query_embeddings)

        # Expand expert tokens to match batch size
        expert_tokens = self.expert_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        # Sequentially apply each cross-modal attention layer for expert feature refinement
        for layer in self.cross_modal_attention_layers:
            expert_tokens, image_attention_weights = layer(img_proj, expert_tokens, query_proj)

        # Classification for each abnormality
        logits = []
        for i in range(len(self.classifiers)):
            logits.append(self.classifiers[i](expert_tokens[:, i, :]))  # Classify each expert token
        logits = torch.stack(logits, dim=1)  # Shape: [batch_size, num_expert_tokens, num_classes]

        return logits, image_attention_weights
