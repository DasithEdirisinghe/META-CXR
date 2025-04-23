import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import InfoNCELoss

# CrossModalEmbeddingAlignment to project image, text, and query embeddings into a common space
class CrossModalEmbeddingAlignment(nn.Module):
    def __init__(self, common_dim, img_dim=1408, txt_dim=768, query_dim=768):
        super(CrossModalEmbeddingAlignment, self).__init__()
        self.vit_proj = nn.Linear(img_dim, common_dim)
        self.cnn_proj = nn.Linear(img_dim, common_dim)
        self.text_proj = nn.Linear(txt_dim, common_dim)
        self.query_proj = nn.Linear(query_dim, common_dim)

    def forward(self, vit_patches, cnn_patches, text_embeddings = None, query_embeddings= None):
        # Project image and text embeddings
        vit_proj = self.vit_proj(vit_patches)
        cnn_proj = self.cnn_proj(cnn_patches)
        txt_proj = self.text_proj(text_embeddings) if text_embeddings is not None else None
        query_proj = self.query_proj(query_embeddings) if query_embeddings is not None else None
        return vit_proj, cnn_proj

class MHCACLayer(nn.Module):
    def __init__(self, query_dim, key_value_dim, num_heads, num_expert_tokens, dropout=0.1):
        super(MHCACLayer, self).__init__()
        
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.key_value_dim = key_value_dim
        self.num_expert_tokens = num_expert_tokens
        self.head_dim = query_dim // num_heads  # Dimension per head

        # Separate query projections for each expert token, shared K and V projections, split across heads
        self.W_Q_list = nn.ModuleList([nn.Linear(query_dim, query_dim) for _ in range(num_expert_tokens)])
        self.W_K = nn.Linear(key_value_dim, query_dim)
        self.W_V = nn.Linear(key_value_dim, query_dim)

        # Feed-forward network applied after attention
        self.feed_forward = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * 4, query_dim),
            nn.Dropout(dropout)
        )

        # Layer normalizations
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        # Dropout for attention
        self.attn_dropout = nn.Dropout(dropout)

        # Scaling factor for dot product attention
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

    def forward(self, expert_tokens, key_value_embeddings):
        batch_size, num_patches, embed_dim = key_value_embeddings.size()
        
        # Project each expert token (Q) independently, then reshape for multi-head
        all_projected_queries = [self.W_Q_list[i](expert_tokens[:, i, :]).view(batch_size, self.num_heads, self.head_dim) for i in range(self.num_expert_tokens)]
        Q = torch.stack(all_projected_queries, dim=1)  # Shape: [batch_size, num_expert_tokens, num_heads, head_dim]

        # Shared projections for K and V across expert tokens, reshape for multi-head
        K = self.W_K(key_value_embeddings).view(batch_size, num_patches, self.num_heads, self.head_dim)  # [batch_size, num_heads, num_patches, head_dim]
        V = self.W_V(key_value_embeddings).view(batch_size, num_patches, self.num_heads, self.head_dim)  # [batch_size, num_heads, num_patches, head_dim]

        # Attention computation for each expert token across heads
        attention_outputs = []
        attention_weights_list = []
        for i in range(self.num_expert_tokens):
            # Compute attention scores and weights across heads
            # print("Shape of Q[:, i].unsqueeze(1):", Q[:, i].unsqueeze(1).shape)
            # print("Shape of K:", K.shape)
            attention_scores = torch.einsum("bnhd,bmhd->bnhm", Q[:, i].unsqueeze(1), K) / self.scale  # Shape: [batch_size, num_heads, num_patches]
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.attn_dropout(attention_weights)
            attention_weights_list.append(attention_weights)

            # Compute attention output (weighted sum of V across heads)
            attention_output = torch.einsum("bnhm,bmhd->bnhd", attention_weights, V).contiguous()  # [batch_size, num_heads, head_dim]
            attention_outputs.append(attention_output.view(batch_size, -1))  # Concatenate heads

        # Stack outputs for each expert token and apply residual connection
        attention_output = torch.stack(attention_outputs, dim=1)  # Shape: [batch_size, num_expert_tokens, query_dim]
        attention_output = self.norm1(expert_tokens + attention_output)  # Residual + normalization

        # Feed-forward network with residual connection
        ff_output = self.feed_forward(attention_output)  # [batch_size, num_expert_tokens, query_dim]
        output = self.norm2(attention_output + ff_output)  # Residual + normalization

        return output, attention_weights_list


# ExpertTokenCrossAttention layer that performs both image and query cross-attention in a single pass
class ExpertTokenCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_abnormalities=14, dropout=0.1):
        super(ExpertTokenCrossAttention, self).__init__()
        
        # Expert-to-image cross-attention using MHCACLayer
        self.vit_cross_attention = MHCACLayer(embed_dim, 768, num_heads, num_abnormalities, dropout)
        
        # Expert-to-query cross-attention using MHCACLayer
        self.cnn_cross_attention = MHCACLayer(embed_dim, 1408, num_heads, num_abnormalities, dropout)
        
        # Self-attention among expert tokens for knowledge sharing
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.fusion_ffn = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Layer normalization for residual connections
        self.norm_fusion = nn.LayerNorm(embed_dim)
        self.norm_self_attention = nn.LayerNorm(embed_dim)
        
    def forward(self, cnn_expert_tokens, vit_expert_tokens, cnn_patches, vit_patches):
        # Expert-to-vit cross-attention
        vit_attended, vit_attention_weights = self.vit_cross_attention(vit_expert_tokens, vit_patches)

        # Expert-to-cnn cross-attention
        cnn_attended, cnn_attention_weights = self.cnn_cross_attention(cnn_expert_tokens, cnn_patches)

        fused_features = torch.cat([vit_attended, cnn_attended], dim=-1)  # [B, N_expert, 2 * D]
        fused_features = self.fusion_ffn(fused_features)  # [B, N_expert, D]
        # expert_tokens = self.norm_fusion(cnn_expert_tokens + vit_expert_tokens + fused_features)
        expert_tokens = self.norm_fusion(fused_features) 
        

        # Self-attention among expert tokens for knowledge sharing
        expert_refined, _ = self.self_attention(
            query=expert_tokens, key=expert_tokens, value=expert_tokens
        )
        expert_tokens = self.norm_self_attention(expert_refined + expert_tokens)  # Residual + Layer norm after self-attention

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
        vit_patches = self.norm_vit(vit_patches) # cnn patches are alread noramlized. check the blp2_qformer
        
        # Initial projection of image, text, and query embeddings
        # vit_proj, cnn_proj = self.embedding_alignment(image_patches[0], image_patches[1])
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
            expert_tokens, vit_expert_tokens, cnn_expert_tokens, vit_attention_weights, cnn_attention_weights = layer(cnn_expert_tokens, vit_expert_tokens, cnn_patches, vit_patches)
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
