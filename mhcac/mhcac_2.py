import torch
import torch.nn as nn
import torch.nn.functional as F

class MHCACLayerSimplified(nn.Module):
    def __init__(self, query_dim, num_heads, dropout=0.2, cross_attention_freq=2):
        super(MHCACLayerSimplified, self).__init__()
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.cross_attention_freq = cross_attention_freq

        # Multi-head attention for cross-attending image patches
        self.multihead_attention = nn.MultiheadAttention(embed_dim=query_dim, num_heads=num_heads, dropout = dropout ,batch_first=True)

        # Layer normalization and dropout
        self.norm = nn.LayerNorm(query_dim)
        self.dropout = nn.Dropout(dropout)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, expert_tokens, image_patches, txt_cls_token=None):
        # Expert tokens cross-attend with the text CLS token if `step` aligns with `cross_attention_freq`
        if txt_cls_token is not None:
            # Perform cross-attention with the CLS token
            txt_cls_token = txt_cls_token.unsqueeze(1).expand(-1, expert_tokens.size(1), -1)  # Expand CLS token
            attention_scores = torch.bmm(expert_tokens, txt_cls_token.transpose(-2, -1))
            attention_weights = F.softmax(attention_scores, dim=-1)
            expert_tokens = torch.bmm(attention_weights, txt_cls_token)

        # Expert tokens cross-attend with image patches
        refined_expert_tokens, _ = self.multihead_attention(
            query=expert_tokens, key=image_patches, value=image_patches
        )

        # Apply residual connection, normalization, and feed-forward network
        refined_expert_tokens = self.norm(expert_tokens + refined_expert_tokens)
        refined_expert_tokens = self.feed_forward(refined_expert_tokens) + refined_expert_tokens

        return refined_expert_tokens

class MHCACSimplified(nn.Module):
    def __init__(self, query_dim, num_abnormalities=14, num_classes=3, num_heads=8, num_layers=2, dropout=0.2, cross_attention_freq=2):
        super(MHCACSimplified, self).__init__()
        
        self.num_heads = num_heads
        self.query_dim = query_dim

        # Expert tokens (learnable parameters for each abnormality)
        self.expert_tokens = nn.Parameter(torch.randn(num_abnormalities, query_dim))
        nn.init.xavier_uniform_(self.expert_tokens)

        # Stack multiple layers
        self.attention_layers = nn.ModuleList([
            MHCACLayerSimplified(query_dim, self.num_heads, dropout, cross_attention_freq) for _ in range(num_layers)
        ])

        # Classification heads: One for each abnormality
        self.classifiers = nn.ModuleList([nn.Linear(query_dim, num_classes) for _ in range(num_abnormalities)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, txt_cls_token, image_patches, training=True):
        batch_size = image_patches.size(0)

        # Expand expert tokens to match batch size
        expanded_expert_tokens = self.expert_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        # During inference, set `cls_token` to `None` to skip cross-attention with text CLS
        txt_cls_token = txt_cls_token if training else None

        # Pass through multiple attention layers
        attention_output = expanded_expert_tokens
        for layer in self.attention_layers:
            attention_output = layer(attention_output, image_patches, txt_cls_token)

        # Classification logits for each abnormality
        logits = []
        for i in range(len(self.classifiers)):
            logits.append(self.classifiers[i](self.dropout(attention_output[:, i, :])))
        logits = torch.stack(logits, dim=1)

        return logits

