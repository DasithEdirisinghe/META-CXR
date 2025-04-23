import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossModalAttentionLayer, self).__init__()
        
        # Attention for cross-attending text embeddings
        self.text_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Attention for cross-attending image patch embeddings
        self.image_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Layer norms and feed-forward layers
        self.norm_text = nn.LayerNorm(embed_dim)
        self.norm_image = nn.LayerNorm(embed_dim)
        
        # Feed-forward layer after combining text and image attended features
        self.feed_forward = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm_combined = nn.LayerNorm(embed_dim)

    def forward(self, expert_tokens, image_patches, text_embeddings=None):
        """
        expert_tokens: Tensor of shape [batch_size, num_expert_tokens, embed_dim]
        image_patches: Tensor of shape [batch_size, num_image_patches, embed_dim]
        text_embeddings: Tensor of shape [batch_size, num_text_tokens, embed_dim] or None
        """
        
        # Check if text_embeddings are provided (for training). Skip if not (for inference).
        if text_embeddings is not None:
            # Cross-attention with text embeddings
            attended_text, _ = self.text_attention(query=expert_tokens, key=text_embeddings, value=text_embeddings)
            attended_text = self.norm_text(attended_text + expert_tokens)  # Residual connection and normalization
        else:
            # If no text embeddings, use expert tokens as they are
            attended_text = expert_tokens
        
        # Cross-attention with image patch embeddings
        attended_image, _ = self.image_attention(query=attended_text, key=image_patches, value=image_patches)
        attended_image = self.norm_image(attended_image + attended_text)  # Residual connection and normalization

        # Concatenate attended text (or expert tokens directly if no text) and image features
        combined_features = torch.cat([attended_text, attended_image], dim=-1)  # Shape: [batch_size, num_expert_tokens, 2 * embed_dim]
        output = self.feed_forward(combined_features)  # Shape: [batch_size, num_expert_tokens, embed_dim]
        output = self.norm_combined(output + attended_image)  # Final residual connection and normalization

        return output  # Shape: [batch_size, num_expert_tokens, embed_dim]


# Example Usage
class CrossModalAttentionClassifier(nn.Module):
    def __init__(self, embed_dim=512, num_abnormalities=14, num_heads=8, num_classes=3, num_layers = 6, dropout=0.1):
        super(CrossModalAttentionClassifier, self).__init__()
        
        # Expert tokens (one for each abnormality)
        self.expert_tokens = nn.Parameter(torch.randn(num_abnormalities, embed_dim))
        nn.init.xavier_uniform_(self.expert_tokens)  # Initialize expert tokens

        # Cross-modal attention layer
        self.attention_layers = nn.ModuleList([
             CrossModalAttentionLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        ])

        # Classification heads for each expert token
        self.classifiers = nn.ModuleList([nn.Linear(embed_dim, num_classes) for _ in range(num_abnormalities)])

    def forward(self, image_patches,  text_embeddings=None):
        batch_size = image_patches.size(0)

        # Expand expert tokens to match batch size
        expert_tokens = self.expert_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        if text_embeddings is not None:
            text_embeddings = text_embeddings.unsqueeze(1)

        # Apply cross-modal attention
        expert_features = expert_tokens
        for layer in self.attention_layers:
            expert_features = layer(expert_features, image_patches, text_embeddings)

        # Classification for each abnormality
        logits = []
        for i in range(len(self.classifiers)):
            logits.append(self.classifiers[i](expert_features[:, i, :]))  # Each expert token's final representation is classified
        logits = torch.stack(logits, dim=1)  # Shape: [batch_size, num_expert_tokens, num_classes]

        return logits

