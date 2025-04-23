import torch
import torch.nn as nn
import torch.nn.functional as F

class MHCACLayer(nn.Module):
    def __init__(self, query_dim, cls_hidden_dim, num_heads, dropout=0.1):
        super(MHCACLayer, self).__init__()
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.cls_hidden_dim = cls_hidden_dim

        # Projections for Q, K, V
        self.W_Q_list = nn.ModuleList([nn.Linear(self.query_dim, self.query_dim) for _ in range(14)])  # Separate W_Q per query
        self.W_K_list = nn.ModuleList([nn.Linear(self.cls_hidden_dim, self.query_dim) for _ in range(self.num_heads)])  # Separate W_K per head
        self.W_V_list = nn.ModuleList([nn.Linear(self.cls_hidden_dim, self.query_dim) for _ in range(self.num_heads)])  # Separate W_V per head

        # Feed-forward network applied after attention
        self.feed_forward = nn.Sequential(
            nn.Linear(self.num_heads * self.query_dim, self.cls_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.cls_hidden_dim, self.cls_hidden_dim),
            nn.Dropout(dropout)
        )

        # Layer normalization (before and after attention, before feed-forward)
        self.norm1 = nn.LayerNorm(self.cls_hidden_dim)
        self.norm2 = nn.LayerNorm(self.cls_hidden_dim)

        # Dropout for attention
        self.attn_dropout = nn.Dropout(dropout)

        # Scaling factor for dot product attention
        self.scale = torch.sqrt(torch.tensor(self.query_dim, dtype=torch.float32))

    def forward(self, classification_queries, cls_token):
        batch_size = cls_token.size(0)
        num_abnormalities = classification_queries.size(0)

        # Project classification queries (Q)
        all_projected_queries = [self.W_Q_list[i](classification_queries[i]) for i in range(num_abnormalities)]
        Q = torch.stack(all_projected_queries, dim=0).unsqueeze(0).expand(batch_size, -1, -1)  # Shape: [batch_size, num_abnormalities, query_dim]

        # Multi-head attention (K and V projections)
        all_heads_output = []
        for h in range(self.num_heads):
            # Project the expanded CLS token to K and V
            K_h = self.W_K_list[h](cls_token)  # Shape: [batch_size, num_abnormalities, query_dim]
            V_h = self.W_V_list[h](cls_token)  # Shape: [batch_size, num_abnormalities, query_dim]
            
            # Attention: dot product Q * K_h, then softmax
            attention_scores = torch.bmm(Q, K_h.transpose(-2, -1)) / self.scale  # Shape: [batch_size, num_abnormalities, 1]
            attention_weights = F.softmax(attention_scores, dim=1)  # Shape: [batch_size, num_abnormalities, 1]

            # Apply dropout to attention weights
            attention_weights = self.attn_dropout(attention_weights)

            # Compute attention output (weighted sum of V_h)
            attention_output = torch.bmm(attention_weights, V_h)
            all_heads_output.append(attention_output)

        # Concatenate attention outputs from all heads
        attention_output = torch.cat(all_heads_output, dim=-1)  # Shape: [batch_size, num_abnormalities, num_heads * query_dim]

        # Add residual connection (residual over the original CLS token)
        attention_output = self.norm1(cls_token + attention_output)  # Residual + normalization

        # Feed-forward network with residual connection
        ff_output = self.feed_forward(attention_output)  # Pass through feed-forward network
        output = self.norm2(attention_output + ff_output)  # Apply second residual + normalization

        return output


class MHCAC(nn.Module):
    def __init__(self, cls_hidden_dim, query_dim, num_abnormalities=14, num_classes=3, num_heads=8, num_layers=2, dropout=0.1):
        super(MHCAC, self).__init__()

        # Ensure that cls_hidden_dim is divisible by num_heads
        assert cls_hidden_dim % query_dim == 0, "cls_hidden_dim must be divisible by num_heads"
        self.num_heads = cls_hidden_dim // query_dim
        self.cls_hidden_dim = cls_hidden_dim

        # Classification queries (learnable parameters for each abnormality)
        self.classification_queries = nn.Parameter(torch.randn(num_abnormalities, query_dim))  # (n_q, d_q)
        nn.init.xavier_uniform_(self.classification_queries)  # Proper Xavier initialization

        # Stack multiple MHCACLayer layers
        self.attention_layers = nn.ModuleList([MHCACLayer(query_dim, cls_hidden_dim, self.num_heads, dropout) for _ in range(num_layers)])

        # Classification heads: One for each abnormality (output 3 classes: positive, negative, uncertain)
        self.classifiers = nn.ModuleList([nn.Linear(cls_hidden_dim, num_classes) for _ in range(num_abnormalities)])

        # Dropout for the classification head
        self.dropout = nn.Dropout(dropout)

    def forward(self, cls_token):
        batch_size = cls_token.size(0)

        # Expand CLS token to match the number of classification queries
        # Shape: [batch_size, num_abnormalities, cls_hidden_dim]
        expanded_cls_token = cls_token.unsqueeze(1).expand(batch_size, len(self.classification_queries), self.cls_hidden_dim)

        # Pass through multiple MHCAC layers
        attention_output = expanded_cls_token  # Initialize with the expanded CLS token
        for layer in self.attention_layers:
            attention_output = layer(self.classification_queries, attention_output)

        # Classification logits for each abnormality
        logits = []
        for i in range(len(self.classifiers)):
            logits.append(self.classifiers[i](self.dropout(attention_output[:, i, :])))  # Shape: [batch_size, num_classes] for each abnormality

        # Stack logits to shape [batch_size, num_abnormalities, num_classes]
        logits = torch.stack(logits, dim=1)  # Shape: [batch_size, num_abnormalities, num_classes]
        
        return logits



