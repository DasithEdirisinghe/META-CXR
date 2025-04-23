import torch
import torch.nn as nn

import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionPooling, self).__init__()
        # Define a learnable query for pooling the image tokens
        self.pooling_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
    def forward(self, image_tokens):
        """
        image_tokens: Tensor of shape [batch_size, num_image_tokens, embed_dim]
        Returns a pooled representation of shape [batch_size, embed_dim]
        """
        batch_size = image_tokens.size(0)
        
        # Expand pooling query to match batch size
        pooling_query = self.pooling_query.expand(batch_size, -1, -1)  # Shape: [batch_size, 1, embed_dim]
        
        # Apply attention where the pooling query attends to the image tokens
        pooled_output, _ = self.attention(query=pooling_query, key=image_tokens, value=image_tokens)
        
        # Remove the singleton query dimension: Shape [batch_size, embed_dim]
        return pooled_output.squeeze(1)

# Example usage in the classification module
class Aggregator(nn.Module):
    def __init__(self, img_dim, query_dim, num_heads):
        super(Aggregator, self).__init__()
        
        # Assume that aggregator is used for 32 query tokens
        # self.query_aggregator = AttentionPooling(embed_dim=query_dim, num_heads=num_heads)
        
        # Attention pooling for image outputs
        self.image_attention_pooling = AttentionPooling(embed_dim=img_dim, num_heads=num_heads)
        
        # Projection layer for final classification after concatenation
        # self.fc = nn.Linear(img_dim + query_dim, 1024)
        self.fc = nn.Linear(img_dim, 1024)
        

    def forward(self, image_tokens, query_tokens):
        # Apply attention pooling for 32 query tokens
        # aggregated_query_output = self.query_aggregator(query_tokens)
        
        # Apply attention pooling for image tokens
        pooled_image_output = self.image_attention_pooling(image_tokens)
        
        # Concatenate query and image outputs
        # concatenated_output = torch.cat([aggregated_query_output, pooled_image_output], dim=1)
        
        # Project the concatenated features to a classification-friendly representation
        # classification_representation = self.fc(concatenated_output)
        classification_representation = self.fc(pooled_image_output)
        
        return classification_representation




class MeanPoolingAggregator(nn.Module):
    def __init__(self):
        super(MeanPoolingAggregator, self).__init__()

    def forward(self, query_outputs):
        # query_outputs: Shape (batch_size, 32, 768)
        
        # Mean pooling over the query dimension
        global_representation = query_outputs.mean(dim=1)  # Shape: (batch_size, 768)
        
        return global_representation


     
    
class QueryAggregator(nn.Module):
    def __init__(self, feature_dim):
        super(QueryAggregator, self).__init__()
        # Learnable attention vector (feature_dim x 1)
        self.attention_vector = nn.Parameter(torch.rand(feature_dim))
        nn.init.normal_(self.attention_vector, mean=0.0, std=0.02)

    def forward(self, query_outputs):
        # Compute attention scores by dot product between query outputs and attention vector
        # query_outputs: (batch_size, 32, feature_dim)
        # attention_vector: (feature_dim)
        
        # Compute attention scores (batch_size, 32)
        attention_scores = torch.matmul(query_outputs, self.attention_vector)
        
        # Normalize attention scores using softmax along the query dimension (dim=1)
        attention_weights = nn.Softmax(dim=1)(attention_scores)  # Shape: (batch_size, 32)

        # Apply attention weights to the query outputs
        # Reshape attention_weights for broadcasting (batch_size, 32, 1)
        attention_weights = attention_weights.unsqueeze(-1)

        # Multiply query outputs by attention weights and su   m  them to get the global representation
        global_representation = (attention_weights * query_outputs).mean(dim=1)  # Shape: (batch_size, feature_dim)
        
        return global_representation
    
