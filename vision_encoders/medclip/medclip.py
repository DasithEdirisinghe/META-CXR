import torch
from torch import nn
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor

class Medclip(nn.Module):
    def __init__(self, device='cuda'):
        super(Medclip, self).__init__()  # Initialize nn.Module
        # Load the MedCLIP model and processor
        self.device = device
        self.model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT).to(self.device)
        self.model.from_pretrained()  # Load pre-trained weights
        self.processor = MedCLIPProcessor()
        
        # Define the MLP to project patch embeddings to 1408 dimensions
        self.mlp = nn.Sequential(
            nn.Linear(768, 1024),  # Bottleneck layer: from 768 (input) to 1024 (hidden)
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1408)  # Final projection to 1408 dimensions
        ).to(self.device)

    def forward(self, image):
        # Load and preprocess the image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move to device

        # Obtain pooled and patch embeddings
        with torch.no_grad():
            pool_embeds, patch_embeddings = self.model.vision_model(pixel_values=inputs['pixel_values'])
        
        # Concatenate CLS token (pool_embeds) with patch embeddings
        final_embeddings = torch.cat((pool_embeds.unsqueeze(1), patch_embeddings), dim=1)
        
        # Project the patch embeddings to 1408 dimensions using the MLP
        batch_size, num_patches_plus_cls, embedding_dim = final_embeddings.shape  # Expecting: (batch_size, num_patches + 1, 768)
        flattened_embeddings = final_embeddings.view(batch_size * num_patches_plus_cls, embedding_dim)  # Flatten for MLP
        projected_embeddings = self.mlp(flattened_embeddings)  # Project to 1408 dimensions
        projected_embeddings = projected_embeddings.view(batch_size, num_patches_plus_cls, -1)  # Reshape back

        return projected_embeddings