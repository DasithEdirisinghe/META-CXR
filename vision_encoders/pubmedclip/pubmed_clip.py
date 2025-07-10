import torch
from torch import nn
from transformers import CLIPModel, CLIPProcessor

class Pubmedclip(nn.Module):
    def __init__(self, aug = None, device='cuda'):
        super(Pubmedclip, self).__init__()  # Initialize nn.Module
        # Load the pre-trained PubMedCLIP model and processor
        self.device = device
        self.aug = aug
        self.model_name = "flaviagiammarino/pubmed-clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        
        # Define the MLP to project patch embeddings to 1408 dimensions
        self.mlp = nn.Sequential(
            nn.Linear(768, 1024),  # Bottleneck layer: from 768 (input) to 1024 (hidden)
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1408)  # Final projection to 1408 dimensions
        ).to(self.device)

    def forward(self, image, apply_aug = True):
        # Load and preprocess the image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move to device
        
        # inputs = inputs['pixel_values'].squeeze(0)
        inputs = inputs['pixel_values']

        if apply_aug and self.aug is not None:
             inputs = self.aug(inputs)
             
        # Obtain patch embeddings
        with torch.no_grad():
            vision_outputs = self.model.vision_model(pixel_values=inputs, output_hidden_states=True)
            patch_embeddings = vision_outputs.hidden_states[-1]  # Last layer's patch embeddings
        
        # Project the patch embeddings to 1408 dimensions using the MLP
        batch_size, num_patches, embedding_dim = patch_embeddings.shape  # Expected: (batch_size, num_patches, 768)
        patch_embeddings_clone = patch_embeddings.view(batch_size * num_patches, embedding_dim)  # Flatten for MLP
        projected_embeddings = self.mlp(patch_embeddings_clone)  # Project to 1408 dimensions
        projected_embeddings = projected_embeddings.view(batch_size, num_patches, -1)  # Reshape back to (batch_size, num_patches, 1408)

        return patch_embeddings, projected_embeddings
