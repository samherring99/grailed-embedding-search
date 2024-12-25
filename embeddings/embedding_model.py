import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
import clip

class ProductEmbeddingModel:
    def __init__(self):
        # Load CLIP model for both image and text embeddings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Freeze the model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
    def get_image_embedding(self, image_url):
        try:
            # Download and process image
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                
            return image_features.cpu().numpy()
            
        except Exception as e:
            print(f"Error processing image {image_url}: {str(e)}")
            return None
            
    def get_text_embedding(self, text):
        try:
            text = clip.tokenize([text]).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.encode_text(text)
                
            return text_features.cpu().numpy()
            
        except Exception as e:
            print(f"Error processing text: {str(e)}")
            return None