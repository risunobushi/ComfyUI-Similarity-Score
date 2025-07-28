import torch
import open_clip
from torchvision import transforms
import lpips
from PIL import Image
import numpy as np

class ImageSimilarityScores:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT")
    RETURN_NAMES = ("clip_similarity", "lpips_similarity")
    FUNCTION = "calculate_similarities"
    CATEGORY = "image/similarity"
    OUTPUT_NODE = False

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Initialize CLIP
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', device=self.device)
        # Initialize LPIPS
        self.lpips_model = lpips.LPIPS(net='alex').to(self.device)

    def calculate_clip_similarity(self, image1, image2):
        # Convert images to CLIP format
        image1 = self.clip_preprocess(image1).unsqueeze(0).to(self.device)
        image2 = self.clip_preprocess(image2).unsqueeze(0).to(self.device)

        # Get image features
        with torch.no_grad():
            image1_features = self.clip_model.encode_image(image1)
            image2_features = self.clip_model.encode_image(image2)

        # Calculate cosine similarity
        similarity = torch.cosine_similarity(image1_features, image2_features)
        return similarity.item()

    def calculate_lpips_similarity(self, image1, image2):
        # Convert images to LPIPS format (normalize to [-1,1])
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        image1 = transform(image1).unsqueeze(0).to(self.device)
        image2 = transform(image2).unsqueeze(0).to(self.device)

        # Calculate LPIPS distance
        with torch.no_grad():
            lpips_distance = self.lpips_model(image1, image2)
        
        return lpips_distance.item()

    def calculate_similarities(self, image1, image2):
        # Convert from ComfyUI format to PIL
        if isinstance(image1, torch.Tensor):
            image1 = image1.cpu().numpy()
        if isinstance(image2, torch.Tensor):
            image2 = image2.cpu().numpy()
            
        if image1.ndim == 4:
            image1 = image1[0]
        if image2.ndim == 4:
            image2 = image2[0]
            
        if image1.shape[0] in [1, 3]:
            image1 = np.transpose(image1, (1, 2, 0))
        if image2.shape[0] in [1, 3]:
            image2 = np.transpose(image2, (1, 2, 0))
            
        image1 = (image1 * 255).astype(np.uint8)
        image2 = (image2 * 255).astype(np.uint8)
        
        image1_pil = Image.fromarray(image1)
        image2_pil = Image.fromarray(image2)

        # Calculate similarities
        clip_score = self.calculate_clip_similarity(image1_pil, image2_pil)
        lpips_score = self.calculate_lpips_similarity(image1_pil, image2_pil)

        return (clip_score, lpips_score)
    
NODE_CLASS_MAPPINGS = {
    "ImageSimilarityScores": ImageSimilarityScores
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSimilarityScores": "Image Similarity (CLIP & LPIPS)"
}
