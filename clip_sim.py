import torch
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def calculate_clip_similarity(image_path, text):
    # Load and preprocess the image
    image = Image.open(image_path)
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    
    # Ensure inputs are correctly formatted
    if 'pixel_values' not in inputs or 'input_ids' not in inputs:
        raise ValueError("Inputs not properly processed")

    # Calculate the CLIP embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        
        # Ensure embeddings are correctly obtained
        if image_embeds.shape[0] != 1 or text_embeds.shape[0] != 1:
            raise ValueError("Embeddings shape mismatch")

        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        # Calculate cosine similarity
        similarity = torch.matmul(image_embeds, text_embeds.T).item()

    # Return the similarity score
    return similarity

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate CLIP similarity")
    parser.add_argument("directory", type=str, help="/home/scur2222/github/UVA-CV2-Project/logs/baseline_ism/corgi_albedo.png")
    parser.add_argument("text", type=str, help="a plush toy of a corgi nurse")

    args = parser.parse_args()
    scores = []
    for filename in os.listdir(args.directory):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(args.directory, filename)
            score = calculate_clip_similarity(image_path, args.text)
            print(f"Processed {filename}: score = {score}")
            scores.append(score)
    
    end_score = sum(scores) / len(scores)
    print(f"CLIP similarity score: {end_score}")