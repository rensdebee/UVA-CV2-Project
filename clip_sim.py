import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def calculate_clip_similarity(image_path, text):
    # Load and preprocess the image
    image = Image.open(image_path)
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)

    # Calculate the CLIP embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

    # Return the similarity score
    return probs[0][0].item()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate CLIP similarity")
    parser.add_argument("image_path", type=str, help="/home/scur2222/github/UVA-CV2-Project/logs/baseline_ism/corgi_albedo.png")
    parser.add_argument("text", type=str, help="a plush toy of a corgi nurse")

    args = parser.parse_args()

    score = calculate_clip_similarity(args.image_path, args.text)
    print(f"CLIP similarity score: {score}")