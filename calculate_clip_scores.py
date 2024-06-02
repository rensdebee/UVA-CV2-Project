import os
import csv
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import open_clip
from open_clip import tokenizer

# Load the CLIP model
# model, _, preprocess = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k_augreg')
model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k')
model.eval()

# Directories
root_dir = 'multiview_images'
output_dir = 'clip_scores'
os.makedirs(output_dir, exist_ok=True)

# Function to calculate CLIP scores
def calculate_clip_scores(image_paths, prompt):
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        images.append(preprocess(image))
    
    image_input = torch.tensor(np.stack(images))
    text_input = tokenizer.tokenize([prompt])
    
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_input).float()
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    similarity = (100.0 * image_features @ text_features.T).cpu().numpy()
    return similarity.flatten()

# Process each subfolder in multiview_images
for model_folder in tqdm(os.listdir(root_dir), desc="Processing model folders"):
    model_path = os.path.join(root_dir, model_folder)
    output_model_path = os.path.join(output_dir, model_folder)
    os.makedirs(output_model_path, exist_ok=True)
    
    # Aggregate scores for texture and shape
    texture_scores = []
    shape_scores = []
    
    for category in ['texture', 'shape']:
        category_path = os.path.join(model_path, category)
        output_category_path = os.path.join(output_model_path, category)
        os.makedirs(output_category_path, exist_ok=True)
        
        category_prompts = os.listdir(category_path)
        prompt_scores = {}
        
        for prompt_folder in tqdm(category_prompts, desc=f"Processing {category} prompts in {model_folder}", leave=False):
            prompt_path = os.path.join(category_path, prompt_folder)
            prompt = ' '.join(prompt_folder.split('_')).lower()
            image_files = [os.path.join(prompt_path, img) for img in os.listdir(prompt_path) if img.endswith('.jpg')]
            
            if image_files:
                scores = calculate_clip_scores(image_files, prompt)
                prompt_scores[prompt_folder] = scores.mean()
                
                # Save individual image scores
                prompt_output_path = os.path.join(output_category_path, prompt_folder)
                os.makedirs(prompt_output_path, exist_ok=True)
                
                csv_path = os.path.join(prompt_output_path, 'clip_scores.csv')
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Image', 'CLIP Score'])
                    for img, score in zip(image_files, scores):
                        writer.writerow([os.path.basename(img), score])
                
        # Save prompt scores
        csv_path = os.path.join(output_category_path, 'prompt_scores.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Prompt', 'Mean CLIP Score'])
            for prompt, score in prompt_scores.items():
                writer.writerow([prompt, score])
                if category == 'texture':
                    texture_scores.append(score)
                else:
                    shape_scores.append(score)
    
    # Save texture and shape scores for the model
    model_scores_path = os.path.join(output_model_path, 'category_scores.csv')
    with open(model_scores_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Category', 'Mean CLIP Score'])
        writer.writerow(['Texture', np.mean(texture_scores)])
        writer.writerow(['Shape', np.mean(shape_scores)])

# Aggregate scores across all models
all_model_scores = []

for model_folder in os.listdir(output_dir):
    model_path = os.path.join(output_dir, model_folder)
    texture_score = np.mean(np.loadtxt(os.path.join(model_path, 'texture', 'prompt_scores.csv'), delimiter=',', skiprows=1, usecols=1))
    shape_score = np.mean(np.loadtxt(os.path.join(model_path, 'shape', 'prompt_scores.csv'), delimiter=',', skiprows=1, usecols=1))
    all_model_scores.append([model_folder, texture_score, shape_score])

# Save aggregated scores
aggregated_scores_path = os.path.join(output_dir, 'aggregated_scores.csv')
with open(aggregated_scores_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Model', 'Texture Mean CLIP Score', 'Shape Mean CLIP Score'])
    writer.writerows(all_model_scores)