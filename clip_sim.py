import torch
import os
import cv2
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

def extract_frames(video_path, output_folder, frame_numbers):
    # Check if output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Extract the specified frames
    for frame_number in frame_numbers:
        if frame_number < 0 or frame_number >= total_frames:
            print(f"Frame number {frame_number} is out of range. Skipping.")
            continue

        # Set the position of the video to the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_number}. Skipping.")
            continue

        # Save the frame as an image file
        output_path = os.path.join(output_folder, f"frame_{frame_number}.jpg")
        cv2.imwrite(output_path, frame)
        print(f"Frame {frame_number} saved to {output_path}")

    # Release the video capture object
    cap.release()

def get_frame_rate(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Release the video capture object
    cap.release()

    return fps

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate CLIP similarity")
    parser.add_argument("directory", type=str, help="image directory")
    parser.add_argument("text", type=str, help="text prompt")
    parser.add_argument("video_path", type=str, help="path to video")

    args = parser.parse_args()
    video_path = args.video_path
    output_folder = args.directory  
    frame_numbers = [0, 30, 60, 90]  
    extract_frames(video_path, output_folder, frame_numbers)
    #print(get_frame_rate(video_path))

    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    with open(args.directory + '/clipscore.txt', 'w') as file:
        scores = []
        for filename in os.listdir(args.directory):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(args.directory, filename)
                score = calculate_clip_similarity(image_path, args.text)
                file.write(f"Processed {filename}: score = {score}" + '\n')
                scores.append(score)
    
        end_score = sum(scores) / len(scores)
        file.write(f"CLIP similarity score: {end_score}")