import cv2
import os

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
    parser = argparse.ArgumentParser(description="Render angles")
    parser.add_argument("video_path", type=str, help="path to video")
    parser.add_argument("output", type=str, help="output directory")

    args = parser.parse_args()
    video_path = args.video_path  # Replace with your video file path
    output_folder = args.output  # Replace with your desired output folder
    frame_numbers = [0, 30, 60, 90]  # Replace with the frame numbers you want to extract
    extract_frames(video_path, output_folder, frame_numbers)
    #print(get_frame_rate(video_path))