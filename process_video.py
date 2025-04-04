import cv2
import numpy as np
import pandas as pd
from emotiefflib.facial_analysis import EmotiEffLibRecognizer
import torch
from typing import List, Dict, Tuple
import time
import argparse
import os

def rotate_frame(frame, rotation_angle):
    """
    Rotate a frame based on the orientation metadata.
    
    Args:
        frame: The input frame
        rotation_angle: The rotation angle in degrees
    
    Returns:
        The rotated frame
    """
    if rotation_angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation_angle == 270 or rotation_angle == -90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return e_x / e_x.sum()

def process_video(video_path: str, output_path: str, mode: str = 'csv') -> None:
    """
    Process a video file and output emotion predictions.
    
    Args:
        video_path (str): Path to the input video file
        output_path (str): Path to save the output file (CSV or video)
        mode (str): Output mode - 'csv' or 'video'
    """
    # Check if video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Initialize the emotion recognizer
    recognizer = EmotiEffLibRecognizer(
        engine="torch",
        model_name="enet_b0_8_best_vgaf",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Get the emotion class mapping
    emotion_classes = list(recognizer.idx_to_emotion_class.values())
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Adjust dimensions for rotated video
    if rotation in [90, 270, -90]:
        width, height = height, width
    
    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Rotation: {rotation} degrees")
    print(f"Emotion classes: {emotion_classes}")
    
    # Initialize results storage for CSV mode
    results = []
    frame_number = 0
    
    # Initialize video writer for video mode
    if mode == 'video':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)  # Use rotated dimensions
        )
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Apply rotation if needed
        if rotation:
            frame = rotate_frame(frame, rotation)
            
        # Convert frame to RGB for the model
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get predictions (logits)
        emotions, scores = recognizer.predict_emotions(rgb_frame, logits=True)
        
        if emotions and len(emotions) > 0:
            # Get the first face prediction
            emotion = emotions[0]
            logits = scores[0]
            # Convert logits to probabilities
            probabilities = softmax(logits)
            
            timestamp = frame_number / fps
            
            if mode == 'csv':
                # Create result dictionary with all emotions' scores
                result = {
                    'frame': frame_number,
                    'timestamp': timestamp,
                    'predicted_emotion': emotion,
                }
                
                # Add logits and probabilities for each emotion class
                for i, emotion_label in enumerate(emotion_classes):
                    result[f'logit_{emotion_label}'] = float(logits[i])
                    result[f'prob_{emotion_label}'] = float(probabilities[i])
                
                results.append(result)
            else:
                # Draw on frame for video output
                # Note: If you need bounding boxes, you'll need to get them from a face detection step
                label = f"{emotion} ({probabilities.max():.2f})"
                cv2.putText(frame, label, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                out.write(frame)
        
        frame_number += 1
        if frame_number % 100 == 0:
            print(f"Processed {frame_number}/{total_frames} frames")
    
    # Clean up
    cap.release()
    if mode == 'video':
        out.release()
    
    # Save CSV if in CSV mode
    if mode == 'csv' and results:
        df = pd.DataFrame(results)
        
        # Print statistics for each emotion
        print("\nEmotion Statistics:")
        for emotion in emotion_classes:
            print(f"\n{emotion}:")
            print(f"Logits - Min: {df[f'logit_{emotion}'].min():.4f}, Max: {df[f'logit_{emotion}'].max():.4f}, Mean: {df[f'logit_{emotion}'].mean():.4f}")
            print(f"Probs  - Min: {df[f'prob_{emotion}'].min():.4f}, Max: {df[f'prob_{emotion}'].max():.4f}, Mean: {df[f'prob_{emotion}'].mean():.4f}")
        
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
    else:
        print(f"Processed video saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Process video for emotion detection')
    parser.add_argument('input_video', type=str, help='Path to input video file')
    parser.add_argument('output_path', type=str, help='Path to save output (CSV or video)')
    parser.add_argument('--mode', type=str, choices=['csv', 'video'], default='csv',
                      help='Output mode: csv (default) or video')
    
    args = parser.parse_args()
    
    try:
        process_video(args.input_video, args.output_path, args.mode)
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())