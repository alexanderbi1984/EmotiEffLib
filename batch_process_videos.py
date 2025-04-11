import os
import argparse
from process_video import process_video
import time
from datetime import datetime

def find_video_files(root_dir: str) -> list:
    """
    Recursively find all .MP4 and .MOV files in the given directory and its subdirectories.
    
    Args:
        root_dir (str): Root directory to start searching from
        
    Returns:
        list: List of paths to video files
    """
    video_files = []
    video_extensions = ('.MP4', '.MOV', '.mp4', '.mov')
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(video_extensions):
                video_files.append(os.path.join(root, file))
    
    return video_files

def process_video_batch(input_dir: str, output_dir: str) -> None:
    """
    Process all video files in the input directory and its subdirectories.
    
    Args:
        input_dir (str): Directory containing video files
        output_dir (str): Directory to save CSV outputs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all video files
    video_files = find_video_files(input_dir)
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} video files to process")
    
    # Process each video file
    for i, video_path in enumerate(video_files, 1):
        try:
            # Create output filename based on video filename
            video_filename = os.path.basename(video_path)
            output_filename = os.path.splitext(video_filename)[0] + '.csv'
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"\nProcessing video {i}/{len(video_files)}: {video_filename}")
            print(f"Output will be saved to: {output_path}")
            
            # Process the video
            start_time = time.time()
            process_video(video_path, output_path, mode='csv')
            end_time = time.time()
            
            processing_time = end_time - start_time
            print(f"Processing completed in {processing_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            continue

def main():
    parser = argparse.ArgumentParser(description='Batch process videos for emotion detection')
    parser.add_argument('input_dir', type=str, help='Directory containing video files')
    parser.add_argument('output_dir', type=str, help='Directory to save CSV outputs')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return 1
    
    # Process videos
    start_time = time.time()
    process_video_batch(args.input_dir, args.output_dir)
    end_time = time.time()
    
    total_time = end_time - start_time
    print(f"\nBatch processing completed in {total_time:.2f} seconds")
    return 0

if __name__ == "__main__":
    exit(main()) 