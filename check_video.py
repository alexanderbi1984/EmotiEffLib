import cv2
import os

def check_video_metadata(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    print("\nVideo Properties:")
    print(f"Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"Frame Count: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
    print(f"Rotation: {cap.get(cv2.CAP_PROP_ORIENTATION_META)}")
    
    # Read first frame to check actual dimensions
    ret, frame = cap.read()
    if ret:
        print(f"\nActual Frame Dimensions:")
        print(f"Frame Width: {frame.shape[1]}")
        print(f"Frame Height: {frame.shape[0]}")
        
        # Save the first frame to check orientation
        cv2.imwrite('sample_frame.jpg', frame)
        print("\nSaved sample frame as 'sample_frame.jpg'")
    
    cap.release()

if __name__ == "__main__":
    video_path = r"C:\pain\syracus\syracuse_pain_videos\001\IMG_0003.MP4"
    check_video_metadata(video_path) 