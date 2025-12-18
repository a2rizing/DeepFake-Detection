"""
Reassemble extracted frames back into video files.

Takes augmented frame folders and creates MP4 videos from them.

Usage:
    python src/preprocessing/frames_to_video.py
    python src/preprocessing/frames_to_video.py --input data/frames_augmented --output data/videos_augmented
"""

import cv2
import os
import argparse
from pathlib import Path
from tqdm import tqdm


def frames_to_video(frames_folder: str, output_path: str, fps: int = 30) -> bool:
    """
    Convert a folder of frames back into a video.
    
    Args:
        frames_folder: Path to folder containing frame images
        output_path: Path for output video file
        fps: Frames per second for output video
    
    Returns:
        True if successful, False otherwise
    """
    # Get all frame files
    frame_files = sorted(Path(frames_folder).glob("*.jpg"))
    if not frame_files:
        frame_files = sorted(Path(frames_folder).glob("*.png"))
    
    if not frame_files:
        print(f"[WARNING] No frames found in {frames_folder}")
        return False
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        print(f"[ERROR] Could not read frame: {frame_files[0]}")
        return False
    
    height, width = first_frame.shape[:2]
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames to video
    for frame_path in frame_files:
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            out.write(frame)
    
    out.release()
    return True


def process_all_frames(input_dir: str, output_dir: str, fps: int = 30):
    """
    Process all frame folders and create videos.
    
    Expected input structure:
        input_dir/{PersonName}/{ViewType}/{VideoName}/frame_XXXX.jpg
    
    Output structure:
        output_dir/{PersonName}/{ViewType}/{VideoName}.mp4
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return
    
    # Find all frame folders
    frame_folders = []
    for person_dir in input_path.iterdir():
        if not person_dir.is_dir():
            continue
        for view_dir in person_dir.iterdir():
            if not view_dir.is_dir():
                continue
            for video_dir in view_dir.iterdir():
                if video_dir.is_dir():
                    frame_folders.append({
                        'path': str(video_dir),
                        'person': person_dir.name,
                        'view': view_dir.name,
                        'video_name': video_dir.name
                    })
    
    if not frame_folders:
        print(f"[ERROR] No frame folders found in {input_dir}")
        return
    
    print(f"[INFO] Found {len(frame_folders)} frame folders to convert")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] FPS: {fps}")
    print("-" * 60)
    
    success_count = 0
    
    for folder_info in tqdm(frame_folders, desc="Creating videos"):
        # Create output path
        output_path = os.path.join(
            output_dir,
            folder_info['person'],
            folder_info['view'],
            f"{folder_info['video_name']}.mp4"
        )
        
        if frames_to_video(folder_info['path'], output_path, fps):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"[DONE] Created {success_count}/{len(frame_folders)} videos")
    print(f"[DONE] Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert frame folders to videos")
    parser.add_argument("--input", type=str, default="data/frames_augmented",
                        help="Input directory with frame folders (default: data/frames_augmented)")
    parser.add_argument("--output", type=str, default="data/videos_augmented",
                        help="Output directory for videos (default: data/videos_augmented)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second for output videos (default: 30)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FRAMES TO VIDEO CONVERTER")
    print("=" * 60)
    
    process_all_frames(args.input, args.output, args.fps)


if __name__ == "__main__":
    main()
