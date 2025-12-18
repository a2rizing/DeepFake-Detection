"""
Extract frames from videos in data/videos/ directory.

Usage:
    python src/preprocessing/video_to_frames.py
    python src/preprocessing/video_to_frames.py --video data/videos/Arhaan_F1.mp4 --output data/frames_test
"""

import cv2
import os
import argparse
import re
from pathlib import Path
from tqdm import tqdm


def parse_video_name(video_path: str) -> dict:
    """
    Parse video filename to extract person name and view type.
    
    Naming convention: {PersonName}_{ViewType}{Number}.mp4
    Example: Arhaan_F1.mp4 -> {'person': 'Arhaan', 'view': 'F', 'number': '1', 'full_name': 'Arhaan_F1'}
    """
    basename = Path(video_path).stem  # e.g., "Arhaan_F1"
    
    # Pattern: PersonName_ViewTypeNumber (e.g., Arhaan_F1, Vedant2_S3)
    match = re.match(r'^(.+?)_([FS])(\d+)$', basename)
    
    if match:
        return {
            'person': match.group(1),
            'view': match.group(2),
            'number': match.group(3),
            'full_name': basename
        }
    else:
        # Fallback: treat entire name as person, unknown view
        return {
            'person': basename,
            'view': 'unknown',
            'number': '1',
            'full_name': basename
        }


def extract_frames(video_path: str, output_dir: str, frame_interval: int = 1) -> int:
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save frames
        frame_interval: Extract every Nth frame (1 = all frames)
    
    Returns:
        Number of frames extracted
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return 0
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    frame_count = 0
    saved_count = 0
    
    # Progress bar for this video
    pbar = tqdm(total=total_frames, desc=f"  Extracting", unit="frame", leave=False)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    return saved_count


def process_all_videos(input_dir: str, output_dir: str, frame_interval: int = 1):
    """
    Process all videos in the input directory.
    
    Output structure:
        output_dir/{PersonName}/{ViewType}/{VideoName}/frame_XXXX.jpg
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    # Find all video files
    for ext in video_extensions:
        video_files.extend(Path(input_dir).glob(f"*{ext}"))
        video_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    if not video_files:
        print(f"[ERROR] No video files found in {input_dir}")
        return
    
    print(f"[INFO] Found {len(video_files)} videos to process")
    print(f"[INFO] Output directory: {output_dir}")
    print("-" * 50)
    
    total_frames = 0
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        video_path = str(video_path)
        info = parse_video_name(video_path)
        
        # Create output path: output_dir/PersonName/ViewType/VideoName/
        video_output_dir = os.path.join(
            output_dir,
            info['person'],
            info['view'],
            info['full_name']
        )
        
        print(f"\n[INFO] {info['full_name']} -> {info['person']}/{info['view']}/")
        
        frames_extracted = extract_frames(video_path, video_output_dir, frame_interval)
        total_frames += frames_extracted
        
        print(f"       Extracted {frames_extracted} frames")
    
    print("\n" + "=" * 50)
    print(f"[DONE] Total frames extracted: {total_frames}")
    print(f"[DONE] Output saved to: {output_dir}")


def process_single_video(video_path: str, output_dir: str, frame_interval: int = 1):
    """Process a single video file."""
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return
    
    info = parse_video_name(video_path)
    
    # Create output path
    video_output_dir = os.path.join(
        output_dir,
        info['person'],
        info['view'],
        info['full_name']
    )
    
    print(f"[INFO] Processing: {info['full_name']}")
    print(f"[INFO] Person: {info['person']}, View: {info['view']}")
    
    frames_extracted = extract_frames(video_path, video_output_dir, frame_interval)
    
    print(f"[DONE] Extracted {frames_extracted} frames to {video_output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos")
    parser.add_argument("--video", type=str, help="Path to a single video file (optional)")
    parser.add_argument("--input", type=str, default="data/videos", 
                        help="Input directory containing videos (default: data/videos)")
    parser.add_argument("--output", type=str, default="data/frames",
                        help="Output directory for frames (default: data/frames)")
    parser.add_argument("--interval", type=int, default=1,
                        help="Extract every Nth frame (default: 1 = all frames)")
    
    args = parser.parse_args()
    
    if args.video:
        # Process single video
        process_single_video(args.video, args.output, args.interval)
    else:
        # Process all videos in directory
        process_all_videos(args.input, args.output, args.interval)


if __name__ == "__main__":
    main()
