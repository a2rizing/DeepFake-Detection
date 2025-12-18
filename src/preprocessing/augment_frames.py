"""
Apply data augmentation to extracted video frames.

Generates 5 augmented versions per video:
1. Temporal - Frame sampling (every 2nd/3rd frame)
2. Flip/Rotation - Horizontal flip for Front views, rotation for Side views
3. Brightness - Random brightness and contrast adjustments
4. Blur - Gaussian blur and noise to simulate different video qualities
5. Color Jitter - Hue, saturation, and value shifts

Usage:
    python src/preprocessing/augment_frames.py
    python src/preprocessing/augment_frames.py --input data/frames --output data/frames_augmented
"""

import cv2
import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A
import random


def get_augmentation_pipelines(view_type: str) -> dict:
    """
    Get augmentation pipelines based on view type.
    
    Args:
        view_type: 'F' for front view, 'S' for side view
    
    Returns:
        Dictionary of augmentation name -> albumentations pipeline
    """
    # Common augmentations for both views
    augmentations = {
        'aug1_temporal': None,  # Handled separately (frame sampling)
        
        'aug3_brightness': A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=1.0
            ),
        ]),
        
        'aug4_blur': A.Compose([
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
            ], p=0.7),
            A.GaussNoise(var_limit=(10, 50), p=0.5),
        ]),
        
        'aug5_colorjitter': A.Compose([
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            ),
        ]),
    }
    
    # View-specific augmentation for aug2
    if view_type == 'F':
        # Front view: horizontal flip is fine (symmetric)
        augmentations['aug2_flip'] = A.Compose([
            A.HorizontalFlip(p=1.0),
        ])
    else:
        # Side view: use rotation instead (flip would reverse walking direction)
        augmentations['aug2_rotation'] = A.Compose([
            A.Rotate(limit=(-10, 10), p=1.0, border_mode=cv2.BORDER_REFLECT),
        ])
    
    return augmentations


def apply_augmentation(image: np.ndarray, transform: A.Compose) -> np.ndarray:
    """Apply an albumentations transform to an image."""
    if transform is None:
        return image
    
    augmented = transform(image=image)
    return augmented['image']


def process_frames_folder(
    input_folder: str,
    output_base: str,
    video_name: str,
    view_type: str,
    person_name: str
):
    """
    Process all frames in a folder and create augmented versions.
    
    Args:
        input_folder: Path to folder with original frames
        output_base: Base output directory for augmented frames
        video_name: Name of the video (e.g., 'Arhaan_F1')
        view_type: 'F' or 'S'
        person_name: Person's name
    """
    # Get all frame files
    frame_files = sorted(Path(input_folder).glob("*.jpg"))
    if not frame_files:
        frame_files = sorted(Path(input_folder).glob("*.png"))
    
    if not frame_files:
        print(f"[WARNING] No frames found in {input_folder}")
        return
    
    # Get augmentation pipelines
    augmentations = get_augmentation_pipelines(view_type)
    
    # Output path base: output_base/PersonName/ViewType/
    output_path_base = os.path.join(output_base, person_name, view_type)
    
    # 1. Copy original frames
    original_output = os.path.join(output_path_base, f"{video_name}_original")
    os.makedirs(original_output, exist_ok=True)
    
    for frame_path in frame_files:
        frame = cv2.imread(str(frame_path))
        output_file = os.path.join(original_output, frame_path.name)
        cv2.imwrite(output_file, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # 2. Temporal augmentation (sample every 2nd frame)
    temporal_output = os.path.join(output_path_base, f"{video_name}_aug1_temporal")
    os.makedirs(temporal_output, exist_ok=True)
    
    for i, frame_path in enumerate(frame_files):
        if i % 2 == 0:  # Every 2nd frame
            frame = cv2.imread(str(frame_path))
            output_file = os.path.join(temporal_output, f"frame_{i//2:04d}.jpg")
            cv2.imwrite(output_file, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # 3-5. Apply other augmentations
    for aug_name, transform in augmentations.items():
        if aug_name == 'aug1_temporal':
            continue  # Already handled above
        
        aug_output = os.path.join(output_path_base, f"{video_name}_{aug_name}")
        os.makedirs(aug_output, exist_ok=True)
        
        # Set a consistent random seed for this augmentation
        random.seed(42)
        np.random.seed(42)
        
        for frame_path in frame_files:
            frame = cv2.imread(str(frame_path))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            augmented = apply_augmentation(frame_rgb, transform)
            augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
            
            output_file = os.path.join(aug_output, frame_path.name)
            cv2.imwrite(output_file, augmented_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


def process_all_frames(input_dir: str, output_dir: str):
    """
    Process all frame folders and create augmented versions.
    
    Expected input structure:
        input_dir/{PersonName}/{ViewType}/{VideoName}/frame_XXXX.jpg
    
    Output structure:
        output_dir/{PersonName}/{ViewType}/{VideoName}_{augmentation}/frame_XXXX.jpg
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return
    
    # Find all video folders (they contain the frames)
    video_folders = []
    for person_dir in input_path.iterdir():
        if not person_dir.is_dir():
            continue
        for view_dir in person_dir.iterdir():
            if not view_dir.is_dir():
                continue
            for video_dir in view_dir.iterdir():
                if video_dir.is_dir():
                    video_folders.append({
                        'path': str(video_dir),
                        'person': person_dir.name,
                        'view': view_dir.name,
                        'video_name': video_dir.name
                    })
    
    if not video_folders:
        print(f"[ERROR] No video frame folders found in {input_dir}")
        print("[INFO] Expected structure: {PersonName}/{ViewType}/{VideoName}/")
        return
    
    print(f"[INFO] Found {len(video_folders)} video folders to process")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Each video will generate 6 versions (1 original + 5 augmented)")
    print("-" * 60)
    
    for video_info in tqdm(video_folders, desc="Augmenting videos"):
        print(f"\n[INFO] Processing: {video_info['person']}/{video_info['view']}/{video_info['video_name']}")
        
        process_frames_folder(
            input_folder=video_info['path'],
            output_base=output_dir,
            video_name=video_info['video_name'],
            view_type=video_info['view'],
            person_name=video_info['person']
        )
        
        print(f"       Created 6 versions (original + 5 augmentations)")
    
    print("\n" + "=" * 60)
    print(f"[DONE] Augmentation complete!")
    print(f"[DONE] Total versions created: {len(video_folders) * 6}")
    print(f"[DONE] Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Augment extracted video frames")
    parser.add_argument("--input", type=str, default="data/frames",
                        help="Input directory with extracted frames (default: data/frames)")
    parser.add_argument("--output", type=str, default="data/frames_augmented",
                        help="Output directory for augmented frames (default: data/frames_augmented)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("VIDEO FRAME DATA AUGMENTATION")
    print("=" * 60)
    print("\nAugmentation types:")
    print("  1. Temporal    - Frame sampling (every 2nd frame)")
    print("  2. Flip/Rotate - Horizontal flip (F) or rotation (S)")
    print("  3. Brightness  - Random brightness/contrast adjustments")
    print("  4. Blur        - Gaussian blur + noise")
    print("  5. ColorJitter - Hue/saturation/value shifts")
    print()
    
    process_all_frames(args.input, args.output)


if __name__ == "__main__":
    main()
