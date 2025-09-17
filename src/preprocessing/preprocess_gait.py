"""
Preprocess gait keypoint CSVs into fixed-length feature arrays for training.

Usage examples:
# Process all csvs in data/keypoints/ and save to data/processed/
python src/preprocess_gait.py --input_glob "data/*.csv" --out_dir "data/processed" --target_frames 64

# Process single file and specify person label
python src/preprocess_gait.py --input_glob "data/abhishek_walk1.csv" --label abhishek
"""
import os
import glob
import json
import argparse
import numpy as np
import pandas as pd

# mediapipe is used only for landmark name -> index mapping (safer than hardcoding)
try:
    import mediapipe as mp
    PoseLandmark = mp.solutions.pose.PoseLandmark
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False
    PoseLandmark = None

EPS = 1e-8

def csv_to_xy_array(csv_path, video_name=None):
    """
    Read a single csv from extract_gait.py and return numpy (T, L, 2) for (x,y).
    If video_name is provided, filter for that specific video from the combined CSV.
    """
    df = pd.read_csv(csv_path)
    
    # If video_name is provided, filter the dataframe for that specific video
    if video_name is not None and 'video_name' in df.columns:
        df = df[df['video_name'] == video_name].copy()
        if len(df) == 0:
            raise ValueError(f"No data found for video: {video_name}")
        # Reset frame numbers to start from 1
        df = df.reset_index(drop=True)
    
    cols = df.columns.tolist()
    
    # Handle both old format (frame only) and new format (video_name, frame)
    if 'video_name' in cols:
        # New format: video_name, frame, x_0, y_0, z_0, vis_0, ...
        n_landmarks = int((len(cols) - 2) / 4)  # subtract 2 for video_name and frame
    else:
        # Old format: frame, x_0, y_0, z_0, vis_0, ...
        n_landmarks = int((len(cols) - 1) / 4)  # subtract 1 for frame only
    
    xs = [f"x_{i}" for i in range(n_landmarks)]
    ys = [f"y_{i}" for i in range(n_landmarks)]
    
    if not set(xs).issubset(cols) or not set(ys).issubset(cols):
        raise ValueError(f"CSV {csv_path} doesn't have expected x_i/y_i columns.")
    
    x_arr = df[xs].values  # (T, L)
    y_arr = df[ys].values  # (T, L)
    T = x_arr.shape[0]
    arr = np.zeros((T, n_landmarks, 2), dtype=np.float32)
    arr[:, :, 0] = x_arr
    arr[:, :, 1] = y_arr
    return arr

def get_landmark_index(name):
    """
    Return mediapipe PoseLandmark index for a name, if mediapipe is available.
    Example: get_landmark_index('LEFT_HIP') -> integer
    """
    if not MP_AVAILABLE:
        raise RuntimeError("mediapipe not installed; landmark name lookup requires mediapipe.")
    try:
        return PoseLandmark[name].value
    except Exception as e:
        raise ValueError(f"Unknown landmark name: {name}") from e

def center_and_scale_sequence(kps_xy, left_hip_idx=None, right_hip_idx=None,
                              left_sh_idx=None, right_sh_idx=None):
    """
    kps_xy: (T, L, 2)
    center: mid-hip average across frames
    scale: mean distance mid_shoulder <-> mid_hip across frames
    Returns normalized (T, L, 2)
    """
    T, L, _ = kps_xy.shape

    # fallback indices if mediapipe not available (MediaPipe default mapping)
    if left_hip_idx is None or right_hip_idx is None:
        # these numbers correspond to mediapipe's typical index mapping
        left_hip_idx = 23
        right_hip_idx = 24
    if left_sh_idx is None or right_sh_idx is None:
        left_sh_idx = 11
        right_sh_idx = 12

    # Some frames might have zeros if detection failed; we'll compute per-frame center and torso.
    hips = kps_xy[:, [left_hip_idx, right_hip_idx], :]  # (T,2,2)
    mid_hip = np.nanmean(hips, axis=1)  # (T,2)
    shoulders = kps_xy[:, [left_sh_idx, right_sh_idx], :]  # (T,2,2)
    mid_sh = np.nanmean(shoulders, axis=1)  # (T,2)

    # center each frame
    centered = kps_xy - mid_hip[:, None, :]  # broadcast

    # torso lengths
    torso_len = np.linalg.norm(mid_sh - mid_hip, axis=1)  # (T,)
    torso_mean = np.nanmean(torso_len)
    if np.isnan(torso_mean) or torso_mean < EPS:
        torso_mean = 1.0  # fallback

    normalized = centered / (torso_mean + EPS)
    return normalized

def angle_at_point(a, b, c):
    """
    Angle at b formed by points a-b-c (in radians).
    a,b,c are (2,) numpy arrays
    """
    ba = a - b
    bc = c - b
    # handle zero-length
    na = np.linalg.norm(ba)
    nb = np.linalg.norm(bc)
    denom = (na * nb) + EPS
    cosang = np.dot(ba, bc) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.arccos(cosang)

def compute_angles_sequence(kps_xy, angle_triplets=None):
    """
    Compute angles for a list of (idx_a, idx_b, idx_c) per frame.
    Returns angles array (T, num_angles)
    """
    T, L, _ = kps_xy.shape
    if angle_triplets is None:
        # Default (MediaPipe indices): knees and elbows
        # Left: hip(23)-knee(25)-ankle(27), Right: hip(24)-knee(26)-ankle(28)
        # Left elbow: shoulder(11)-elbow(13)-wrist(15), Right elbow: 12-14-16
        angle_triplets = [
            (get_idx('LEFT_HIP', 23), get_idx('LEFT_KNEE', 25), get_idx('LEFT_ANKLE', 27)),
            (get_idx('RIGHT_HIP', 24), get_idx('RIGHT_KNEE', 26), get_idx('RIGHT_ANKLE', 28)),
            (get_idx('LEFT_SHOULDER', 11), get_idx('LEFT_ELBOW', 13), get_idx('LEFT_WRIST', 15)),
            (get_idx('RIGHT_SHOULDER', 12), get_idx('RIGHT_ELBOW', 14), get_idx('RIGHT_WRIST', 16)),
        ]
    num_angles = len(angle_triplets)
    angles = np.zeros((T, num_angles), dtype=np.float32)
    for t in range(T):
        for i, (a, b, c) in enumerate(angle_triplets):
            A = kps_xy[t, a, :]
            B = kps_xy[t, b, :]
            C = kps_xy[t, c, :]
            angles[t, i] = angle_at_point(A, B, C)
    return angles  # radians

def get_idx(name, fallback):
    """Helper: use mediapipe mapping if available, else fallback index"""
    if MP_AVAILABLE:
        try:
            return PoseLandmark[name].value
        except Exception:
            return fallback
    else:
        return fallback

def resample_sequence(seq, target_len):
    """
    seq: (T, D) returns (target_len, D) by linear interpolation per channel
    """
    T, D = seq.shape
    if T == target_len:
        return seq.copy()
    # old timeline
    old_idx = np.linspace(0, 1, T)
    new_idx = np.linspace(0, 1, target_len)
    resampled = np.zeros((target_len, D), dtype=seq.dtype)
    for d in range(D):
        resampled[:, d] = np.interp(new_idx, old_idx, seq[:, d])
    return resampled

def sequence_to_feature_matrix(kps_xy, include_angles=True, target_frames=64):
    """
    Convert a (T, L, 2) sequence to fixed-length feature matrix (target_frames, D)
    where D = L*2 (+ num_angles if included)
    """
    # normalize
    kps_norm = center_and_scale_sequence(kps_xy)
    T, L, _ = kps_norm.shape
    # flatten x,y per frame
    flat = kps_norm.reshape((T, L * 2))  # (T, L*2)
    if include_angles:
        angles = compute_angles_sequence(kps_norm)
        fps_feat = np.concatenate([flat, angles], axis=1)  # (T, L*2 + A)
    else:
        fps_feat = flat
    feat_resampled = resample_sequence(fps_feat, target_frames)
    return feat_resampled  # (target_frames, D)

def infer_label_from_filename(path):
    """
    Default heuristic: label = filename prefix before first underscore or dash.
    e.g. "abhishek_walk1.csv" -> "abhishek"
    """
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    label = name.split('_')[0].split('-')[0]
    return label

def build_dataset_from_glob(input_glob, out_dir, target_frames=64, include_angles=True):
    """
    Original function for processing multiple separate CSV files using glob patterns.
    """
    files = sorted(glob.glob(input_glob))
    if len(files) == 0:
        raise RuntimeError(f"No files matched: {input_glob}")

    X_list = []
    y_list = []
    label_map = {}
    next_label = 0

    for p in files:
        print(f"[INFO] Processing {p}")
        try:
            arr = csv_to_xy_array(p)  # (T, L, 2)
        except Exception as e:
            print(f"[WARN] Skipping {p} due to parse error: {e}")
            continue
        feat = sequence_to_feature_matrix(arr, include_angles=include_angles, target_frames=target_frames)  # (T, D)
        # get label
        label_name = infer_label_from_filename(p)
        if label_name not in label_map:
            label_map[label_name] = next_label
            next_label += 1
        label_id = label_map[label_name]
        X_list.append(feat)     # (T, D)
        y_list.append(label_id)

    X = np.stack(X_list, axis=0)  # (N, T, D)
    y = np.array(y_list, dtype=np.int64)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)
    with open(os.path.join(out_dir, "labels.json"), "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"[DONE] Saved X.npy ({X.shape}), y.npy ({y.shape}), labels.json -> {out_dir}")

def get_video_names_from_csv(csv_path):
    """
    Get all unique video names from a combined CSV file.
    Returns empty list if no video_name column exists.
    """
    df = pd.read_csv(csv_path)
    if 'video_name' in df.columns:
        return df['video_name'].unique().tolist()
    else:
        return []

def build_dataset_from_combined_csv(csv_path, out_dir, target_frames=64, include_angles=True):
    """
    Build dataset from a single combined CSV file containing multiple videos.
    Each video becomes a separate sample.
    """
    print(f"[INFO] Processing combined CSV: {csv_path}")
    
    # Get all video names from the CSV
    video_names = get_video_names_from_csv(csv_path)
    
    if not video_names:
        raise ValueError(f"No video_name column found in {csv_path}. Use build_dataset_from_glob for single video CSVs.")
    
    print(f"[INFO] Found {len(video_names)} videos: {video_names}")
    
    X_list = []
    y_list = []
    label_map = {}
    next_label = 0
    
    for video_name in video_names:
        print(f"[INFO] Processing video: {video_name}")
        try:
            arr = csv_to_xy_array(csv_path, video_name=video_name)  # (T, L, 2)
            print(f"[INFO] Video {video_name}: {arr.shape[0]} frames, {arr.shape[1]} landmarks")
        except Exception as e:
            print(f"[WARN] Skipping video {video_name} due to error: {e}")
            continue
            
        feat = sequence_to_feature_matrix(arr, include_angles=include_angles, target_frames=target_frames)  # (T, D)
        
        # Use video name as label
        label_name = video_name
        if label_name not in label_map:
            label_map[label_name] = next_label
            next_label += 1
        label_id = label_map[label_name]
        
        X_list.append(feat)     # (T, D)
        y_list.append(label_id)
        print(f"[INFO] Video {video_name} -> label {label_id}, feature shape: {feat.shape}")
    
    if not X_list:
        raise RuntimeError("No valid videos processed!")
    
    X = np.stack(X_list, axis=0)  # (N, T, D)
    y = np.array(y_list, dtype=np.int64)
    
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)
    with open(os.path.join(out_dir, "labels.json"), "w") as f:
        json.dump(label_map, f, indent=2)
    
    print(f"[DONE] Saved X.npy ({X.shape}), y.npy ({y.shape}), labels.json -> {out_dir}")
    print(f"[INFO] Label mapping: {label_map}")
    return X, y, label_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", type=str, default="data/gait_keypoints.csv",
                        help="glob for input gait CSVs or path to combined CSV file")
    parser.add_argument("--out_dir", type=str, default="data/processed",
                        help="directory to save X.npy,y.npy,labels.json")
    parser.add_argument("--target_frames", type=int, default=64,
                        help="resample each sequence to this many frames")
    parser.add_argument("--no_angles", action="store_true",
                        help="disable computing joint angles (faster)")
    parser.add_argument("--force_glob", action="store_true",
                        help="force using glob pattern matching instead of combined CSV")
    args = parser.parse_args()

    include_angles = not args.no_angles
    
    # Check if input is a single file and contains video_name column (combined CSV)
    if not args.force_glob and not ('*' in args.input_glob or '?' in args.input_glob):
        if os.path.isfile(args.input_glob):
            try:
                # Check if it's a combined CSV with video_name column
                df_test = pd.read_csv(args.input_glob)
                if 'video_name' in df_test.columns:
                    print(f"[INFO] Detected combined CSV format with video_name column")
                    build_dataset_from_combined_csv(args.input_glob, args.out_dir,
                                                   target_frames=args.target_frames,
                                                   include_angles=include_angles)
                    return
            except Exception as e:
                print(f"[WARN] Could not read as combined CSV: {e}")
    
    # Fall back to glob pattern matching for multiple files
    print(f"[INFO] Using glob pattern matching for: {args.input_glob}")
    build_dataset_from_glob(args.input_glob, args.out_dir,
                            target_frames=args.target_frames,
                            include_angles=include_angles)

if __name__ == "__main__":
    main()
