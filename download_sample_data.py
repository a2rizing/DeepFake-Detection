"""
Simple script to download sample videos for deepfake detection
No complicated setup - just downloads publicly available test videos
"""

import os
import requests
from tqdm import tqdm
import gdown

def download_file(url, destination):
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = file.write(chunk)
                bar.update(size)
        
        print(f"✅ Downloaded: {destination}")
        return True
    except Exception as e:
        print(f"❌ Error downloading {destination}: {e}")
        return False

def download_from_gdrive(file_id, destination):
    """Download from Google Drive"""
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, destination, quiet=False)
        print(f"✅ Downloaded: {destination}")
        return True
    except Exception as e:
        print(f"❌ Error downloading from GDrive: {e}")
        return False

def main():
    """Download sample videos for testing"""
    
    print("🎬 Downloading Sample Videos for Deepfake Detection")
    print("=" * 60)
    
    # Create directories
    os.makedirs('data/raw/real', exist_ok=True)
    os.makedirs('data/raw/fake', exist_ok=True)
    
    print("\n📂 Directory structure created!")
    print("   data/raw/real/  - For authentic videos")
    print("   data/raw/fake/  - For deepfake videos")
    
    print("\n" + "=" * 60)
    print("IMPORTANT: Download Instructions")
    print("=" * 60)
    print("\nSince automated downloads can be unreliable, please manually download")
    print("sample videos from these sources:\n")
    
    print("📹 REAL VIDEOS (place in data/raw/real/):")
    print("   1. Pexels (free stock videos): https://www.pexels.com/search/videos/walking%20person/")
    print("   2. Pixabay: https://pixabay.com/videos/search/person%20walking/")
    print("   - Download 5-10 videos of people walking")
    print("   - Rename them: real_1.mp4, real_2.mp4, etc.\n")
    
    print("🎭 DEEPFAKE VIDEOS (place in data/raw/fake/):")
    print("   1. Celeb-DF samples: https://github.com/yuezunli/celeb-deepfakeforensics")
    print("   2. DFDC samples: https://www.kaggle.com/c/deepfake-detection-challenge")
    print("   3. Or create your own using: https://github.com/iperov/DeepFaceLive")
    print("   - Download 5-10 deepfake videos")
    print("   - Rename them: fake_1.mp4, fake_2.mp4, etc.\n")
    
    print("=" * 60)
    print("\n🎯 Quick Start Option:")
    print("If you want to test the system immediately, I can help you create")
    print("synthetic test data using your webcam or screen recordings!")
    print("\nRun: python create_test_videos.py")
    
    # Create a sample list file
    with open('data/DOWNLOAD_SOURCES.txt', 'w') as f:
        f.write("""
Deepfake Detection - Video Sources
===================================

REAL VIDEOS:
- Pexels: https://www.pexels.com/search/videos/walking%20person/
- Pixabay: https://pixabay.com/videos/search/person%20walking/
- YouTube (use pytube): Search for "person walking full body"

DEEPFAKE VIDEOS:
- Celeb-DF: https://github.com/yuezunli/celeb-deepfakeforensics
  (Register and download sample videos)
  
- DFDC Dataset: https://www.kaggle.com/c/deepfake-detection-challenge
  (Requires Kaggle account)
  
- FaceForensics++: https://github.com/ondyari/FaceForensics
  (Research dataset, requires application)

QUICK TEST:
- Use DeepFaceLive to create your own deepfakes: 
  https://github.com/iperov/DeepFaceLive
  
- Or use online tools like:
  https://www.myheritage.com/deep-nostalgia (for testing)

NAMING CONVENTION:
- Real videos: real_1.mp4, real_2.mp4, ...
- Fake videos: fake_1.mp4, fake_2.mp4, ...

Place them in:
- data/raw/real/
- data/raw/fake/
        """)
    
    print("\n✅ Instructions saved to: data/DOWNLOAD_SOURCES.txt")
    print("\n" + "=" * 60)
    print("Once you have videos downloaded, run:")
    print("   python extract_and_train.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
