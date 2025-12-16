"""
Create test videos using webcam for quick testing
This allows you to test the system without downloading datasets
"""

import cv2
import os
from datetime import datetime

def record_test_videos():
    """Record test videos from webcam"""
    
    print("🎥 Webcam Test Video Creator")
    print("=" * 60)
    print("\nThis will help you create test videos using your webcam.")
    print("You'll record yourself walking/moving naturally.")
    print("\nInstructions:")
    print("1. Position yourself so your full body is visible")
    print("2. Walk naturally in front of the camera")
    print("3. Press 'q' to stop recording each video")
    print("\nPress Enter to start...")
    input()
    
    os.makedirs('data/raw/real', exist_ok=True)
    os.makedirs('data/raw/fake', exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam!")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Get codec and FPS
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    for i in range(1, 6):  # Record 5 videos
        print(f"\n📹 Recording video {i}/5...")
        print("Walk naturally. Press 'q' when done (record 10-15 seconds)")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f'data/raw/real/real_{i}_{timestamp}.mp4'
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))
        
        recording = False
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Start recording after 3 seconds
            if not recording:
                cv2.putText(frame, "Get ready! Starting in 3...", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Recording', frame)
                cv2.waitKey(1000)
                recording = True
            
            # Record frame
            out.write(frame)
            frame_count += 1
            
            # Display
            cv2.putText(frame, f"RECORDING - Press 'q' to stop", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Frames: {frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Recording', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        out.release()
        print(f"✅ Video {i} saved: {output_path}")
        
        if i < 5:
            print("\nReady for next video? Press Enter...")
            input()
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("✅ Recording complete!")
    print(f"✅ Created 5 test videos in data/raw/real/")
    print("\n💡 Note: For deepfake videos, you would need actual deepfake samples.")
    print("   Or use online deepfake generators with your recorded videos.")
    print("\n🚀 Next step: Run python extract_and_train.py")
    print("=" * 60)

if __name__ == "__main__":
    try:
        record_test_videos()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nIf webcam doesn't work, please download sample videos instead.")
        print("Run: python download_sample_data.py for instructions")
