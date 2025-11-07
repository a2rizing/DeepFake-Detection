# 🚀 Quick Setup Guide - Deepfake Detection

## What I've Created For You

I've simplified everything into easy-to-run scripts. **NO COMPLICATED DOWNLOADS NEEDED!**

## 📋 Step-by-Step Instructions

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

This installs everything you need.

---

### Step 2: Get Sample Videos

You have **3 OPTIONS** (choose the easiest one for you):

#### **Option A: Download Manually (RECOMMENDED - Easiest)**

1. Create these folders (if not exist):
   ```powershell
   mkdir data\raw\real
   mkdir data\raw\fake
   ```

2. Download sample videos:

   **For REAL videos:**
   - Go to https://www.pexels.com/search/videos/walking%20person/
   - Download 5-10 videos of people walking
   - Save them as: `real_1.mp4`, `real_2.mp4`, etc. in `data/raw/real/`

   **For FAKE videos:**
   - Download from: https://github.com/yuezunli/celeb-deepfakeforensics
   - Or search "deepfake sample videos" on Google
   - Save them as: `fake_1.mp4`, `fake_2.mp4`, etc. in `data/raw/fake/`

#### **Option B: Get Instructions**
```powershell
python download_sample_data.py
```
This will show you where to download videos from.

#### **Option C: Create Your Own Test Videos**
```powershell
python create_test_videos.py
```
This will use your webcam to record test videos (you'll still need deepfake videos separately).

---

### Step 3: Extract Features & Train Model

Once you have videos in both folders, run:

```powershell
python extract_and_train.py
```

This will:
- ✅ Extract gait features from all videos
- ✅ Train multiple ML models
- ✅ Save the best model
- ✅ Show accuracy metrics

**Expected runtime:** 5-30 minutes (depending on number of videos)

---

### Step 4: Test on New Videos

```powershell
python simple_demo.py path\to\test\video.mp4
```

Example:
```powershell
python simple_demo.py data\raw\real\real_1.mp4
```

This will analyze the video and tell you if it's real or fake!

---

## 🎯 Quick Start Summary

```powershell
# 1. Install
pip install -r requirements.txt

# 2. Get videos (manual download recommended)
# Put them in data/raw/real/ and data/raw/fake/

# 3. Train
python extract_and_train.py

# 4. Test
python simple_demo.py path\to\video.mp4
```

---

## 📊 What You'll See

### During Training:
```
✅ Found 10 real videos
✅ Found 10 fake videos
🔄 Training Random Forest...
✅ Random Forest Accuracy: 0.85
🏆 BEST MODEL: Random Forest
🎯 Accuracy: 0.85
```

### During Testing:
```
🎯 RESULTS
✅ Prediction: AUTHENTIC (Real)
   Confidence: 92.5%

📊 Probability Breakdown:
   Real: 92.5%
   Fake: 7.5%
```

---

## ❓ Troubleshooting

### "Model not found"
- Run `python extract_and_train.py` first

### "No videos found"
- Make sure videos are in `data/raw/real/` and `data/raw/fake/`
- Check file extensions (.mp4, .avi, .mov)

### "Could not extract features"
- Make sure full body is visible in video
- Video should show person walking
- Check video isn't corrupted

---

## 🎓 For Your Presentation

After training, you'll have:
- ✅ Trained model with accuracy metrics
- ✅ Feature extraction pipeline
- ✅ Working demo
- ✅ Results and visualizations

You can demonstrate:
1. Upload a real video → Shows "AUTHENTIC"
2. Upload a deepfake video → Shows "DEEPFAKE"
3. Show confidence scores

---

## 💡 Need Help?

- Check `data/DOWNLOAD_SOURCES.txt` for video sources
- All code is commented and documented
- Scripts have error messages to guide you

**You don't need to understand FaceForensics or complex downloads. Just follow the 4 steps above!** 🎉
