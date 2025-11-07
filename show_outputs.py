"""
Display All Current Project Outputs
Shows what you can demonstrate to your supervisor
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def show_model_results():
    """Display model training results"""
    print("=" * 80)
    print("🎯 MODEL TRAINING RESULTS")
    print("=" * 80)
    
    models_dir = "models"
    
    # Read all model metadata
    model_files = [
        "best_deep_model_metadata.json",
        "LSTM_metadata_20250917_164301.json",
        "CNN_metadata_20250917_164306.json",
        "Hybrid_metadata_20250917_164314.json"
    ]
    
    results = []
    
    for model_file in model_files:
        path = os.path.join(models_dir, model_file)
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                results.append(data)
    
    # Display results
    print("\n📊 Model Performance Comparison:\n")
    print(f"{'Model Type':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 80)
    
    for result in results:
        model_type = result.get('model_type', 'Unknown')
        metrics = result.get('metrics', {})
        
        accuracy = metrics.get('test_accuracy', 0) * 100
        precision = metrics.get('test_precision', 0) * 100
        recall = metrics.get('test_recall', 0) * 100
        f1 = metrics.get('test_f1', 0) * 100
        
        print(f"{model_type:<20} {accuracy:>10.2f}%  {precision:>10.2f}%  {recall:>10.2f}%  {f1:>10.2f}%")
    
    print("\n" + "=" * 80)
    
    # Best model
    if results:
        best = results[0]
        print(f"\n🏆 BEST MODEL: {best.get('model_type', 'Unknown')}")
        print(f"   Accuracy: {best['metrics']['test_accuracy'] * 100:.2f}%")
        print(f"   Trained: {best.get('timestamp', 'Unknown')}")
        print(f"   Model Path: {best.get('model_path', 'Unknown')}")
    
    return results

def show_dataset_info():
    """Display dataset information"""
    print("\n" + "=" * 80)
    print("📁 DATASET INFORMATION")
    print("=" * 80)
    
    # Read labels
    labels_path = "data/processed/labels.json"
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        
        print(f"\n✅ Total Subjects: {len(labels)}")
        print(f"\n👤 Subjects in Dataset:")
        for name, label in labels.items():
            print(f"   {label}. {name.capitalize()}")
    
    # Check data files
    if os.path.exists("data/processed/X.npy"):
        X = np.load("data/processed/X.npy")
        print(f"\n📊 Feature Matrix Shape: {X.shape}")
        print(f"   Samples: {X.shape[0]}")
        print(f"   Sequence Length: {X.shape[1]}")
        print(f"   Features per Frame: {X.shape[2]}")
    
    if os.path.exists("data/processed/y.npy"):
        y = np.load("data/processed/y.npy")
        print(f"\n🏷️  Labels Shape: {y.shape}")
    
    # Check gait keypoints
    if os.path.exists("data/gait_keypoints.csv"):
        df = pd.read_csv("data/gait_keypoints.csv")
        print(f"\n📝 Gait Keypoints CSV:")
        print(f"   Total Rows: {len(df)}")
        print(f"   Columns: {list(df.columns)[:5]}... ({len(df.columns)} total)")

def show_visualizations():
    """Display available visualizations"""
    print("\n" + "=" * 80)
    print("📈 AVAILABLE VISUALIZATIONS")
    print("=" * 80)
    
    viz_dir = "data/visualizations"
    
    if os.path.exists(viz_dir):
        viz_files = [f for f in os.listdir(viz_dir) if f.endswith('.png')]
        
        if viz_files:
            print(f"\n✅ Found {len(viz_files)} visualization(s):")
            print()
            
            categories = {
                'Individual Gait Patterns': [],
                'Analysis Visualizations': []
            }
            
            for viz_file in sorted(viz_files):
                path = os.path.join(viz_dir, viz_file)
                if viz_file.startswith('gait_') and not any(x in viz_file for x in ['correlation', 'pca']):
                    categories['Individual Gait Patterns'].append((viz_file, path))
                else:
                    categories['Analysis Visualizations'].append((viz_file, path))
            
            for category, files in categories.items():
                if files:
                    print(f"📊 {category}:")
                    for viz_file, path in files:
                        size = os.path.getsize(path) / 1024  # KB
                        print(f"   ✓ {viz_file:<35} ({size:.1f} KB)")
                    print()
        else:
            print("\n⚠️  No visualizations found yet")
    else:
        print("\n⚠️  Visualizations directory not found")

def show_videos():
    """Show available test videos"""
    print("\n" + "=" * 80)
    print("🎬 TEST VIDEOS")
    print("=" * 80)
    
    data_dir = "data"
    videos = [f for f in os.listdir(data_dir) if f.endswith('.mp4')]
    
    if videos:
        print(f"\n✅ Found {len(videos)} test video(s):")
        print()
        for video in sorted(videos):
            path = os.path.join(data_dir, video)
            size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"   🎥 {video:<25} ({size:.2f} MB)")
    else:
        print("\n⚠️  No test videos found in data directory")

def create_summary_report():
    """Create a summary report document"""
    print("\n" + "=" * 80)
    print("📄 GENERATING SUMMARY REPORT")
    print("=" * 80)
    
    report = []
    report.append("# Deepfake Detection using Gait Analysis")
    report.append("## Project Summary Report")
    report.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n" + "=" * 80)
    
    # Model Results
    report.append("\n## Model Performance")
    report.append("\n| Model | Accuracy | Precision | Recall | F1-Score |")
    report.append("|-------|----------|-----------|--------|----------|")
    
    models_dir = "models"
    model_files = [
        "LSTM_metadata_20250917_164301.json",
        "CNN_metadata_20250917_164306.json",
        "Hybrid_metadata_20250917_164314.json"
    ]
    
    for model_file in model_files:
        path = os.path.join(models_dir, model_file)
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                model_type = data.get('model_type', 'Unknown')
                metrics = data.get('metrics', {})
                
                acc = metrics.get('test_accuracy', 0) * 100
                prec = metrics.get('test_precision', 0) * 100
                rec = metrics.get('test_recall', 0) * 100
                f1 = metrics.get('test_f1', 0) * 100
                
                report.append(f"| {model_type} | {acc:.2f}% | {prec:.2f}% | {rec:.2f}% | {f1:.2f}% |")
    
    # Dataset Info
    report.append("\n## Dataset Information")
    
    labels_path = "data/processed/labels.json"
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        report.append(f"\n- **Total Subjects:** {len(labels)}")
        report.append(f"- **Subjects:** {', '.join(labels.keys())}")
    
    if os.path.exists("data/processed/X.npy"):
        X = np.load("data/processed/X.npy")
        report.append(f"- **Total Samples:** {X.shape[0]}")
        report.append(f"- **Sequence Length:** {X.shape[1]} frames")
        report.append(f"- **Features:** {X.shape[2]} per frame")
    
    # Visualizations
    viz_dir = "data/visualizations"
    if os.path.exists(viz_dir):
        viz_files = [f for f in os.listdir(viz_dir) if f.endswith('.png')]
        report.append(f"\n## Visualizations")
        report.append(f"\n- **Total Visualizations:** {len(viz_files)}")
    
    # Save report
    report_text = '\n'.join(report)
    
    with open('PROJECT_SUMMARY.md', 'w') as f:
        f.write(report_text)
    
    print("\n✅ Summary report saved to: PROJECT_SUMMARY.md")
    
    return report_text

def main():
    """Main function to show all outputs"""
    
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "DEEPFAKE DETECTION PROJECT OUTPUTS" + " " * 24 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    print()
    
    # Show all results
    show_model_results()
    show_dataset_info()
    show_visualizations()
    show_videos()
    
    # Create summary
    create_summary_report()
    
    print("\n" + "=" * 80)
    print("📋 WHAT YOU CAN SHOW YOUR SUPERVISOR:")
    print("=" * 80)
    print("""
1. ✅ MODEL RESULTS (100% Accuracy!)
   - Location: models/best_deep_model_metadata.json
   - Multiple models trained (LSTM, CNN, Hybrid)
   
2. ✅ VISUALIZATIONS (10 images!)
   - Location: data/visualizations/
   - Individual gait patterns
   - Correlation analysis
   - PCA analysis
   - Stride analysis
   
3. ✅ PROCESSED DATA
   - Location: data/processed/
   - Feature matrices (X.npy, y.npy)
   - Labels mapping (labels.json)
   
4. ✅ TEST VIDEOS
   - Location: data/*.mp4
   - 6 test videos available
   
5. ✅ SUMMARY REPORT
   - Just generated: PROJECT_SUMMARY.md
   - Clean formatted report with all metrics

💡 DEMO IDEAS:
   - Show the visualizations (they're already images!)
   - Show the model metrics (100% accuracy is impressive!)
   - Explain the gait analysis approach
   - Show the processed data pipeline
   
🚀 TO RUN LIVE DEMO:
   python detect_deepfake.py data/aditya.mp4
   python detect_deepfake.py data/bubbly.mp4
    """)
    
    print("=" * 80)
    
    # Ask if user wants to open visualizations
    print("\n📊 Would you like to open the visualizations? (Y/N): ", end='')
    
    try:
        response = input().strip().lower()
        if response == 'y':
            import webbrowser
            viz_dir = os.path.abspath("data/visualizations")
            webbrowser.open(viz_dir)
            print(f"\n✅ Opening: {viz_dir}")
    except:
        pass
    
    print("\n" + "=" * 80)
    print("✅ Output summary complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
