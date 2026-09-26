# 🏋️ Exercise Form Checker - AI-Powered Fitness Coach

An AI-powered web application that analyzes your exercise form in real-time using YOLOv8 pose estimation and provides detailed biomechanical feedback.

## 🎯 Features

- **15 Movement Types (Penn Action classes)**: Pullup, Pushup, Squat, Situp, Bench Press, Jumping Jacks, Jump Rope, Tennis Serve/Forehand, Baseball Swing/Pitch, Golf Swing, Bowling, Clean & Jerk, Guitar Strumming
- **Real-time Pose Detection**: YOLOv8 with 13 keypoint tracking
- **Biomechanical Analysis**: Joint angles, body alignment, range of motion
- **Form Scoring**: 0-10 scale with detailed breakdown
- **Visual Feedback**: Annotated videos with pose overlays
- **Export Reports**: JSON and CSV analysis reports

## 🚀 Model

- **Architecture**: YOLOv8n-pose, fine-tuned (`best_full.pt`, Ultralytics 8.3.226, trained Nov 2025)
- **Data**: the Penn Action dataset (163,841 frames across 2,326 videos, 15 classes)
- **Evaluation**: the checkpoint's own validation metrics are 95.24% pose mAP@50 and 78.37% pose mAP@50-95. These aren't quoted as held-out results: the train/validation split isn't in this repo, so I can't confirm validation frames came from videos the model never saw during training. Adjacent frames from one video are nearly identical, so a frame-level split would overstate accuracy.

## 📊 How It Works

1. **Upload Video**: Upload your exercise video (MP4, AVI, or MOV)
2. **AI Detection**: YOLOv8 detects your pose and identifies the exercise
3. **Form Analysis**: Analyzes joint angles, alignment, and technique
4. **Get Feedback**: Receive detailed feedback with improvement suggestions

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI Model**: YOLOv8 (Ultralytics)
- **Analysis Engine**: Custom biomechanical analysis (1,284 lines)
- **Video Processing**: OpenCV
- **Availability**: Runs locally; no hosted demo yet

## 📝 License

MIT License - Feel free to use and modify!

## 🙏 Acknowledgments

- Penn Action Dataset for training data
- Ultralytics for YOLOv8 framework
