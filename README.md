# 🏋️ Exercise Form Checker - AI-Powered Fitness Coach

An AI-powered web application that analyzes your exercise form in real-time using YOLOv8 pose estimation and provides detailed biomechanical feedback.

## 🎯 Features

- **15 Exercise Types**: Pullup, Pushup, Squat, Situp, Bench Press, Jumping Jacks, Jump Rope, Tennis Serve/Forehand, Baseball Swing/Pitch, Golf Swing, Bowling, Clean & Jerk, Guitar Strumming
- **Real-time Pose Detection**: YOLOv8 with 13 keypoint tracking
- **Biomechanical Analysis**: Joint angles, body alignment, range of motion
- **Form Scoring**: 0-10 scale with detailed breakdown
- **Visual Feedback**: Annotated videos with pose overlays
- **Export Reports**: JSON and CSV analysis reports

## 🚀 Model Performance

- **Accuracy**: 89.49% mAP@50
- **Training Data**: 163,841 frames from Penn Action dataset
- **Architecture**: YOLOv8n-pose (fine-tuned)

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
- **Deployment**: Streamlit Community Cloud

## 📝 License

MIT License - Feel free to use and modify!

## 🙏 Acknowledgments

- Penn Action Dataset for training data
- Ultralytics for YOLOv8 framework
