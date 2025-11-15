"""
Real-Time Exercise Form Analysis with WebRTC
Provides instant feedback as you exercise using browser camera
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
from ultralytics import YOLO
import numpy as np
from collections import deque
import cv2

from form_analysis import analyze_video

# Class mapping
CLASS_NAMES = [
    'baseball_pitch', 'baseball_swing', 'bench_press', 'bowl',
    'clean_and_jerk', 'golf_swing', 'jumping_jacks', 'jump_rope',
    'pullup', 'pushup', 'situp', 'squat', 'strum_guitar',
    'tennis_forehand', 'tennis_serve'
]

EXERCISE_NAME_MAPPING = {name: name for name in CLASS_NAMES}


def normalize_exercise_name(penn_action_name: str) -> str:
    return EXERCISE_NAME_MAPPING.get(penn_action_name, penn_action_name)


class VideoProcessor:
    """Process video frames with YOLO model"""

    def __init__(self):
        self.model = None
        self.keypoints_buffer = deque(maxlen=90)
        self.current_exercise = None
        self.exercise_confidence = {}
        self.confidence_threshold = 0.5

    def load_model(self, model_path):
        """Load YOLO model"""
        if self.model is None:
            self.model = YOLO(model_path)

    def recv(self, frame):
        """Process each video frame"""
        img = frame.to_ndarray(format="bgr24")

        if self.model is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Run YOLO inference
        results = self.model(img, conf=self.confidence_threshold, verbose=False)

        # Process results
        current_keypoints = None
        current_exercise = None
        current_conf = 0

        for result in results:
            # Draw annotations
            img = result.plot()

            # Extract keypoints
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                kp = result.keypoints
                if hasattr(kp, 'data') and kp.data is not None and len(kp.data) > 0:
                    keypoints = kp.data[0].cpu().numpy()[:13]
                    current_keypoints = keypoints
                    self.keypoints_buffer.append(keypoints)

            # Get exercise classification
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                for box in result.boxes:
                    cls_idx = int(box.cls[0])
                    conf = float(box.conf[0])

                    if conf > current_conf:
                        current_conf = conf
                        current_exercise = CLASS_NAMES[cls_idx]

                        if current_exercise not in self.exercise_confidence:
                            self.exercise_confidence[current_exercise] = []
                        self.exercise_confidence[current_exercise].append(conf)

        # Update current exercise
        if current_exercise:
            self.current_exercise = current_exercise

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    """Main Real-Time Streamlit app with WebRTC"""

    st.set_page_config(
        page_title="Live Exercise Form Analysis",
        page_icon="🎥",
        layout="wide"
    )

    st.title("🎥 Live Exercise Form Analysis")

    st.markdown("""
    Get **real-time feedback** on your exercise form using your webcam!

    **How to use:**
    1. Click "START" below to allow camera access
    2. Position yourself in frame
    3. Start exercising - you'll get instant feedback!
    """)

    # Sidebar settings
    st.sidebar.header("Settings")
    model_path = st.sidebar.text_input(
        "Model Path",
        value="best_full.pt",
        help="Path to your trained YOLO pose model"
    )

    confidence_threshold = st.sidebar.slider(
        "Detection Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence for detections"
    )

    feedback_window = st.sidebar.slider(
        "Feedback Window (frames)",
        min_value=30,
        max_value=300,
        value=90,
        step=30,
        help="Number of frames to analyze for feedback"
    )

    # Initialize session state
    if 'ctx' not in st.session_state:
        st.session_state.ctx = None

    # Create layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # WebRTC configuration
        RTC_CONFIGURATION = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        # Create video processor
        video_processor = VideoProcessor()
        video_processor.load_model(model_path)
        video_processor.confidence_threshold = confidence_threshold
        video_processor.keypoints_buffer = deque(maxlen=feedback_window)

        # WebRTC streamer
        ctx = webrtc_streamer(
            key="exercise-form-analysis",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: video_processor,
            async_processing=True,
            media_stream_constraints={"video": True, "audio": False},
        )

        st.session_state.ctx = ctx

    with col2:
        st.subheader("📊 Live Feedback")

        # Display feedback
        if st.session_state.ctx and st.session_state.ctx.state.playing:
            # Get the video processor instance
            if hasattr(st.session_state.ctx, 'video_processor'):
                vp = st.session_state.ctx.video_processor

                if vp and vp.current_exercise:
                    exercise_display = vp.current_exercise.replace('_', ' ').title()
                    st.markdown(f"### 🏋️ {exercise_display}")

                    avg_conf = np.mean(vp.exercise_confidence.get(vp.current_exercise, [0]))
                    st.metric("Confidence", f"{avg_conf*100:.1f}%")

                    # Generate feedback if we have enough frames
                    if len(vp.keypoints_buffer) >= 30:
                        try:
                            normalized_name = normalize_exercise_name(vp.current_exercise)
                            analysis = analyze_video(
                                list(vp.keypoints_buffer),
                                normalized_name
                            )

                            # Display score
                            score = analysis.overall_score
                            if score >= 8.5:
                                score_color = "🟢"
                                rating = "EXCELLENT"
                            elif score >= 7.0:
                                score_color = "🟡"
                                rating = "GOOD"
                            elif score >= 5.5:
                                score_color = "🟠"
                                rating = "FAIR"
                            else:
                                score_color = "🔴"
                                rating = "NEEDS IMPROVEMENT"

                            st.markdown(f"""
                            ### {score_color} {score:.1f}/10
                            **{rating}**
                            """)

                            # Display rep count
                            if analysis.rep_count:
                                st.metric("🔁 Reps", analysis.rep_count)

                            # Display recent feedback
                            st.markdown("### 💬 Recent Feedback")

                            recent_feedbacks = sorted(
                                analysis.feedbacks,
                                key=lambda x: x.frame_number,
                                reverse=True
                            )[:10]

                            critical = [fb for fb in recent_feedbacks if fb.level.value == 'critical']
                            warnings = [fb for fb in recent_feedbacks if fb.level.value == 'warning']
                            good = [fb for fb in recent_feedbacks if fb.level.value == 'good']

                            if critical:
                                st.markdown("**❌ Critical Issues:**")
                                for fb in critical[:3]:
                                    st.write(f"- {fb.message}")

                            if warnings:
                                st.markdown("**⚠️ Warnings:**")
                                for fb in warnings[:3]:
                                    st.write(f"- {fb.message}")

                            if good and not critical:
                                st.markdown("**✅ Good Form:**")
                                for fb in good[:2]:
                                    st.write(f"- {fb.message}")

                        except Exception as e:
                            st.warning(f"⚠️ Analysis in progress...")
                else:
                    st.info("👋 Start exercising to see feedback!")
        else:
            st.info("👆 Click START to begin real-time analysis!")

            with st.expander("ℹ️ Tips for best results"):
                st.markdown("""
                - **Lighting**: Make sure you're in a well-lit area
                - **Position**: Stand far enough that your whole body is visible
                - **Background**: Plain background works best
                - **Movement**: Start slow to calibrate, then go at normal pace
                - **Camera**: Use a stable camera position (tripod or stand)

                **Supported Exercises:**
                - Strength: Squat, Pushup, Pullup, Bench Press, Situp, Clean & Jerk
                - Cardio: Jumping Jacks, Jump Rope
                - Sports: Tennis, Baseball, Golf, Bowling
                """)


if __name__ == "__main__":
    main()
