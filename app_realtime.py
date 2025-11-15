"""
Real-Time Exercise Form Analysis with Live Camera Feed
Provides instant feedback as you exercise
"""

import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import time
from collections import deque

from form_analysis import (
    analyze_video,
    ExerciseAnalysis
)

# Class mapping (MUST match training!)
CLASS_NAMES = [
    'baseball_pitch',      # 0
    'baseball_swing',      # 1
    'bench_press',         # 2
    'bowl',                # 3
    'clean_and_jerk',      # 4
    'golf_swing',          # 5
    'jumping_jacks',       # 6
    'jump_rope',           # 7
    'pullup',              # 8
    'pushup',              # 9
    'situp',               # 10
    'squat',               # 11
    'strum_guitar',        # 12
    'tennis_forehand',     # 13
    'tennis_serve'         # 14
]

EXERCISE_NAME_MAPPING = {
    'pullup': 'pullup',
    'pushup': 'pushup',
    'situp': 'situp',
    'squat': 'squat',
    'strum_guitar': 'strum_guitar',
    'baseball_pitch': 'baseball_pitch',
    'baseball_swing': 'baseball_swing',
    'bench_press': 'bench_press',
    'bowl': 'bowl',
    'clean_and_jerk': 'clean_and_jerk',
    'golf_swing': 'golf_swing',
    'jumping_jacks': 'jumping_jacks',
    'jump_rope': 'jump_rope',
    'tennis_forehand': 'tennis_forehand',
    'tennis_serve': 'tennis_serve',
}


def normalize_exercise_name(penn_action_name: str) -> str:
    """Convert Penn Action exercise name to form_analysis.py format"""
    return EXERCISE_NAME_MAPPING.get(penn_action_name, penn_action_name)


def main():
    """Main Real-Time Streamlit app"""

    st.set_page_config(
        page_title="Live Exercise Form Analysis",
        page_icon="🎥",
        layout="wide"
    )

    st.title("🎥 Live Exercise Form Analysis")

    st.markdown("""
    Get **real-time feedback** on your exercise form using your webcam!

    **How to use:**
    1. Click "Start Camera" below
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
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    if 'keypoints_buffer' not in st.session_state:
        st.session_state.keypoints_buffer = deque(maxlen=feedback_window)
    if 'current_exercise' not in st.session_state:
        st.session_state.current_exercise = None
    if 'exercise_confidence' not in st.session_state:
        st.session_state.exercise_confidence = {}

    # Camera control
    col1, col2 = st.columns([1, 4])

    with col1:
        if st.button("🎥 Start Camera" if not st.session_state.camera_running else "⏹️ Stop Camera"):
            st.session_state.camera_running = not st.session_state.camera_running
            if not st.session_state.camera_running:
                st.session_state.keypoints_buffer.clear()
                st.session_state.current_exercise = None
                st.session_state.exercise_confidence = {}

    # Create layout
    video_col, feedback_col = st.columns([2, 1])

    with video_col:
        video_placeholder = st.empty()

    with feedback_col:
        exercise_placeholder = st.empty()
        confidence_placeholder = st.empty()
        score_placeholder = st.empty()
        rep_placeholder = st.empty()
        feedback_placeholder = st.empty()

    if st.session_state.camera_running:
        try:
            # Load model
            with st.spinner("Loading model..."):
                model = YOLO(model_path)

            # Open camera
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("❌ Could not access camera. Please check permissions.")
                st.session_state.camera_running = False
                return

            frame_count = 0

            while st.session_state.camera_running:
                ret, frame = cap.read()

                if not ret:
                    st.error("❌ Failed to read from camera")
                    break

                # Convert to RGB for YOLO
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Run YOLO inference
                results = model(frame_rgb, conf=confidence_threshold, verbose=False)

                # Get annotated frame
                annotated_frame = frame_rgb.copy()
                current_exercise = None
                current_conf = 0
                current_keypoints = None

                for result in results:
                    # Draw bounding boxes and keypoints
                    annotated_frame = result.plot()

                    # Extract keypoints
                    if hasattr(result, 'keypoints') and result.keypoints is not None:
                        kp = result.keypoints
                        if hasattr(kp, 'data') and kp.data is not None and len(kp.data) > 0:
                            keypoints = kp.data[0].cpu().numpy()[:13]
                            current_keypoints = keypoints
                            st.session_state.keypoints_buffer.append(keypoints)

                    # Get exercise classification
                    if hasattr(result, 'boxes') and len(result.boxes) > 0:
                        for box in result.boxes:
                            cls_idx = int(box.cls[0])
                            conf = float(box.conf[0])

                            if conf > current_conf:
                                current_conf = conf
                                current_exercise = CLASS_NAMES[cls_idx]

                                # Track confidence over time
                                if current_exercise not in st.session_state.exercise_confidence:
                                    st.session_state.exercise_confidence[current_exercise] = []
                                st.session_state.exercise_confidence[current_exercise].append(conf)

                # Display video frame
                video_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

                # Update current exercise
                if current_exercise:
                    st.session_state.current_exercise = current_exercise

                # Display feedback
                if st.session_state.current_exercise:
                    exercise_display = st.session_state.current_exercise.replace('_', ' ').title()
                    exercise_placeholder.markdown(f"### 🏋️ {exercise_display}")

                    avg_conf = np.mean(st.session_state.exercise_confidence.get(st.session_state.current_exercise, [0]))
                    confidence_placeholder.metric("Confidence", f"{avg_conf*100:.1f}%")

                    # Generate feedback if we have enough frames
                    if len(st.session_state.keypoints_buffer) >= 30:
                        try:
                            normalized_name = normalize_exercise_name(st.session_state.current_exercise)
                            analysis = analyze_video(
                                list(st.session_state.keypoints_buffer),
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

                            score_placeholder.markdown(f"""
                            ### {score_color} {score:.1f}/10
                            **{rating}**
                            """)

                            # Display rep count
                            if analysis.rep_count:
                                rep_placeholder.metric("🔁 Reps", analysis.rep_count)
                            else:
                                rep_placeholder.empty()

                            # Display recent feedback
                            feedback_html = "### 💬 Live Feedback\n\n"

                            # Get most recent feedback
                            recent_feedbacks = sorted(
                                analysis.feedbacks,
                                key=lambda x: x.frame_number,
                                reverse=True
                            )[:10]

                            # Group by level
                            critical = [fb for fb in recent_feedbacks if fb.level.value == 'critical']
                            warnings = [fb for fb in recent_feedbacks if fb.level.value == 'warning']
                            good = [fb for fb in recent_feedbacks if fb.level.value == 'good']

                            if critical:
                                feedback_html += "**❌ Critical Issues:**\n"
                                for fb in critical[:3]:
                                    feedback_html += f"- {fb.message}\n"
                                feedback_html += "\n"

                            if warnings:
                                feedback_html += "**⚠️ Warnings:**\n"
                                for fb in warnings[:3]:
                                    feedback_html += f"- {fb.message}\n"
                                feedback_html += "\n"

                            if good and not critical:
                                feedback_html += "**✅ Good Form:**\n"
                                for fb in good[:2]:
                                    feedback_html += f"- {fb.message}\n"

                            feedback_placeholder.markdown(feedback_html)

                        except Exception as e:
                            feedback_placeholder.warning(f"⚠️ Analysis in progress...")

                else:
                    exercise_placeholder.info("👋 Position yourself in frame and start exercising!")

                frame_count += 1

                # Small delay to prevent overwhelming the UI
                time.sleep(0.01)

            # Release camera
            cap.release()

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            if 'cap' in locals():
                cap.release()
            st.session_state.camera_running = False

    else:
        video_col.info("👆 Click 'Start Camera' to begin real-time analysis!")

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
