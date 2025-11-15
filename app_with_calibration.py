"""
Real-Time Exercise Form Analysis with Personalized Calibration
Includes user-specific calibration for improved accuracy
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
from ultralytics import YOLO
import numpy as np
from collections import deque
import cv2
from pathlib import Path

from form_analysis import analyze_video
from calibration_system import CalibrationSession, UserCalibration

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


class CalibratedVideoProcessor:
    """Process video frames with calibration support"""

    def __init__(self, user_profile: UserCalibration = None):
        self.model = None
        self.keypoints_buffer = deque(maxlen=90)
        self.current_exercise = None
        self.exercise_confidence = {}
        self.confidence_threshold = 0.5
        self.user_profile = user_profile
        self.calibration_session = None
        self.is_calibrating = False

    def load_model(self, model_path):
        """Load YOLO model"""
        if self.model is None:
            self.model = YOLO(model_path)

    def start_calibration(self, user_id: str):
        """Start calibration mode"""
        self.calibration_session = CalibrationSession(user_id)
        self.is_calibrating = True

    def recv(self, frame):
        """Process each video frame"""
        img = frame.to_ndarray(format="bgr24")

        if self.model is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Run YOLO inference
        results = self.model(img, conf=self.confidence_threshold, verbose=False)

        # Process results
        annotated_frame = img.copy()
        current_keypoints = None
        current_exercise = None
        current_conf = 0

        for result in results:
            # Draw annotations
            annotated_frame = result.plot()

            # Extract keypoints
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                kp = result.keypoints
                if hasattr(kp, 'data') and kp.data is not None and len(kp.data) > 0:
                    keypoints = kp.data[0].cpu().numpy()[:13]
                    current_keypoints = keypoints
                    self.keypoints_buffer.append(keypoints)

                    # If calibrating, collect keypoints
                    if self.is_calibrating and self.calibration_session:
                        self.calibration_session.add_frame(keypoints)

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

            # Start calibration for this exercise if needed
            if self.is_calibrating and self.calibration_session.current_exercise is None:
                self.calibration_session.start_calibration(current_exercise)

        # Overlay calibration progress if calibrating
        if self.is_calibrating and self.calibration_session:
            current, target = self.calibration_session.get_progress()
            if target > 0:
                progress = min(100, int((current / target) * 100))
                cv2.putText(annotated_frame, f"Calibrating: {progress}%",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


def main():
    """Main app with calibration support"""

    st.set_page_config(
        page_title="Personalized Exercise Analysis",
        page_icon="🎯",
        layout="wide"
    )

    st.title("🎯 Personalized Exercise Form Analysis")

    st.markdown("""
    **NEW!** Calibrate the system to your body for personalized feedback!

    **Steps:**
    1. Enter your name
    2. Click "Start Calibration" and perform the exercises shown
    3. Start your regular workout with personalized feedback!
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
        step=0.05
    )

    feedback_window = st.sidebar.slider(
        "Feedback Window (frames)",
        min_value=30,
        max_value=300,
        value=90,
        step=30
    )

    # User profile management
    st.sidebar.header("👤 User Profile")
    user_id = st.sidebar.text_input("Your Name", value="User1")

    # Check for existing profile
    profile_path = Path.home() / "Desktop" / "AI Club" / "user_profiles" / f"{user_id}.json"
    user_profile = None

    if profile_path.exists():
        user_profile = UserCalibration(user_id)
        user_profile.load_profile(profile_path)
        st.sidebar.success(f"✅ Profile loaded!")
        calibrated_exercises = user_profile.calibration_data.get('exercises_calibrated', [])
        if calibrated_exercises:
            st.sidebar.write(f"Calibrated: {', '.join(calibrated_exercises)}")
    else:
        st.sidebar.info("No profile found. Start calibration!")

    # Calibration controls
    if st.sidebar.button("🎯 Start Calibration"):
        st.session_state.calibration_mode = True
        st.session_state.calibrating = True

    if st.sidebar.button("✅ Finish Calibration"):
        st.session_state.calibrating = False

    # Initialize session state
    if 'ctx' not in st.session_state:
        st.session_state.ctx = None
    if 'calibration_mode' not in st.session_state:
        st.session_state.calibration_mode = False
    if 'calibrating' not in st.session_state:
        st.session_state.calibrating = False

    # Create layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # WebRTC configuration
        RTC_CONFIGURATION = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        # Create video processor
        video_processor = CalibratedVideoProcessor(user_profile)
        video_processor.load_model(model_path)
        video_processor.confidence_threshold = confidence_threshold
        video_processor.keypoints_buffer = deque(maxlen=feedback_window)

        # Start calibration if requested
        if st.session_state.get('calibrating', False):
            video_processor.start_calibration(user_id)

        # WebRTC streamer
        ctx = webrtc_streamer(
            key="calibrated-exercise-analysis",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: video_processor,
            async_processing=True,
            media_stream_constraints={"video": True, "audio": False},
        )

        st.session_state.ctx = ctx

    with col2:
        # Show calibration instructions or feedback
        if st.session_state.get('calibrating', False):
            st.subheader("🎯 Calibration Mode")
            st.markdown("""
            **Instructions:**
            1. Perform 3-5 slow reps of the detected exercise
            2. Focus on full range of motion
            3. Click "Finish Calibration" when done

            The system is learning:
            - Your body proportions
            - Your flexibility range
            - Your baseline form
            """)

            if st.session_state.ctx and hasattr(st.session_state.ctx, 'video_processor'):
                vp = st.session_state.ctx.video_processor
                if vp and vp.calibration_session:
                    current, target = vp.calibration_session.get_progress()
                    if target > 0:
                        progress = min(1.0, current / target)
                        st.progress(progress)
                        st.write(f"Progress: {int(progress * 100)}%")

        else:
            st.subheader("📊 Live Feedback")

            # Display feedback
            if st.session_state.ctx and st.session_state.ctx.state.playing:
                if hasattr(st.session_state.ctx, 'video_processor'):
                    vp = st.session_state.ctx.video_processor

                    if vp and vp.current_exercise:
                        exercise_display = vp.current_exercise.replace('_', ' ').title()

                        # Show if calibrated for this exercise
                        is_calibrated = (user_profile and
                                       user_profile.is_calibrated(vp.current_exercise))

                        if is_calibrated:
                            st.success(f"🎯 Using personalized thresholds for {exercise_display}")
                        else:
                            st.info(f"Using standard thresholds for {exercise_display}")

                        st.markdown(f"### 🏋️ {exercise_display}")

                        avg_conf = np.mean(vp.exercise_confidence.get(vp.current_exercise, [0]))
                        st.metric("Confidence", f"{avg_conf*100:.1f}%")

                        # Generate feedback
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
                st.info("👆 Click START to begin!")

                with st.expander("ℹ️ Why Calibration?"):
                    st.markdown("""
                    **Personalized calibration improves accuracy by:**

                    - 📏 Learning your body proportions
                    - 🤸 Understanding your flexibility limits
                    - 🎯 Adapting thresholds to YOUR form
                    - 📹 Accounting for camera angle

                    **Result:** More accurate, personalized feedback!
                    """)


if __name__ == "__main__":
    main()
