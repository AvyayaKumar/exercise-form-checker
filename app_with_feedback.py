"""
Enhanced Streamlit App with Form Feedback
Integrates YOLO pose detection with comprehensive form analysis
"""

import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import numpy as np
from pathlib import Path
import json

from form_analysis import (
    analyze_video,
    export_analysis_json,
    export_analysis_csv,
    ExerciseAnalysis
)

# Class mapping (MUST match FULL_DATASET_TRAINING.ipynb exactly!)
# Penn Action dataset class names in order
CLASS_NAMES = [
    'baseball_pitch',      # 0
    'baseball_swing',      # 1
    'bench_press',         # 2
    'bowl',                # 3
    'clean_and_jerk',      # 4
    'golf_swing',          # 5
    'jumping_jacks',       # 6
    'jump_rope',           # 7
    'pullup',              # 8  - Penn Action uses 'pullup' not 'pull_ups'
    'pushup',              # 9  - Penn Action uses 'pushup' not 'push_ups'
    'situp',               # 10 - Penn Action uses 'situp' not 'sit_ups'
    'squat',               # 11 - Penn Action uses 'squat' not 'squats'
    'strum_guitar',        # 12 - Penn Action uses 'strum_guitar' not 'strumming_guitar'
    'tennis_forehand',     # 13
    'tennis_serve'         # 14
]

# Normalize Penn Action names to form_analysis.py expected names
EXERCISE_NAME_MAPPING = {
    'pullup': 'pullup',              # Already matches
    'pushup': 'pushup',              # Already matches
    'situp': 'situp',                # Already matches
    'squat': 'squat',                # Already matches
    'strum_guitar': 'strum_guitar',  # Already matches
    # Others already match
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


def extract_keypoints_from_results(results) -> list:
    """
    Extract keypoints from YOLO results for all frames

    Returns:
        List of keypoint arrays, one per frame, shape (13, 3)
    """
    all_keypoints = []

    for result in results:
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            kp = result.keypoints
            if hasattr(kp, 'data') and kp.data is not None:
                # Get keypoints for first person detected
                if len(kp.data) > 0:
                    keypoints = kp.data[0].cpu().numpy()  # Shape: (13, 3) or (17, 3)
                    # Penn Action uses 13 keypoints, YOLO might use 17
                    # Take first 13 if needed
                    keypoints = keypoints[:13]
                    all_keypoints.append(keypoints)
                else:
                    # No person detected, add dummy
                    all_keypoints.append(np.zeros((13, 3)))
            else:
                all_keypoints.append(np.zeros((13, 3)))
        else:
            all_keypoints.append(np.zeros((13, 3)))

    return all_keypoints


def process_video_with_feedback(video_path, model_path, output_path, progress_callback=None):
    """
    Process video with YOLO and generate form feedback

    Returns:
        (output_video_path, best_class_name, confidence, analysis)
    """
    try:
        # Load model
        model = YOLO(model_path)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Prepare output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Storage for analysis
        all_keypoints = []
        best_confidence = 0
        best_class_idx = None
        frame_count = 0

        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)

            for result in results:
                # Get annotated frame
                annotated_frame = result.plot()
                out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

                # Extract keypoints
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    kp = result.keypoints
                    if hasattr(kp, 'data') and kp.data is not None and len(kp.data) > 0:
                        keypoints = kp.data[0].cpu().numpy()[:13]  # First person, first 13 keypoints
                        all_keypoints.append(keypoints)
                    else:
                        all_keypoints.append(np.zeros((13, 3)))
                else:
                    all_keypoints.append(np.zeros((13, 3)))

                # Track best detection
                for box in result.boxes:
                    cls_idx = int(box.cls[0])
                    conf = float(box.conf[0])

                    if conf > best_confidence:
                        best_confidence = conf
                        best_class_idx = cls_idx

            frame_count += 1

            # Progress callback
            if progress_callback:
                progress_callback(frame_count, total_frames)

        cap.release()
        out.release()

        # Determine exercise
        best_class_name = CLASS_NAMES[best_class_idx] if best_class_idx is not None else "unknown"

        # Run form analysis with normalized exercise name
        if best_class_name != "unknown" and len(all_keypoints) > 0:
            normalized_name = normalize_exercise_name(best_class_name)
            analysis = analyze_video(all_keypoints, normalized_name)
        else:
            analysis = None

        return output_path, best_class_name, best_confidence, analysis

    except Exception as e:
        raise Exception(f"Error processing video: {str(e)}")


def display_analysis_results(analysis: ExerciseAnalysis):
    """Display the form analysis results in Streamlit"""

    st.markdown("---")
    st.header("📊 Form Analysis Results")

    # Overall score with color coding
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

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Overall Score", f"{score:.1f}/10")
        st.write(f"{score_color} **{rating}**")

    with col2:
        st.metric("Total Frames Analyzed", len(analysis.frame_scores))

    with col3:
        if analysis.rep_count:
            st.metric("Estimated Reps", analysis.rep_count)
        else:
            st.metric("Estimated Reps", "N/A")

    # Summary
    st.subheader("Summary")
    st.text(analysis.summary)

    # Feedback breakdown by category
    st.subheader("Detailed Feedback")

    feedback_by_type = {}
    for fb in analysis.feedbacks:
        if fb.feedback_type not in feedback_by_type:
            feedback_by_type[fb.feedback_type] = []
        feedback_by_type[fb.feedback_type].append(fb)

    for feedback_type, feedbacks in feedback_by_type.items():
        with st.expander(f"📌 {feedback_type.upper()} Feedback ({len(feedbacks)} items)"):
            # Group by level
            by_level = {}
            for fb in feedbacks:
                level_name = fb.level.value
                if level_name not in by_level:
                    by_level[level_name] = []
                by_level[level_name].append(fb)

            for level in ["critical", "warning", "good", "excellent"]:
                if level in by_level:
                    st.markdown(f"**{level.upper()}:**")
                    unique_messages = list(set([fb.message for fb in by_level[level]]))
                    for msg in unique_messages[:5]:  # Show top 5
                        icon = "❌" if level == "critical" else "⚠️" if level == "warning" else "✅" if level == "good" else "🌟"
                        st.write(f"{icon} {msg}")

    # Frame-by-frame score plot
    st.subheader("Frame-by-Frame Performance")

    # Create simple line chart data
    import pandas as pd
    df = pd.DataFrame({
        'Frame': list(range(len(analysis.frame_scores))),
        'Score': analysis.frame_scores
    })

    st.line_chart(df.set_index('Frame'))

    # Export options
    st.subheader("Export Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # JSON export
        json_data = {
            'exercise': analysis.exercise,
            'overall_score': float(analysis.overall_score),
            'rep_count': analysis.rep_count,
            'summary': analysis.summary,
            'feedbacks': [
                {
                    'frame': fb.frame_number,
                    'type': fb.feedback_type,
                    'message': fb.message,
                    'level': fb.level.value,
                    'details': fb.details
                }
                for fb in analysis.feedbacks
            ],
            'frame_scores': [float(s) for s in analysis.frame_scores]
        }

        st.download_button(
            label="📥 Download Analysis (JSON)",
            data=json.dumps(json_data, indent=2),
            file_name=f"{analysis.exercise}_analysis.json",
            mime="application/json"
        )

    with col2:
        # CSV export
        csv_lines = ["Frame,Score,Feedback Count,Issues\n"]

        frame_feedbacks = {}
        for fb in analysis.feedbacks:
            if fb.frame_number not in frame_feedbacks:
                frame_feedbacks[fb.frame_number] = []
            frame_feedbacks[fb.frame_number].append(fb)

        for frame_num, score in enumerate(analysis.frame_scores):
            fbs = frame_feedbacks.get(frame_num, [])
            issues = '; '.join([f"{fb.level.value}: {fb.message}" for fb in fbs])
            csv_lines.append(f"{frame_num},{score:.2f},{len(fbs)},\"{issues}\"\n")

        csv_data = "".join(csv_lines)

        st.download_button(
            label="📥 Download Analysis (CSV)",
            data=csv_data,
            file_name=f"{analysis.exercise}_analysis.csv",
            mime="text/csv"
        )


def main():
    """Main Streamlit app"""

    st.set_page_config(
        page_title="Exercise Form Analysis",
        page_icon="🏋️",
        layout="wide"
    )

    st.title("🏋️ Exercise Form Analysis with AI")

    st.markdown("""
    Upload a video of your exercise and get detailed form feedback!

    **Supported Exercises:**
    - Strength: Squat, Pushup, Pullup, Bench Press, Situp, Clean & Jerk
    - Cardio: Jumping Jacks, Jump Rope
    - Sports: Tennis Serve/Forehand, Baseball Swing/Pitch, Golf Swing, Bowling
    - Other: Guitar Strumming
    """)

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your exercise video",
        type=["mp4", "avi", "mov"],
        help="Upload a video showing your exercise form"
    )

    # Model path selection
    st.sidebar.header("Settings")
    model_path = st.sidebar.text_input(
        "Model Path",
        value="best_full.pt",
        help="Path to your trained YOLO pose model (89.49% mAP@50)"
    )

    if uploaded_file is not None:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

        try:
            st.info("🔄 Processing video... This may take a few minutes.")

            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(current, total):
                progress = current / total
                progress_bar.progress(min(progress, 1.0))
                status_text.text(f"Processing: {current}/{total} frames")

            # Process video
            output_video, best_class, confidence, analysis = process_video_with_feedback(
                video_path, model_path, output_path, progress_callback=update_progress
            )

            progress_bar.empty()
            status_text.empty()

            st.success("✅ Video processing complete!")

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Detection Results")
                st.write(f"**Detected Exercise:** {best_class.replace('_', ' ').title()}")
                st.write(f"**Confidence:** {confidence*100:.2f}%")

            with col2:
                st.subheader("Download Annotated Video")
                if os.path.exists(output_path):
                    with open(output_path, 'rb') as video_file:
                        video_bytes = video_file.read()

                    st.download_button(
                        label="📥 Download Annotated Video",
                        data=video_bytes,
                        file_name=f"{best_class}_annotated.mp4",
                        mime="video/mp4"
                    )

            # Display form analysis
            if analysis:
                display_analysis_results(analysis)
            else:
                st.warning("⚠️ Could not generate form analysis. Make sure the video contains clear pose detections.")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

        finally:
            # Cleanup
            if os.path.exists(video_path):
                os.unlink(video_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    else:
        # Show example
        st.info("👆 Upload a video to get started!")

        with st.expander("ℹ️ How it works"):
            st.markdown("""
            1. **Upload Video**: Choose a video of yourself performing an exercise
            2. **AI Detection**: YOLOv8 detects your pose and identifies the exercise
            3. **Form Analysis**: Advanced algorithms analyze your form based on:
               - Joint angles (e.g., knee angle in squats)
               - Body alignment (e.g., back straightness)
               - Position (e.g., knees over toes)
               - Range of motion
            4. **Feedback Report**: Get detailed feedback with:
               - Overall score (0-10)
               - Frame-by-frame analysis
               - Specific improvement areas
               - Exportable reports
            """)


if __name__ == "__main__":
    main()
