"""
Personalized Calibration System for Exercise Form Analysis
Creates user-specific profiles to improve accuracy
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import cv2


class UserCalibration:
    """Stores and manages user-specific calibration data"""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.calibration_data = {
            'user_id': user_id,
            'timestamp': None,
            'body_proportions': {},
            'flexibility_ranges': {},
            'baseline_keypoints': {},
            'exercises_calibrated': [],
        }

    def calibrate_exercise(self, exercise_name: str, keypoints_sequence: List[np.ndarray]):
        """
        Calibrate for a specific exercise using sample reps

        Args:
            exercise_name: Name of exercise (e.g., 'pushup', 'squat')
            keypoints_sequence: List of keypoint arrays from calibration reps
        """
        print(f"📊 Calibrating {exercise_name}...")

        # Calculate body proportions from keypoints
        body_props = self._calculate_body_proportions(keypoints_sequence)

        # Calculate range of motion
        rom = self._calculate_range_of_motion(keypoints_sequence, exercise_name)

        # Store baseline keypoints
        baseline = self._calculate_baseline_keypoints(keypoints_sequence)

        # Update calibration data
        self.calibration_data['body_proportions'][exercise_name] = body_props
        self.calibration_data['flexibility_ranges'][exercise_name] = rom
        self.calibration_data['baseline_keypoints'][exercise_name] = baseline
        self.calibration_data['exercises_calibrated'].append(exercise_name)
        self.calibration_data['timestamp'] = datetime.now().isoformat()

        print(f"✅ {exercise_name} calibrated!")
        print(f"   - Body proportions: {body_props}")
        print(f"   - Range of motion: {rom}")

    def _calculate_body_proportions(self, keypoints_sequence: List[np.ndarray]) -> Dict:
        """Calculate user's body proportions from keypoints"""

        # Average keypoints across all frames
        avg_keypoints = np.mean(keypoints_sequence, axis=0)

        # Penn Action keypoint indices
        # 0: Head, 1: L Shoulder, 2: R Shoulder, 3: L Elbow, 4: R Elbow
        # 5: L Wrist, 6: R Wrist, 7: L Hip, 8: R Hip, 9: L Knee
        # 10: R Knee, 11: L Ankle, 12: R Ankle

        props = {}

        # Shoulder width
        if avg_keypoints[1, 2] > 0.3 and avg_keypoints[2, 2] > 0.3:
            props['shoulder_width'] = float(np.linalg.norm(
                avg_keypoints[1, :2] - avg_keypoints[2, :2]
            ))

        # Hip width
        if avg_keypoints[7, 2] > 0.3 and avg_keypoints[8, 2] > 0.3:
            props['hip_width'] = float(np.linalg.norm(
                avg_keypoints[7, :2] - avg_keypoints[8, :2]
            ))

        # Torso length (shoulder to hip)
        if avg_keypoints[1, 2] > 0.3 and avg_keypoints[7, 2] > 0.3:
            props['torso_length'] = float(np.linalg.norm(
                avg_keypoints[1, :2] - avg_keypoints[7, :2]
            ))

        # Upper leg length (hip to knee)
        if avg_keypoints[7, 2] > 0.3 and avg_keypoints[9, 2] > 0.3:
            props['upper_leg_length'] = float(np.linalg.norm(
                avg_keypoints[7, :2] - avg_keypoints[9, :2]
            ))

        # Lower leg length (knee to ankle)
        if avg_keypoints[9, 2] > 0.3 and avg_keypoints[11, 2] > 0.3:
            props['lower_leg_length'] = float(np.linalg.norm(
                avg_keypoints[9, :2] - avg_keypoints[11, :2]
            ))

        # Upper arm length (shoulder to elbow)
        if avg_keypoints[1, 2] > 0.3 and avg_keypoints[3, 2] > 0.3:
            props['upper_arm_length'] = float(np.linalg.norm(
                avg_keypoints[1, :2] - avg_keypoints[3, :2]
            ))

        # Lower arm length (elbow to wrist)
        if avg_keypoints[3, 2] > 0.3 and avg_keypoints[5, 2] > 0.3:
            props['lower_arm_length'] = float(np.linalg.norm(
                avg_keypoints[3, :2] - avg_keypoints[5, :2]
            ))

        # Height estimation (head to ankle)
        if avg_keypoints[0, 2] > 0.3 and avg_keypoints[11, 2] > 0.3:
            props['height_estimate'] = float(np.linalg.norm(
                avg_keypoints[0, :2] - avg_keypoints[11, :2]
            ))

        return props

    def _calculate_range_of_motion(self, keypoints_sequence: List[np.ndarray],
                                   exercise: str) -> Dict:
        """Calculate user's range of motion for specific exercise"""

        rom = {}

        if exercise in ['squat', 'situp']:
            # Track knee angle range
            knee_angles = []
            for kp in keypoints_sequence:
                if all(kp[i, 2] > 0.3 for i in [7, 9, 11]):  # hip, knee, ankle visible
                    angle = self._calculate_angle(kp[7, :2], kp[9, :2], kp[11, :2])
                    knee_angles.append(angle)

            if knee_angles:
                rom['knee_angle_min'] = float(np.min(knee_angles))
                rom['knee_angle_max'] = float(np.max(knee_angles))
                rom['knee_angle_range'] = float(np.max(knee_angles) - np.min(knee_angles))

        if exercise in ['pushup', 'pullup', 'bench_press']:
            # Track elbow angle range
            elbow_angles = []
            for kp in keypoints_sequence:
                if all(kp[i, 2] > 0.3 for i in [1, 3, 5]):  # shoulder, elbow, wrist visible
                    angle = self._calculate_angle(kp[1, :2], kp[3, :2], kp[5, :2])
                    elbow_angles.append(angle)

            if elbow_angles:
                rom['elbow_angle_min'] = float(np.min(elbow_angles))
                rom['elbow_angle_max'] = float(np.max(elbow_angles))
                rom['elbow_angle_range'] = float(np.max(elbow_angles) - np.min(elbow_angles))

        if exercise in ['squat', 'pushup', 'jumping_jacks']:
            # Track vertical range (for depth/height)
            hip_y_positions = []
            for kp in keypoints_sequence:
                if kp[7, 2] > 0.3:  # left hip visible
                    hip_y_positions.append(kp[7, 1])

            if hip_y_positions:
                rom['vertical_range'] = float(np.max(hip_y_positions) - np.min(hip_y_positions))

        return rom

    def _calculate_baseline_keypoints(self, keypoints_sequence: List[np.ndarray]) -> Dict:
        """Calculate baseline/reference pose for the user"""

        # Get standing/neutral pose (typically first few frames)
        neutral_frames = keypoints_sequence[:5]

        baseline = {
            'neutral_pose': np.mean(neutral_frames, axis=0).tolist(),
            'std_deviation': np.std(neutral_frames, axis=0).tolist(),
        }

        return baseline

    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle at p2 formed by p1-p2-p3"""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def get_adaptive_thresholds(self, exercise: str) -> Dict:
        """
        Get personalized thresholds based on calibration

        Returns:
            Dictionary of adjusted thresholds for form analysis
        """
        if exercise not in self.calibration_data['exercises_calibrated']:
            # Return default thresholds if not calibrated
            return self._get_default_thresholds(exercise)

        thresholds = {}
        rom = self.calibration_data['flexibility_ranges'].get(exercise, {})

        if exercise == 'squat':
            # Adjust squat depth threshold based on user's ROM
            if 'knee_angle_min' in rom:
                # User's minimum knee angle becomes their target
                thresholds['excellent_knee_angle'] = rom['knee_angle_min'] + 10
                thresholds['good_knee_angle'] = rom['knee_angle_min'] + 20
                thresholds['acceptable_knee_angle'] = rom['knee_angle_min'] + 35

        elif exercise == 'pushup':
            # Adjust pushup depth based on user's ROM
            if 'elbow_angle_min' in rom:
                thresholds['excellent_elbow_angle'] = rom['elbow_angle_min'] + 5
                thresholds['good_elbow_angle'] = rom['elbow_angle_min'] + 15
                thresholds['acceptable_elbow_angle'] = rom['elbow_angle_min'] + 30

        elif exercise == 'pullup':
            # Adjust pullup height based on user's ROM
            if 'elbow_angle_min' in rom:
                thresholds['excellent_pull_angle'] = rom['elbow_angle_min'] + 10
                thresholds['good_pull_angle'] = rom['elbow_angle_min'] + 20

        return thresholds

    def _get_default_thresholds(self, exercise: str) -> Dict:
        """Return default thresholds when no calibration exists"""
        defaults = {
            'squat': {
                'excellent_knee_angle': 70,
                'good_knee_angle': 90,
                'acceptable_knee_angle': 110,
            },
            'pushup': {
                'excellent_elbow_angle': 70,
                'good_elbow_angle': 90,
                'acceptable_elbow_angle': 110,
            },
            'pullup': {
                'excellent_pull_angle': 40,
                'good_pull_angle': 60,
            },
        }
        return defaults.get(exercise, {})

    def save_profile(self, filepath: Path):
        """Save calibration profile to file"""
        with open(filepath, 'w') as f:
            json.dump(self.calibration_data, f, indent=2)
        print(f"💾 Profile saved to {filepath}")

    def load_profile(self, filepath: Path):
        """Load calibration profile from file"""
        if filepath.exists():
            with open(filepath, 'r') as f:
                self.calibration_data = json.load(f)
            self.user_id = self.calibration_data['user_id']
            print(f"📂 Profile loaded: {self.user_id}")
            print(f"   Calibrated exercises: {', '.join(self.calibration_data['exercises_calibrated'])}")
            return True
        return False

    def is_calibrated(self, exercise: str) -> bool:
        """Check if user is calibrated for specific exercise"""
        return exercise in self.calibration_data['exercises_calibrated']


# Calibration workflow helper
class CalibrationSession:
    """Manages a calibration session"""

    CALIBRATION_EXERCISES = {
        'squat': {
            'reps': 3,
            'instructions': '3 slow squats to calibrate your depth and form',
        },
        'pushup': {
            'reps': 3,
            'instructions': '3 slow pushups to calibrate your range',
        },
        'jumping_jacks': {
            'reps': 5,
            'instructions': '5 jumping jacks to calibrate',
        },
    }

    def __init__(self, user_id: str = "default"):
        self.calibration = UserCalibration(user_id)
        self.current_exercise = None
        self.collected_keypoints = []

    def start_calibration(self, exercise: str) -> str:
        """Start calibration for an exercise"""
        if exercise not in self.CALIBRATION_EXERCISES:
            return f"No calibration needed for {exercise}"

        self.current_exercise = exercise
        self.collected_keypoints = []

        config = self.CALIBRATION_EXERCISES[exercise]
        return f"Please perform {config['instructions']}"

    def add_frame(self, keypoints: np.ndarray):
        """Add a frame's keypoints to calibration"""
        if self.current_exercise:
            self.collected_keypoints.append(keypoints)

    def finish_calibration(self):
        """Complete calibration for current exercise"""
        if self.current_exercise and len(self.collected_keypoints) > 10:
            self.calibration.calibrate_exercise(
                self.current_exercise,
                self.collected_keypoints
            )

            # Save profile
            profile_path = Path.home() / "Desktop" / "AI Club" / "user_profiles" / f"{self.calibration.user_id}.json"
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            self.calibration.save_profile(profile_path)

            self.current_exercise = None
            self.collected_keypoints = []
            return True
        return False

    def get_progress(self) -> Tuple[int, int]:
        """Get calibration progress (current frames, target frames)"""
        if not self.current_exercise:
            return (0, 0)

        config = self.CALIBRATION_EXERCISES[self.current_exercise]
        target_frames = config['reps'] * 30  # Assuming ~30 frames per rep
        return (len(self.collected_keypoints), target_frames)
