"""
Form Analysis Module for Exercise Feedback
Uses 13 keypoints from Penn Action dataset to provide comprehensive form feedback

Keypoints (indices 0-12):
0: Head, 1: Left Shoulder, 2: Right Shoulder, 3: Left Elbow, 4: Right Elbow
5: Left Wrist, 6: Right Wrist, 7: Left Hip, 8: Right Hip, 9: Left Knee
10: Right Knee, 11: Left Ankle, 12: Right Ankle
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class FeedbackLevel(Enum):
    """Severity level of form feedback"""
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class FormFeedback:
    """Single piece of form feedback"""
    exercise: str
    feedback_type: str  # "angle", "alignment", "position", "score"
    message: str
    level: FeedbackLevel
    frame_number: int
    details: Dict  # Additional numerical details

@dataclass
class ExerciseAnalysis:
    """Complete analysis for one video"""
    exercise: str
    overall_score: float  # 0-10
    frame_scores: List[float]
    feedbacks: List[FormFeedback]
    summary: str
    rep_count: Optional[int] = None

# ============================================================================
# GEOMETRY UTILITIES
# ============================================================================

def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate angle at p2 formed by points p1-p2-p3

    Args:
        p1, p2, p3: Points as [x, y] arrays

    Returns:
        Angle in degrees (0-180)
    """
    v1 = p1 - p2
    v2 = p3 - p2

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    return np.degrees(angle)

def calculate_line_angle(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculate angle of line from horizontal

    Returns:
        Angle in degrees (-180 to 180)
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.degrees(np.arctan2(dy, dx))

def point_to_line_distance(point: np.ndarray, line_p1: np.ndarray, line_p2: np.ndarray) -> float:
    """
    Calculate perpendicular distance from point to line

    Returns:
        Distance in normalized coordinates
    """
    num = abs((line_p2[1] - line_p1[1]) * point[0] -
              (line_p2[0] - line_p1[0]) * point[1] +
              line_p2[0] * line_p1[1] - line_p2[1] * line_p1[0])
    den = np.sqrt((line_p2[1] - line_p1[1])**2 + (line_p2[0] - line_p1[0])**2)
    return num / (den + 1e-8)

def calculate_body_alignment(head: np.ndarray, hip_mid: np.ndarray, ankle_mid: np.ndarray) -> float:
    """
    Calculate forward/backward lean of the body

    Returns:
        Angle in degrees from vertical (0 = perfectly vertical)
    """
    spine_vector = head - hip_mid
    vertical = np.array([0, -1])  # Negative y is up in image coordinates

    cos_angle = np.dot(spine_vector, vertical) / (np.linalg.norm(spine_vector) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    deviation = np.degrees(np.arccos(cos_angle))

    return deviation

def get_midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Calculate midpoint between two points"""
    return (p1 + p2) / 2

# ============================================================================
# KEYPOINT EXTRACTION
# ============================================================================

def extract_keypoints(keypoints: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract named keypoints from YOLO pose output

    Args:
        keypoints: Array of shape (13, 3) with [x, y, visibility]

    Returns:
        Dictionary mapping keypoint names to [x, y] coordinates
    """
    names = [
        'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
        'right_knee', 'left_ankle', 'right_ankle'
    ]

    kp_dict = {}
    for i, name in enumerate(names):
        if keypoints[i, 2] > 0.3:  # Visibility threshold
            kp_dict[name] = keypoints[i, :2]
        else:
            kp_dict[name] = None

    # Calculate helpful midpoints
    if kp_dict['left_shoulder'] is not None and kp_dict['right_shoulder'] is not None:
        kp_dict['shoulder_mid'] = get_midpoint(kp_dict['left_shoulder'], kp_dict['right_shoulder'])

    if kp_dict['left_hip'] is not None and kp_dict['right_hip'] is not None:
        kp_dict['hip_mid'] = get_midpoint(kp_dict['left_hip'], kp_dict['right_hip'])

    if kp_dict['left_ankle'] is not None and kp_dict['right_ankle'] is not None:
        kp_dict['ankle_mid'] = get_midpoint(kp_dict['left_ankle'], kp_dict['right_ankle'])

    return kp_dict

# ============================================================================
# EXERCISE-SPECIFIC FORM CHECKERS
# ============================================================================

def analyze_squat(kp: Dict[str, np.ndarray], frame_num: int) -> List[FormFeedback]:
    """Analyze squat form"""
    feedbacks = []

    # Check knee angle (depth)
    if all(k in kp and kp[k] is not None for k in ['left_hip', 'left_knee', 'left_ankle']):
        knee_angle = calculate_angle(kp['left_hip'], kp['left_knee'], kp['left_ankle'])

        if knee_angle < 70:
            feedbacks.append(FormFeedback(
                exercise="squat",
                feedback_type="angle",
                message=f"Excellent squat depth! Knee angle: {knee_angle:.1f}°",
                level=FeedbackLevel.EXCELLENT,
                frame_number=frame_num,
                details={'knee_angle': knee_angle, 'target_range': (70, 100)}
            ))
        elif knee_angle < 90:
            feedbacks.append(FormFeedback(
                exercise="squat",
                feedback_type="angle",
                message=f"Good squat depth. Knee angle: {knee_angle:.1f}°",
                level=FeedbackLevel.GOOD,
                frame_number=frame_num,
                details={'knee_angle': knee_angle, 'target_range': (70, 100)}
            ))
        elif knee_angle < 120:
            feedbacks.append(FormFeedback(
                exercise="squat",
                feedback_type="angle",
                message=f"Squat depth insufficient. Knee angle: {knee_angle:.1f}° (aim for <90°)",
                level=FeedbackLevel.WARNING,
                frame_number=frame_num,
                details={'knee_angle': knee_angle, 'target_range': (70, 100)}
            ))
        else:
            feedbacks.append(FormFeedback(
                exercise="squat",
                feedback_type="angle",
                message=f"Squat too shallow. Knee angle: {knee_angle:.1f}° (aim for <90°)",
                level=FeedbackLevel.CRITICAL,
                frame_number=frame_num,
                details={'knee_angle': knee_angle, 'target_range': (70, 100)}
            ))

    # Check back alignment
    if all(k in kp and kp[k] is not None for k in ['head', 'hip_mid', 'ankle_mid']):
        lean = calculate_body_alignment(kp['head'], kp['hip_mid'], kp['ankle_mid'])

        if lean < 15:
            feedbacks.append(FormFeedback(
                exercise="squat",
                feedback_type="alignment",
                message=f"Excellent upright posture! Lean: {lean:.1f}°",
                level=FeedbackLevel.EXCELLENT,
                frame_number=frame_num,
                details={'lean_angle': lean, 'target_max': 25}
            ))
        elif lean < 25:
            feedbacks.append(FormFeedback(
                exercise="squat",
                feedback_type="alignment",
                message=f"Good posture with slight forward lean: {lean:.1f}°",
                level=FeedbackLevel.GOOD,
                frame_number=frame_num,
                details={'lean_angle': lean, 'target_max': 25}
            ))
        else:
            feedbacks.append(FormFeedback(
                exercise="squat",
                feedback_type="alignment",
                message=f"Excessive forward lean: {lean:.1f}° (keep back straight)",
                level=FeedbackLevel.WARNING,
                frame_number=frame_num,
                details={'lean_angle': lean, 'target_max': 25}
            ))

    # Check knees over toes
    if all(k in kp and kp[k] is not None for k in ['left_knee', 'left_ankle']):
        knee_forward = kp['left_knee'][0] - kp['left_ankle'][0]

        if abs(knee_forward) < 0.05:
            feedbacks.append(FormFeedback(
                exercise="squat",
                feedback_type="position",
                message="Perfect knee position relative to toes",
                level=FeedbackLevel.EXCELLENT,
                frame_number=frame_num,
                details={'knee_forward_distance': knee_forward}
            ))
        elif abs(knee_forward) < 0.1:
            feedbacks.append(FormFeedback(
                exercise="squat",
                feedback_type="position",
                message="Knees slightly past toes (acceptable for some variations)",
                level=FeedbackLevel.GOOD,
                frame_number=frame_num,
                details={'knee_forward_distance': knee_forward}
            ))
        elif knee_forward > 0.1:
            feedbacks.append(FormFeedback(
                exercise="squat",
                feedback_type="position",
                message="Knees too far forward past toes (injury risk)",
                level=FeedbackLevel.WARNING,
                frame_number=frame_num,
                details={'knee_forward_distance': knee_forward}
            ))

    return feedbacks


def analyze_pushup(kp: Dict[str, np.ndarray], frame_num: int) -> List[FormFeedback]:
    """Analyze pushup form"""
    feedbacks = []

    # Check elbow angle (depth)
    if all(k in kp and kp[k] is not None for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
        elbow_angle = calculate_angle(kp['left_shoulder'], kp['left_elbow'], kp['left_wrist'])

        if elbow_angle < 70:
            feedbacks.append(FormFeedback(
                exercise="pushup",
                feedback_type="angle",
                message=f"Excellent depth! Elbow angle: {elbow_angle:.1f}°",
                level=FeedbackLevel.EXCELLENT,
                frame_number=frame_num,
                details={'elbow_angle': elbow_angle, 'target_range': (70, 110)}
            ))
        elif elbow_angle < 110:
            feedbacks.append(FormFeedback(
                exercise="pushup",
                feedback_type="angle",
                message=f"Good pushup depth. Elbow angle: {elbow_angle:.1f}°",
                level=FeedbackLevel.GOOD,
                frame_number=frame_num,
                details={'elbow_angle': elbow_angle, 'target_range': (70, 110)}
            ))
        else:
            feedbacks.append(FormFeedback(
                exercise="pushup",
                feedback_type="angle",
                message=f"Not lowering enough. Elbow angle: {elbow_angle:.1f}° (aim for <110°)",
                level=FeedbackLevel.WARNING,
                frame_number=frame_num,
                details={'elbow_angle': elbow_angle, 'target_range': (70, 110)}
            ))

    # Check body alignment (straight line from head to ankles)
    if all(k in kp and kp[k] is not None for k in ['head', 'hip_mid', 'ankle_mid']):
        # Calculate if hips are sagging or piking
        head_to_ankle = kp['ankle_mid'] - kp['head']
        hip_deviation = point_to_line_distance(kp['hip_mid'], kp['head'], kp['ankle_mid'])

        if hip_deviation < 0.05:
            feedbacks.append(FormFeedback(
                exercise="pushup",
                feedback_type="alignment",
                message="Perfect plank position! Body is straight",
                level=FeedbackLevel.EXCELLENT,
                frame_number=frame_num,
                details={'hip_deviation': hip_deviation, 'target_max': 0.05}
            ))
        elif hip_deviation < 0.1:
            feedbacks.append(FormFeedback(
                exercise="pushup",
                feedback_type="alignment",
                message="Good body alignment with minor deviation",
                level=FeedbackLevel.GOOD,
                frame_number=frame_num,
                details={'hip_deviation': hip_deviation, 'target_max': 0.05}
            ))
        else:
            # Determine if sagging or piking
            if kp['hip_mid'][1] > get_midpoint(kp['head'], kp['ankle_mid'])[1]:
                feedbacks.append(FormFeedback(
                    exercise="pushup",
                    feedback_type="alignment",
                    message="Hips are sagging - engage your core!",
                    level=FeedbackLevel.WARNING,
                    frame_number=frame_num,
                    details={'hip_deviation': hip_deviation, 'direction': 'sagging'}
                ))
            else:
                feedbacks.append(FormFeedback(
                    exercise="pushup",
                    feedback_type="alignment",
                    message="Hips are too high (piking) - lower them",
                    level=FeedbackLevel.WARNING,
                    frame_number=frame_num,
                    details={'hip_deviation': hip_deviation, 'direction': 'piking'}
                ))

    return feedbacks


def analyze_pullup(kp: Dict[str, np.ndarray], frame_num: int) -> List[FormFeedback]:
    """Analyze pullup form"""
    feedbacks = []

    # Check if chin is above bar (approximate bar position as above head)
    if all(k in kp and kp[k] is not None for k in ['head', 'left_wrist', 'right_wrist']):
        bar_height = min(kp['left_wrist'][1], kp['right_wrist'][1])  # Lower y = higher position
        head_height = kp['head'][1]

        if head_height < bar_height - 0.05:
            feedbacks.append(FormFeedback(
                exercise="pullup",
                feedback_type="position",
                message="Excellent! Chin clearly above bar",
                level=FeedbackLevel.EXCELLENT,
                frame_number=frame_num,
                details={'head_bar_distance': bar_height - head_height}
            ))
        elif head_height < bar_height:
            feedbacks.append(FormFeedback(
                exercise="pullup",
                feedback_type="position",
                message="Good rep! Chin at or slightly above bar",
                level=FeedbackLevel.GOOD,
                frame_number=frame_num,
                details={'head_bar_distance': bar_height - head_height}
            ))
        else:
            feedbacks.append(FormFeedback(
                exercise="pullup",
                feedback_type="position",
                message="Not pulling high enough - chin must clear the bar",
                level=FeedbackLevel.WARNING,
                frame_number=frame_num,
                details={'head_bar_distance': bar_height - head_height}
            ))

    # Check arm extension at bottom
    if all(k in kp and kp[k] is not None for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
        elbow_angle = calculate_angle(kp['left_shoulder'], kp['left_elbow'], kp['left_wrist'])

        if elbow_angle > 160:
            feedbacks.append(FormFeedback(
                exercise="pullup",
                feedback_type="angle",
                message="Full range of motion - arms fully extended",
                level=FeedbackLevel.EXCELLENT,
                frame_number=frame_num,
                details={'elbow_angle': elbow_angle, 'target_min': 160}
            ))
        elif elbow_angle > 140:
            feedbacks.append(FormFeedback(
                exercise="pullup",
                feedback_type="angle",
                message="Good extension at bottom",
                level=FeedbackLevel.GOOD,
                frame_number=frame_num,
                details={'elbow_angle': elbow_angle, 'target_min': 160}
            ))
        else:
            feedbacks.append(FormFeedback(
                exercise="pullup",
                feedback_type="angle",
                message="Not extending fully at bottom - complete full range",
                level=FeedbackLevel.WARNING,
                frame_number=frame_num,
                details={'elbow_angle': elbow_angle, 'target_min': 160}
            ))

    return feedbacks


def analyze_bench_press(kp: Dict[str, np.ndarray], frame_num: int) -> List[FormFeedback]:
    """Analyze bench press form"""
    feedbacks = []

    # Check elbow angle at bottom
    if all(k in kp and kp[k] is not None for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
        elbow_angle = calculate_angle(kp['left_shoulder'], kp['left_elbow'], kp['left_wrist'])

        if 70 <= elbow_angle <= 90:
            feedbacks.append(FormFeedback(
                exercise="bench_press",
                feedback_type="angle",
                message=f"Perfect elbow angle: {elbow_angle:.1f}°",
                level=FeedbackLevel.EXCELLENT,
                frame_number=frame_num,
                details={'elbow_angle': elbow_angle, 'target_range': (70, 90)}
            ))
        elif elbow_angle < 70:
            feedbacks.append(FormFeedback(
                exercise="bench_press",
                feedback_type="angle",
                message=f"Elbows too close to body: {elbow_angle:.1f}° (reduce shoulder stress)",
                level=FeedbackLevel.WARNING,
                frame_number=frame_num,
                details={'elbow_angle': elbow_angle, 'target_range': (70, 90)}
            ))
        elif elbow_angle > 90:
            feedbacks.append(FormFeedback(
                exercise="bench_press",
                feedback_type="angle",
                message=f"Elbows flaring too wide: {elbow_angle:.1f}° (injury risk)",
                level=FeedbackLevel.WARNING,
                frame_number=frame_num,
                details={'elbow_angle': elbow_angle, 'target_range': (70, 90)}
            ))

    # Check wrist alignment (should be roughly vertical)
    if all(k in kp and kp[k] is not None for k in ['left_elbow', 'left_wrist']):
        wrist_angle = abs(calculate_line_angle(kp['left_elbow'], kp['left_wrist']))

        if wrist_angle < 15:
            feedbacks.append(FormFeedback(
                exercise="bench_press",
                feedback_type="alignment",
                message="Wrists properly stacked over elbows",
                level=FeedbackLevel.EXCELLENT,
                frame_number=frame_num,
                details={'wrist_angle_from_vertical': wrist_angle}
            ))
        elif wrist_angle < 30:
            feedbacks.append(FormFeedback(
                exercise="bench_press",
                feedback_type="alignment",
                message="Wrist alignment acceptable",
                level=FeedbackLevel.GOOD,
                frame_number=frame_num,
                details={'wrist_angle_from_vertical': wrist_angle}
            ))
        else:
            feedbacks.append(FormFeedback(
                exercise="bench_press",
                feedback_type="alignment",
                message="Wrists bent - keep them straight to avoid injury",
                level=FeedbackLevel.WARNING,
                frame_number=frame_num,
                details={'wrist_angle_from_vertical': wrist_angle}
            ))

    return feedbacks


def analyze_situp(kp: Dict[str, np.ndarray], frame_num: int) -> List[FormFeedback]:
    """Analyze situp form"""
    feedbacks = []

    # Check hip angle
    if all(k in kp and kp[k] is not None for k in ['left_shoulder', 'left_hip', 'left_knee']):
        hip_angle = calculate_angle(kp['left_shoulder'], kp['left_hip'], kp['left_knee'])

        if hip_angle < 70:
            feedbacks.append(FormFeedback(
                exercise="situp",
                feedback_type="angle",
                message=f"Full situp! Hip angle: {hip_angle:.1f}°",
                level=FeedbackLevel.EXCELLENT,
                frame_number=frame_num,
                details={'hip_angle': hip_angle}
            ))
        elif hip_angle < 90:
            feedbacks.append(FormFeedback(
                exercise="situp",
                feedback_type="angle",
                message=f"Good situp. Hip angle: {hip_angle:.1f}°",
                level=FeedbackLevel.GOOD,
                frame_number=frame_num,
                details={'hip_angle': hip_angle}
            ))
        else:
            feedbacks.append(FormFeedback(
                exercise="situp",
                feedback_type="angle",
                message=f"Not coming up enough. Hip angle: {hip_angle:.1f}°",
                level=FeedbackLevel.WARNING,
                frame_number=frame_num,
                details={'hip_angle': hip_angle}
            ))

    # Check neck alignment (avoid pulling on neck)
    if all(k in kp and kp[k] is not None for k in ['head', 'shoulder_mid']):
        head_shoulder_distance = np.linalg.norm(kp['head'] - kp['shoulder_mid'])

        if head_shoulder_distance > 0.15:
            feedbacks.append(FormFeedback(
                exercise="situp",
                feedback_type="alignment",
                message="Good - not straining neck",
                level=FeedbackLevel.EXCELLENT,
                frame_number=frame_num,
                details={'head_shoulder_distance': head_shoulder_distance}
            ))
        elif head_shoulder_distance < 0.1:
            feedbacks.append(FormFeedback(
                exercise="situp",
                feedback_type="alignment",
                message="Warning: May be pulling on neck - use core muscles",
                level=FeedbackLevel.WARNING,
                frame_number=frame_num,
                details={'head_shoulder_distance': head_shoulder_distance}
            ))

    return feedbacks


def analyze_jumping_jacks(kp: Dict[str, np.ndarray], frame_num: int) -> List[FormFeedback]:
    """Analyze jumping jacks form"""
    feedbacks = []

    # Check arm extension overhead
    if all(k in kp and kp[k] is not None for k in ['left_shoulder', 'left_elbow', 'left_wrist', 'head']):
        elbow_angle = calculate_angle(kp['left_shoulder'], kp['left_elbow'], kp['left_wrist'])
        wrist_height = kp['left_wrist'][1]
        head_height = kp['head'][1]

        if wrist_height < head_height and elbow_angle > 160:
            feedbacks.append(FormFeedback(
                exercise="jumping_jacks",
                feedback_type="position",
                message="Arms fully extended overhead!",
                level=FeedbackLevel.EXCELLENT,
                frame_number=frame_num,
                details={'arm_extension': elbow_angle}
            ))
        elif wrist_height < head_height:
            feedbacks.append(FormFeedback(
                exercise="jumping_jacks",
                feedback_type="position",
                message="Arms overhead but not fully extended",
                level=FeedbackLevel.GOOD,
                frame_number=frame_num,
                details={'arm_extension': elbow_angle}
            ))
        else:
            feedbacks.append(FormFeedback(
                exercise="jumping_jacks",
                feedback_type="position",
                message="Raise arms higher - should go overhead",
                level=FeedbackLevel.WARNING,
                frame_number=frame_num,
                details={'arm_extension': elbow_angle}
            ))

    # Check feet position (wide vs together)
    if all(k in kp and kp[k] is not None for k in ['left_ankle', 'right_ankle']):
        feet_distance = abs(kp['left_ankle'][0] - kp['right_ankle'][0])

        if feet_distance > 0.3:
            feedbacks.append(FormFeedback(
                exercise="jumping_jacks",
                feedback_type="position",
                message="Good wide stance",
                level=FeedbackLevel.GOOD,
                frame_number=frame_num,
                details={'feet_distance': feet_distance}
            ))

    return feedbacks


def analyze_jump_rope(kp: Dict[str, np.ndarray], frame_num: int) -> List[FormFeedback]:
    """Analyze jump rope form"""
    feedbacks = []

    # Check if jumping (feet off ground)
    if all(k in kp and kp[k] is not None for k in ['left_ankle', 'left_knee', 'left_hip']):
        knee_angle = calculate_angle(kp['left_hip'], kp['left_knee'], kp['left_ankle'])

        if knee_angle < 160:
            feedbacks.append(FormFeedback(
                exercise="jump_rope",
                feedback_type="position",
                message="Jumping detected",
                level=FeedbackLevel.GOOD,
                frame_number=frame_num,
                details={'knee_angle': knee_angle}
            ))

    # Check upright posture
    if all(k in kp and kp[k] is not None for k in ['head', 'hip_mid', 'ankle_mid']):
        lean = calculate_body_alignment(kp['head'], kp['hip_mid'], kp['ankle_mid'])

        if lean < 10:
            feedbacks.append(FormFeedback(
                exercise="jump_rope",
                feedback_type="alignment",
                message="Excellent upright posture",
                level=FeedbackLevel.EXCELLENT,
                frame_number=frame_num,
                details={'lean_angle': lean}
            ))
        elif lean > 20:
            feedbacks.append(FormFeedback(
                exercise="jump_rope",
                feedback_type="alignment",
                message="Leaning too far forward - stand upright",
                level=FeedbackLevel.WARNING,
                frame_number=frame_num,
                details={'lean_angle': lean}
            ))

    return feedbacks


def analyze_clean_and_jerk(kp: Dict[str, np.ndarray], frame_num: int) -> List[FormFeedback]:
    """Analyze clean and jerk form"""
    feedbacks = []

    # Check overhead lockout (arms straight)
    if all(k in kp and kp[k] is not None for k in ['left_shoulder', 'left_elbow', 'left_wrist', 'head']):
        elbow_angle = calculate_angle(kp['left_shoulder'], kp['left_elbow'], kp['left_wrist'])
        wrist_height = kp['left_wrist'][1]
        head_height = kp['head'][1]

        if wrist_height < head_height - 0.1 and elbow_angle > 170:
            feedbacks.append(FormFeedback(
                exercise="clean_and_jerk",
                feedback_type="angle",
                message="Perfect overhead lockout!",
                level=FeedbackLevel.EXCELLENT,
                frame_number=frame_num,
                details={'elbow_angle': elbow_angle}
            ))
        elif elbow_angle < 160:
            feedbacks.append(FormFeedback(
                exercise="clean_and_jerk",
                feedback_type="angle",
                message="Arms not fully locked out overhead",
                level=FeedbackLevel.WARNING,
                frame_number=frame_num,
                details={'elbow_angle': elbow_angle}
            ))

    # Check upright torso
    if all(k in kp and kp[k] is not None for k in ['head', 'hip_mid', 'ankle_mid']):
        lean = calculate_body_alignment(kp['head'], kp['hip_mid'], kp['ankle_mid'])

        if lean < 15:
            feedbacks.append(FormFeedback(
                exercise="clean_and_jerk",
                feedback_type="alignment",
                message="Good upright torso position",
                level=FeedbackLevel.EXCELLENT,
                frame_number=frame_num,
                details={'lean_angle': lean}
            ))
        else:
            feedbacks.append(FormFeedback(
                exercise="clean_and_jerk",
                feedback_type="alignment",
                message="Leaning too far - maintain upright posture",
                level=FeedbackLevel.WARNING,
                frame_number=frame_num,
                details={'lean_angle': lean}
            ))

    return feedbacks


def analyze_tennis_serve(kp: Dict[str, np.ndarray], frame_num: int) -> List[FormFeedback]:
    """Analyze tennis serve form"""
    feedbacks = []

    # Check arm extension at contact
    if all(k in kp and kp[k] is not None for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
        elbow_angle = calculate_angle(kp['right_shoulder'], kp['right_elbow'], kp['right_wrist'])

        if elbow_angle > 160:
            feedbacks.append(FormFeedback(
                exercise="tennis_serve",
                feedback_type="angle",
                message="Full arm extension at contact point",
                level=FeedbackLevel.EXCELLENT,
                frame_number=frame_num,
                details={'elbow_angle': elbow_angle}
            ))
        elif elbow_angle < 140:
            feedbacks.append(FormFeedback(
                exercise="tennis_serve",
                feedback_type="angle",
                message="Not extending arm fully - reach higher",
                level=FeedbackLevel.WARNING,
                frame_number=frame_num,
                details={'elbow_angle': elbow_angle}
            ))

    # Check body rotation (shoulder rotation)
    if all(k in kp and kp[k] is not None for k in ['left_shoulder', 'right_shoulder', 'hip_mid']):
        shoulder_line_angle = calculate_line_angle(kp['left_shoulder'], kp['right_shoulder'])

        feedbacks.append(FormFeedback(
            exercise="tennis_serve",
            feedback_type="alignment",
            message=f"Shoulder rotation: {abs(shoulder_line_angle):.1f}°",
            level=FeedbackLevel.GOOD,
            frame_number=frame_num,
            details={'shoulder_rotation': shoulder_line_angle}
        ))

    return feedbacks


def analyze_tennis_forehand(kp: Dict[str, np.ndarray], frame_num: int) -> List[FormFeedback]:
    """Analyze tennis forehand form"""
    feedbacks = []

    # Similar to serve but check different mechanics
    if all(k in kp and kp[k] is not None for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
        elbow_angle = calculate_angle(kp['right_shoulder'], kp['right_elbow'], kp['right_wrist'])

        if 120 <= elbow_angle <= 160:
            feedbacks.append(FormFeedback(
                exercise="tennis_forehand",
                feedback_type="angle",
                message="Good arm position for forehand",
                level=FeedbackLevel.GOOD,
                frame_number=frame_num,
                details={'elbow_angle': elbow_angle}
            ))

    # Check hip and shoulder rotation
    if all(k in kp and kp[k] is not None for k in ['left_shoulder', 'right_shoulder']):
        shoulder_angle = calculate_line_angle(kp['left_shoulder'], kp['right_shoulder'])

        feedbacks.append(FormFeedback(
            exercise="tennis_forehand",
            feedback_type="alignment",
            message=f"Body rotation: {abs(shoulder_angle):.1f}°",
            level=FeedbackLevel.GOOD,
            frame_number=frame_num,
            details={'shoulder_rotation': shoulder_angle}
        ))

    return feedbacks


def analyze_baseball_swing(kp: Dict[str, np.ndarray], frame_num: int) -> List[FormFeedback]:
    """Analyze baseball swing form"""
    feedbacks = []

    # Check hip rotation
    if all(k in kp and kp[k] is not None for k in ['left_hip', 'right_hip']):
        hip_rotation = calculate_line_angle(kp['left_hip'], kp['right_hip'])

        if abs(hip_rotation) > 20:
            feedbacks.append(FormFeedback(
                exercise="baseball_swing",
                feedback_type="alignment",
                message=f"Good hip rotation: {abs(hip_rotation):.1f}°",
                level=FeedbackLevel.EXCELLENT,
                frame_number=frame_num,
                details={'hip_rotation': hip_rotation}
            ))
        else:
            feedbacks.append(FormFeedback(
                exercise="baseball_swing",
                feedback_type="alignment",
                message="Rotate hips more for power",
                level=FeedbackLevel.WARNING,
                frame_number=frame_num,
                details={'hip_rotation': hip_rotation}
            ))

    # Check weight transfer (back leg to front leg)
    if all(k in kp and kp[k] is not None for k in ['left_knee', 'right_knee']):
        knee_angles_similar = abs(kp['left_knee'][1] - kp['right_knee'][1]) < 0.1

        if not knee_angles_similar:
            feedbacks.append(FormFeedback(
                exercise="baseball_swing",
                feedback_type="position",
                message="Weight transfer detected",
                level=FeedbackLevel.GOOD,
                frame_number=frame_num,
                details={}
            ))

    return feedbacks


def analyze_baseball_pitch(kp: Dict[str, np.ndarray], frame_num: int) -> List[FormFeedback]:
    """Analyze baseball pitch form"""
    feedbacks = []

    # Check arm angle at release
    if all(k in kp and kp[k] is not None for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
        elbow_angle = calculate_angle(kp['right_shoulder'], kp['right_elbow'], kp['right_wrist'])

        if elbow_angle > 140:
            feedbacks.append(FormFeedback(
                exercise="baseball_pitch",
                feedback_type="angle",
                message="Good arm extension at release",
                level=FeedbackLevel.EXCELLENT,
                frame_number=frame_num,
                details={'elbow_angle': elbow_angle}
            ))
        elif elbow_angle < 120:
            feedbacks.append(FormFeedback(
                exercise="baseball_pitch",
                feedback_type="angle",
                message="Arm too bent - may reduce velocity",
                level=FeedbackLevel.WARNING,
                frame_number=frame_num,
                details={'elbow_angle': elbow_angle}
            ))

    # Check stride length
    if all(k in kp and kp[k] is not None for k in ['left_ankle', 'right_ankle']):
        stride = np.linalg.norm(kp['left_ankle'] - kp['right_ankle'])

        if stride > 0.4:
            feedbacks.append(FormFeedback(
                exercise="baseball_pitch",
                feedback_type="position",
                message="Good stride length for power",
                level=FeedbackLevel.EXCELLENT,
                frame_number=frame_num,
                details={'stride_length': stride}
            ))
        elif stride < 0.2:
            feedbacks.append(FormFeedback(
                exercise="baseball_pitch",
                feedback_type="position",
                message="Stride too short - extend more",
                level=FeedbackLevel.WARNING,
                frame_number=frame_num,
                details={'stride_length': stride}
            ))

    return feedbacks


def analyze_golf_swing(kp: Dict[str, np.ndarray], frame_num: int) -> List[FormFeedback]:
    """Analyze golf swing form"""
    feedbacks = []

    # Check posture (slight forward bend at hips)
    if all(k in kp and kp[k] is not None for k in ['head', 'hip_mid', 'ankle_mid']):
        lean = calculate_body_alignment(kp['head'], kp['hip_mid'], kp['ankle_mid'])

        if 20 <= lean <= 40:
            feedbacks.append(FormFeedback(
                exercise="golf_swing",
                feedback_type="alignment",
                message=f"Good golf posture with {lean:.1f}° forward tilt",
                level=FeedbackLevel.EXCELLENT,
                frame_number=frame_num,
                details={'forward_tilt': lean, 'target_range': (20, 40)}
            ))
        elif lean < 20:
            feedbacks.append(FormFeedback(
                exercise="golf_swing",
                feedback_type="alignment",
                message="Bend forward more at hips for proper posture",
                level=FeedbackLevel.WARNING,
                frame_number=frame_num,
                details={'forward_tilt': lean, 'target_range': (20, 40)}
            ))
        elif lean > 40:
            feedbacks.append(FormFeedback(
                exercise="golf_swing",
                feedback_type="alignment",
                message="Too much forward bend - stand more upright",
                level=FeedbackLevel.WARNING,
                frame_number=frame_num,
                details={'forward_tilt': lean, 'target_range': (20, 40)}
            ))

    # Check shoulder rotation
    if all(k in kp and kp[k] is not None for k in ['left_shoulder', 'right_shoulder']):
        shoulder_rotation = calculate_line_angle(kp['left_shoulder'], kp['right_shoulder'])

        if abs(shoulder_rotation) > 30:
            feedbacks.append(FormFeedback(
                exercise="golf_swing",
                feedback_type="alignment",
                message=f"Excellent shoulder turn: {abs(shoulder_rotation):.1f}°",
                level=FeedbackLevel.EXCELLENT,
                frame_number=frame_num,
                details={'shoulder_rotation': shoulder_rotation}
            ))
        elif abs(shoulder_rotation) < 15:
            feedbacks.append(FormFeedback(
                exercise="golf_swing",
                feedback_type="alignment",
                message="Rotate shoulders more for power",
                level=FeedbackLevel.WARNING,
                frame_number=frame_num,
                details={'shoulder_rotation': shoulder_rotation}
            ))

    return feedbacks


def analyze_bowl(kp: Dict[str, np.ndarray], frame_num: int) -> List[FormFeedback]:
    """Analyze bowling form"""
    feedbacks = []

    # Check arm swing (should be relatively straight)
    if all(k in kp and kp[k] is not None for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
        elbow_angle = calculate_angle(kp['right_shoulder'], kp['right_elbow'], kp['right_wrist'])

        if elbow_angle > 160:
            feedbacks.append(FormFeedback(
                exercise="bowl",
                feedback_type="angle",
                message="Excellent straight arm swing",
                level=FeedbackLevel.EXCELLENT,
                frame_number=frame_num,
                details={'elbow_angle': elbow_angle}
            ))
        elif elbow_angle < 140:
            feedbacks.append(FormFeedback(
                exercise="bowl",
                feedback_type="angle",
                message="Arm too bent - keep it straighter",
                level=FeedbackLevel.WARNING,
                frame_number=frame_num,
                details={'elbow_angle': elbow_angle}
            ))

    # Check knee bend at release
    if all(k in kp and kp[k] is not None for k in ['left_hip', 'left_knee', 'left_ankle']):
        knee_angle = calculate_angle(kp['left_hip'], kp['left_knee'], kp['left_ankle'])

        if 100 <= knee_angle <= 130:
            feedbacks.append(FormFeedback(
                exercise="bowl",
                feedback_type="angle",
                message="Good knee bend at release",
                level=FeedbackLevel.EXCELLENT,
                frame_number=frame_num,
                details={'knee_angle': knee_angle}
            ))
        elif knee_angle > 150:
            feedbacks.append(FormFeedback(
                exercise="bowl",
                feedback_type="angle",
                message="Bend knees more for better control",
                level=FeedbackLevel.WARNING,
                frame_number=frame_num,
                details={'knee_angle': knee_angle}
            ))

    return feedbacks


def analyze_strum_guitar(kp: Dict[str, np.ndarray], frame_num: int) -> List[FormFeedback]:
    """Analyze guitar strumming form (limited feedback for this action)"""
    feedbacks = []

    # Check arm position
    if all(k in kp and kp[k] is not None for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
        elbow_angle = calculate_angle(kp['right_shoulder'], kp['right_elbow'], kp['right_wrist'])

        if 70 <= elbow_angle <= 120:
            feedbacks.append(FormFeedback(
                exercise="strum_guitar",
                feedback_type="angle",
                message="Comfortable strumming position",
                level=FeedbackLevel.GOOD,
                frame_number=frame_num,
                details={'elbow_angle': elbow_angle}
            ))

    # Check posture
    if all(k in kp and kp[k] is not None for k in ['head', 'hip_mid']):
        lean = calculate_body_alignment(kp['head'], kp['hip_mid'], kp['hip_mid'])

        if lean < 20:
            feedbacks.append(FormFeedback(
                exercise="strum_guitar",
                feedback_type="alignment",
                message="Good upright posture while playing",
                level=FeedbackLevel.EXCELLENT,
                frame_number=frame_num,
                details={'posture_lean': lean}
            ))

    return feedbacks


# ============================================================================
# MAIN ANALYSIS ORCHESTRATOR
# ============================================================================

# Map exercise names to analysis functions
EXERCISE_ANALYZERS = {
    'squat': analyze_squat,
    'pushup': analyze_pushup,
    'pullup': analyze_pullup,
    'bench_press': analyze_bench_press,
    'situp': analyze_situp,
    'jumping_jacks': analyze_jumping_jacks,
    'jump_rope': analyze_jump_rope,
    'clean_and_jerk': analyze_clean_and_jerk,
    'tennis_serve': analyze_tennis_serve,
    'tennis_forehand': analyze_tennis_forehand,
    'baseball_swing': analyze_baseball_swing,
    'baseball_pitch': analyze_baseball_pitch,
    'golf_swing': analyze_golf_swing,
    'bowl': analyze_bowl,
    'strum_guitar': analyze_strum_guitar,
}


def analyze_frame(keypoints: np.ndarray, exercise: str, frame_num: int) -> Tuple[List[FormFeedback], float]:
    """
    Analyze a single frame for form feedback

    Args:
        keypoints: Array of shape (13, 3) with [x, y, visibility]
        exercise: Exercise name
        frame_num: Frame number for tracking

    Returns:
        (feedbacks, frame_score)
    """
    kp = extract_keypoints(keypoints)

    # Get exercise-specific analyzer
    analyzer = EXERCISE_ANALYZERS.get(exercise)
    if analyzer is None:
        return [], 5.0  # Default neutral score

    # Run analysis
    feedbacks = analyzer(kp, frame_num)

    # Calculate frame score based on feedback levels
    if not feedbacks:
        frame_score = 7.0  # Default good score
    else:
        level_scores = {
            FeedbackLevel.EXCELLENT: 10.0,
            FeedbackLevel.GOOD: 8.0,
            FeedbackLevel.WARNING: 5.0,
            FeedbackLevel.CRITICAL: 2.0
        }
        frame_score = np.mean([level_scores[fb.level] for fb in feedbacks])

    return feedbacks, frame_score


def analyze_video(video_keypoints: List[np.ndarray], exercise: str) -> ExerciseAnalysis:
    """
    Analyze entire video and generate post-analysis report

    Args:
        video_keypoints: List of keypoint arrays (one per frame), each shape (13, 3)
        exercise: Exercise name

    Returns:
        ExerciseAnalysis with comprehensive feedback
    """
    all_feedbacks = []
    frame_scores = []

    # Analyze each frame
    for frame_num, keypoints in enumerate(video_keypoints):
        feedbacks, score = analyze_frame(keypoints, exercise, frame_num)
        all_feedbacks.extend(feedbacks)
        frame_scores.append(score)

    # Calculate overall score
    overall_score = np.mean(frame_scores) if frame_scores else 5.0

    # Generate summary
    summary = generate_summary(exercise, all_feedbacks, overall_score)

    # Attempt rep counting (simple heuristic based on score variation)
    rep_count = estimate_rep_count(frame_scores) if len(frame_scores) > 10 else None

    return ExerciseAnalysis(
        exercise=exercise,
        overall_score=overall_score,
        frame_scores=frame_scores,
        feedbacks=all_feedbacks,
        summary=summary,
        rep_count=rep_count
    )


def generate_summary(exercise: str, feedbacks: List[FormFeedback], overall_score: float) -> str:
    """Generate human-readable summary of analysis"""

    # Count feedback by level
    excellent_count = sum(1 for fb in feedbacks if fb.level == FeedbackLevel.EXCELLENT)
    good_count = sum(1 for fb in feedbacks if fb.level == FeedbackLevel.GOOD)
    warning_count = sum(1 for fb in feedbacks if fb.level == FeedbackLevel.WARNING)
    critical_count = sum(1 for fb in feedbacks if fb.level == FeedbackLevel.CRITICAL)

    summary_parts = [
        f"=== {exercise.upper().replace('_', ' ')} FORM ANALYSIS ===\n",
        f"Overall Score: {overall_score:.1f}/10\n",
        f"\nFeedback Summary:",
        f"  ✓ Excellent: {excellent_count}",
        f"  ✓ Good: {good_count}",
        f"  ⚠ Warnings: {warning_count}",
        f"  ✗ Critical: {critical_count}\n",
    ]

    # Add key issues
    if warning_count + critical_count > 0:
        summary_parts.append("\nKey Areas for Improvement:")

        # Get most common warning types
        warning_types = {}
        for fb in feedbacks:
            if fb.level in [FeedbackLevel.WARNING, FeedbackLevel.CRITICAL]:
                key = fb.feedback_type
                if key not in warning_types:
                    warning_types[key] = []
                warning_types[key].append(fb.message)

        for feedback_type, messages in warning_types.items():
            summary_parts.append(f"\n  {feedback_type.upper()}:")
            # Get unique messages
            unique_messages = list(set(messages))[:3]  # Top 3
            for msg in unique_messages:
                summary_parts.append(f"    - {msg}")

    # Add strengths
    if excellent_count > 0:
        summary_parts.append("\n\nStrengths:")
        excellent_feedbacks = [fb for fb in feedbacks if fb.level == FeedbackLevel.EXCELLENT]
        unique_excellent = list(set([fb.message for fb in excellent_feedbacks]))[:3]
        for msg in unique_excellent:
            summary_parts.append(f"  ✓ {msg}")

    # Overall recommendation
    summary_parts.append("\n\n" + "="*50)
    if overall_score >= 8.5:
        summary_parts.append("\n🎉 EXCELLENT FORM! Keep up the great work!")
    elif overall_score >= 7.0:
        summary_parts.append("\n👍 GOOD FORM with minor improvements possible")
    elif overall_score >= 5.5:
        summary_parts.append("\n⚠️  MODERATE FORM - focus on the areas highlighted above")
    else:
        summary_parts.append("\n⚠️  FORM NEEDS ATTENTION - consider reviewing technique")

    return "\n".join(summary_parts)


def estimate_rep_count(frame_scores: List[float], threshold: float = 0.3) -> Optional[int]:
    """
    Estimate repetition count from score variation
    Very basic heuristic - counts peaks in frame scores
    """
    if len(frame_scores) < 10:
        return None

    scores = np.array(frame_scores)

    # Smooth scores
    window = min(5, len(scores) // 10)
    if window < 2:
        return None

    smoothed = np.convolve(scores, np.ones(window)/window, mode='valid')

    # Find peaks (local maxima)
    reps = 0
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
            if smoothed[i] - min(smoothed) > threshold * (max(smoothed) - min(smoothed)):
                reps += 1

    return reps if reps > 0 else None


# ============================================================================
# EXPORT AND VISUALIZATION
# ============================================================================

def export_analysis_json(analysis: ExerciseAnalysis, output_path: str):
    """Export analysis to JSON file"""
    import json

    def convert_to_python_types(obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_python_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    data = {
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
                'details': convert_to_python_types(fb.details)
            }
            for fb in analysis.feedbacks
        ],
        'frame_scores': [float(s) for s in analysis.frame_scores]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def export_analysis_csv(analysis: ExerciseAnalysis, output_path: str):
    """Export frame-by-frame analysis to CSV"""
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'Score', 'Feedback Count', 'Issues'])

        # Group feedbacks by frame
        frame_feedbacks = {}
        for fb in analysis.feedbacks:
            if fb.frame_number not in frame_feedbacks:
                frame_feedbacks[fb.frame_number] = []
            frame_feedbacks[fb.frame_number].append(fb)

        for frame_num, score in enumerate(analysis.frame_scores):
            fbs = frame_feedbacks.get(frame_num, [])
            issues = '; '.join([f"{fb.level.value}: {fb.message}" for fb in fbs])
            writer.writerow([frame_num, f"{score:.2f}", len(fbs), issues])
