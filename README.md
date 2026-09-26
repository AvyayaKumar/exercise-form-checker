# Exercise Form Checker

A Streamlit app that runs a fine-tuned YOLOv8 pose model over an exercise video, tracks 13 body keypoints frame by frame, and applies hand-written angle and position rules to produce feedback messages and a 0–10 rule-based score.

It's a pose-estimation project with a rule layer on top. It is not a validated coaching tool: the feedback has never been compared with ratings from coaches or trainers, so the score says how many of my rules a video passed, not how good the form actually is.

## What it does, precisely

1. **Pose and movement type.** Every frame goes through `best_full.pt`, a YOLOv8n-pose model fine-tuned on Penn Action. For each frame it returns person boxes, a class label from the 15 Penn Action movement types, and 13 keypoints (head, shoulders, elbows, wrists, hips, knees, ankles). The app keeps the first detected person's keypoints. The movement type for the whole video is the class of the single highest-confidence box in any frame.
2. **Rules per frame.** `form_analysis.py` has one function per movement type. Each computes a few 2D measurements from the keypoints (joint angles such as hip–knee–ankle, torso lean from vertical, left/right symmetry, stance or stride width) and compares them with fixed thresholds that I chose. For example, a squat frame with a knee angle under 70° is "excellent", under 90° "good", under 120° "warning", and anything else "critical". Keypoints with confidence below 0.3 are skipped.
3. **Score.** Each message maps to a number (excellent 10, good 8, warning 5, critical 2). A frame's score is the mean of its messages (7 if no rule fired), and the video's score is the mean over all frames.
4. **Output.** An annotated video with the skeleton drawn on, a text summary of the most common warnings, a rough rep count (peaks in the smoothed per-frame score), and JSON/CSV exports of every message.

The 15 movement types are the Penn Action classes: pull-up, push-up, squat, sit-up, bench press, jumping jacks, jump rope, tennis serve, tennis forehand, baseball swing, baseball pitch, golf swing, bowling, clean and jerk, and guitar strumming.

## Model

- **Architecture:** YOLOv8n-pose, fine-tuned (`best_full.pt`, Ultralytics 8.3.226, trained Nov 2025)
- **Data:** Penn Action (163,841 frames across 2,326 videos, 15 classes)
- **Evaluation:** the checkpoint's own validation metrics are 95.24% pose mAP@50 and 78.37% pose mAP@50-95. I don't quote these as held-out results: the train/validation split isn't in this repo, so I can't confirm the validation frames came from videos the model never saw. Adjacent frames from one video are nearly identical, so a frame-level split would overstate accuracy. Movement-type classification accuracy hasn't been measured on its own.

## Limitations

- **The rules aren't validated.** The thresholds are my own estimates, not taken from a coaching or biomechanics source, and no one has checked the messages against expert judgment. Treat the output as a description of measured angles, not a verdict on form or injury risk.
- **Every frame counts toward the score.** A squat is scored on the standing frames too, where the knee angle is large and gets marked "too shallow", so the score mixes rep phases together instead of judging the bottom of each rep.
- **A known bug in distance rules.** Some checks (knees past toes, hip sag in push-ups, stance and stride width) compare distances with thresholds like 0.05 or 0.1, which assume normalized coordinates, but the app passes pixel coordinates. Those checks are almost always off by orders of magnitude.
- **One 2D camera.** Angles are measured in the image plane, so they change with camera angle, and depth (for example knees caving inward seen from the side) is invisible. Some rules only look at the left side of the body.
- **One person, one movement per video.** Only the first detected person is analyzed, and one label is picked for the whole clip.
- **Rep counting is a heuristic** based on score peaks, not on movement, and hasn't been checked against real counts.
- **Penn Action is mostly sports and broadcast footage**, not home workout videos, so pose accuracy on typical phone recordings is unmeasured.

## Running it

Runs locally; there's no hosted demo.

```bash
pip install -r requirements.txt
streamlit run app_with_feedback.py      # upload a video
streamlit run app_with_calibration.py   # live camera, with an optional per-user calibration step
```

`app_realtime.py` and `app_realtime_webrtc.py` are earlier live-camera versions.

## Stack

Streamlit, Ultralytics YOLOv8, OpenCV, NumPy. The rule layer (`form_analysis.py`) is about 1,300 lines.

## License

MIT

## Acknowledgments

- Penn Action dataset for training data
- Ultralytics for YOLOv8
