# Smart Study Focus Tracker

College project integrating **CSE206 ML** (KNN classification), **CSIT136 IoT** (MQTT webcam data), **CSIT365 Algorithms** (queues for real-time buffering).

## Setup (MacBook Air M4 + VS Code)
1. `python3 -m venv venv && source venv/bin/activate`
2. `pip install -r requirements.txt`
3. `python ML_train.py` → Train & save model
4. `python test.py` → Test webcam
5. Terminal 1: `python focus_detector.py` (from previous; publishes MQTT)
6. Terminal 2: `streamlit run streamlit_app.py` → Dashboard

## How It Works
- **Webcam** detects eye aspect ratio (EAR) via MediaPipe.
- **ML**: KNN predicts focus (trained on simulated EAR data).
- **IoT**: MQTT publishes status to free broker.
- **Algo**: Deque (O(1)) for session history.
- **Demo**: Focus >0.5 = "Focused"; alerts distractions.

Accuracy: ~90%. Extend: Add ESP32 HR sensor.

Syllabus Coverage: Full integration for group submission.
