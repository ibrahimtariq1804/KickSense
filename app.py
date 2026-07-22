import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import tempfile
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="KickSense",
    page_icon="⚽",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #1F3864, #2E75B6);
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #f0f4f8;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border-left: 5px solid #2E75B6;
    }
    .prediction-box {
        background: linear-gradient(135deg, #1F3864, #2E75B6);
        border-radius: 10px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>⚽ KickSense</h1>
    <p>Intelligent Penalty Kick Analysis Using Pose Estimation</p>
    <p><small>Bahria University — Final Year Project</small></p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        'outputs/model/best_model_improved.keras'
    )
    return model

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def extract_frames(video_path, max_frames=26):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, max_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

def get_keypoints_from_frame(frame):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.3
    ) as pose:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            annotated = image_rgb.copy()
            mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            kp = []
            for lm in results.pose_landmarks.landmark:
                kp.append([lm.x, lm.y, lm.z, lm.visibility])
            return np.array(kp), annotated

        return None, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def normalize_keypoints(keypoints):
    normalized = keypoints.copy()
    left_hip = keypoints[23, :2]
    right_hip = keypoints[24, :2]
    hip_center = (left_hip + right_hip) / 2
    left_shoulder = keypoints[11, :2]
    right_shoulder = keypoints[12, :2]
    shoulder_center = (left_shoulder + right_shoulder) / 2
    torso_height = np.linalg.norm(shoulder_center - hip_center)
    if torso_height < 1e-6:
        torso_height = 1.0
    normalized[:, 0] = (keypoints[:, 0] - hip_center[0]) / torso_height
    normalized[:, 1] = (keypoints[:, 1] - hip_center[1]) / torso_height
    return normalized

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ⚽ About KickSense")
    st.markdown("""
    KickSense analyzes a footballer's body pose
    during a penalty kick and predicts:

    - 🎯 **Shot Direction** (Left/Center/Right)
    - 💪 **Shot Power** (Scale 1-10)

    **How it works:**
    1. Upload a penalty kick video
    2. MediaPipe extracts 33 body keypoints
    3. CNN-LSTM model analyzes the pose
    4. Prediction is displayed instantly
    """)
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    - **Architecture:** CNN-LSTM
    - **Accuracy:** 50%
    - **Training clips:** 117
    - **Keypoints:** 33 per frame
    - **Sequence:** 26 frames
    """)
    st.markdown("---")
    st.markdown("**Ashir Ijaz & Ibrahim Tariq**")
    st.markdown("*Supervisor: Ms. Zainab Iftikhar*")

# ============================================================
# MAIN CONTENT
# ============================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Penalty Kick Video")
    uploaded_file = st.file_uploader(
        "Upload a penalty kick clip (.mp4)",
        type=['mp4', 'avi', 'mov'],
        help="Upload a short penalty kick video clip"
    )
    if uploaded_file:
        st.video(uploaded_file)

with col2:
    st.markdown("### 📋 Instructions")
    st.info("""
    **For best results:**
    - Upload a clip showing the full run-up
    - Make sure the striker is clearly visible
    - Camera should show striker + goalkeeper + goal
    - Clip length: 5-15 seconds recommended
    """)
    st.markdown("### 🎯 What We Predict")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Shot Direction", "Left / Center / Right")
    with c2:
        st.metric("Shot Power", "Scale 1-10")

# ============================================================
# PREDICTION
# ============================================================
if uploaded_file:
    if st.button("🔍 Analyze Penalty Kick", type="primary",
                 use_container_width=True):

        with st.spinner("Analyzing penalty kick..."):

            # Save uploaded file temporarily
            tfile = tempfile.NamedTemporaryFile(
                delete=False, suffix='.mp4'
            )
            tfile.write(uploaded_file.read())
            video_path = tfile.name

            # Load model
            model = load_model()

            # Progress
            progress = st.progress(0)
            status = st.empty()

            # Extract frames
            status.text("Extracting frames...")
            frames = extract_frames(video_path, max_frames=26)
            progress.progress(25)

            # Extract keypoints
            status.text("Estimating pose...")
            keypoints_list = []
            best_frame = None

            for frame in frames:
                kp, annotated = get_keypoints_from_frame(frame)
                if kp is not None:
                    kp_norm = normalize_keypoints(kp)
                    keypoints_list.append(kp_norm.flatten())
                    if best_frame is None:
                        best_frame = annotated

            progress.progress(60)

            if len(keypoints_list) < 5:
                st.error("Could not detect pose. Please try a clearer video.")
            else:
                status.text("Processing sequences...")
                while len(keypoints_list) < 26:
                    keypoints_list.append(keypoints_list[-1])
                keypoints_list = keypoints_list[:26]

                sequence = np.array(keypoints_list)
                sequence = sequence.reshape(1, 26, 132)
                progress.progress(80)

                # Predict
                status.text("Predicting...")
                predictions = model.predict(sequence, verbose=0)
                direction_probs = predictions[0][0]
                power_pred = predictions[1][0][0] * 10

                direction_labels = ['Left', 'Center', 'Right']
                direction_idx = np.argmax(direction_probs)
                direction = direction_labels[direction_idx]
                confidence = direction_probs[direction_idx] * 100

                progress.progress(100)
                status.empty()
                progress.empty()

                # ============================================================
                # RESULTS
                # ============================================================
                st.markdown("---")
                st.markdown("## 🎯 Prediction Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    direction_emoji = {
                        'Left': '⬅️',
                        'Center': '⬆️',
                        'Right': '➡️'
                    }
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>{direction_emoji[direction]}</h2>
                        <h3>Shot Direction</h3>
                        <h1>{direction}</h1>
                        <p>Confidence: {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    if power_pred >= 7:
                        power_color = "#1a5c38"
                        power_label = "High Power"
                    elif power_pred >= 4:
                        power_color = "#8B6914"
                        power_label = "Medium Power"
                    else:
                        power_color = "#8B1A1A"
                        power_label = "Low Power"

                    st.markdown(f"""
                    <div style="background: {power_color};
                         border-radius: 10px; padding: 20px;
                         color: white; text-align: center;">
                        <h2>💪</h2>
                        <h3>Shot Power</h3>
                        <h1>{power_pred:.1f}/10</h1>
                        <p>{power_label}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h2>📊</h2>
                        <h3>Probabilities</h3>
                        <p>⬅️ Left: {direction_probs[0]*100:.1f}%</p>
                        <p>⬆️ Center: {direction_probs[1]*100:.1f}%</p>
                        <p>➡️ Right: {direction_probs[2]*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Pose visualization
                st.markdown("---")
                st.markdown("### 🦴 Pose Estimation Visualization")

                if best_frame is not None:
                    st.image(
                        best_frame,
                        caption="Detected skeletal pose on penalty kick frame",
                        use_column_width=True
                    )

                # Direction probability bars
                st.markdown("### 📊 Direction Probability Breakdown")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⬅️ Left", f"{direction_probs[0]*100:.1f}%")
                    st.progress(float(direction_probs[0]))
                with col2:
                    st.metric("⬆️ Center", f"{direction_probs[1]*100:.1f}%")
                    st.progress(float(direction_probs[1]))
                with col3:
                    st.metric("➡️ Right", f"{direction_probs[2]*100:.1f}%")
                    st.progress(float(direction_probs[2]))

            # Cleanup
            os.unlink(video_path)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 10px;">
    <p>KickSense — Bahria University Islamabad |
    Department of Computer Science | Session 2026-2027</p>
    <p>Ashir Ijaz (01-135231-017) |
    Ibrahim Tariq (01-135231-037)</p>
</div>
""", unsafe_allow_html=True)