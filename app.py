import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import os

from scripts.contact_detector import detect_contact_and_extract_sequence

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="KickSense — AI Penalty Anticipation",
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
        padding: 24px;
        background: linear-gradient(135deg, #091224, #13274f);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
    }
    .metric-card {
        background: #0f182c;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
        color: white;
    }
    .prediction-box {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 8px 24px -4px rgba(16, 185, 129, 0.4);
    }
    .power-box {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 8px 24px -4px rgba(245, 158, 11, 0.4);
    }
    .telemetry-box {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>⚽ KickSense AI</h1>
    <p style="font-size: 1.1rem; color: #38bdf8;">Penalty Kick Anticipation & Striker Biomechanics Platform</p>
    <p><small style="color: #94a3b8;">Bahria University Islamabad · Final Year Project · Ashir Ijaz & Ibrahim Tariq</small></p>
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
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ⚽ About KickSense")
    st.markdown("""
    KickSense isolates a striker's approach run-up and predicts shot trajectory **strictly before or at foot-ball contact**:

    - 🎯 **Direction Anticipation** (Left / Center / Right)
    - 💪 **Power Rating** (1.0 to 10.0 scale)
    - 🦴 **Striker Biomechanics** (Torso lean, hip angle, plant offset)
    """)
    st.markdown("---")
    st.markdown("### 📊 Verified Dataset & Metrics")
    st.markdown("""
    - **Total Labeled Clips:** 460
    - **Clean Sequences:** 426
    - **Architecture:** Multi-Scale Inception + BiLSTM
    - **Direction Accuracy:** **53.1%** (vs 33.3% random)
    - **Weighted Precision:** **60.0%**
    - **Center Precision:** **78.6%**
    - **Shot Power MAE:** **2.12**
    - **Input Contract:** (26 frames, 132 features)
    """)
    st.markdown("---")
    st.markdown("**Supervisor:** Ms. Zainab Iftikhar")
    st.markdown("**Institution:** Bahria University Islamabad")

# ============================================================
# MAIN CONTENT
# ============================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Penalty Kick Video")
    uploaded_file = st.file_uploader(
        "Upload penalty kick video (.mp4, .mov, .avi)",
        type=['mp4', 'avi', 'mov'],
        help="Supports broadcast penalties or smartphone recordings"
    )
    if uploaded_file:
        st.video(uploaded_file)

with col2:
    st.markdown("### 📋 Smart Contact Detection")
    st.info("""
    **Automatic Temporal Alignment:**
    - Full penalty video clips are automatically analyzed.
    - MediaPipe tracks 33 3D skeletal landmarks.
    - System calculates ankle velocity peaks and plant foot deceleration to isolate the foot-ball strike moment ($t \\le \\text{strike}$).
    - Trajectory features are normalized to hip center and torso height scale.
    """)
    st.markdown("### 🎯 System Predictions")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Direction Anticipation", "Left / Center / Right")
    with c2:
        st.metric("Power Rating", "Scale 1.0 – 10.0")

# ============================================================
# PREDICTION PIPELINE
# ============================================================
if uploaded_file:
    if st.button("🔍 Analyze Penalty Kick", type="primary", use_container_width=True):
        with st.spinner("Extracting kinematics and detecting foot-ball contact..."):

            # Save uploaded file temporarily
            ext = os.path.splitext(uploaded_file.name)[1] or '.mp4'
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tfile:
                tfile.write(uploaded_file.read())
                video_path = tfile.name

            try:
                # 1. Run Smart Contact & Pose Detection
                res = detect_contact_and_extract_sequence(video_path, target_frames=26)
                
                if res is None:
                    st.error("Could not detect striker's body pose. Please provide a clear clip showing the full striker body.")
                else:
                    # 2. Model Inference
                    model = load_model()
                    seq = res['sequence'].reshape(1, 26, 132)
                    predictions = model.predict(seq, verbose=0)

                    direction_probs = predictions[0][0] # Left, Center, Right
                    power_norm = float(predictions[1][0][0])
                    power_pred = float(np.clip(power_norm * 10.0, 1.0, 10.0))

                    direction_labels = ['Left', 'Center', 'Right']
                    direction_idx = int(np.argmax(direction_probs))
                    direction = direction_labels[direction_idx]
                    confidence = float(direction_probs[direction_idx] * 100.0)

                    if power_pred >= 7.5:
                        power_label = "High Power (Driven Strike)"
                    elif power_pred >= 4.5:
                        power_label = "Medium Power (Balanced)"
                    else:
                        power_label = "Low Power (Finesse Placement)"

                    # ============================================================
                    # RESULTS DISPLAY
                    # ============================================================
                    st.markdown("---")
                    st.markdown("## 🎯 Prediction Results")

                    col_dir, col_pwr, col_prob = st.columns(3)

                    with col_dir:
                        direction_emoji = {'Left': '⬅️', 'Center': '⬆️', 'Right': '➡️'}
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h1 style="font-size: 3rem; margin: 0;">{direction_emoji[direction]}</h1>
                            <h3 style="margin-top: 5px;">Predicted Direction</h3>
                            <h2 style="font-size: 2.2rem; font-weight: bold; margin: 5px 0;">{direction.upper()}</h2>
                            <p style="font-size: 1.1rem; opacity: 0.95;">Confidence: <strong>{confidence:.1f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_pwr:
                        st.markdown(f"""
                        <div class="power-box">
                            <h1 style="font-size: 3rem; margin: 0;">💪</h1>
                            <h3 style="margin-top: 5px;">Shot Power Rating</h3>
                            <h2 style="font-size: 2.2rem; font-weight: bold; margin: 5px 0;">{power_pred:.1f} / 10</h2>
                            <p style="font-size: 1.1rem; opacity: 0.95;">{power_label}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_prob:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>📊 Class Probabilities</h3>
                            <hr style="border-color: rgba(255,255,255,0.1); margin: 10px 0;">
                            <p style="font-size: 1.05rem;">⬅️ <strong>Left:</strong> {direction_probs[0]*100:.1f}%</p>
                            <p style="font-size: 1.05rem;">⬆️ <strong>Center:</strong> {direction_probs[1]*100:.1f}%</p>
                            <p style="font-size: 1.05rem;">➡️ <strong>Right:</strong> {direction_probs[2]*100:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Strike Contact Skeleton
                    st.markdown("---")
                    st.markdown(f"### 🦴 Contact Frame Skeletal Pose (Frame #{res['strike_frame_idx']})")

                    if res['strike_image'] is not None:
                        rgb_strike = cv2.cvtColor(res['strike_image'], cv2.COLOR_BGR2RGB)
                        st.image(
                            rgb_strike,
                            caption=f"Strike moment isolated at frame {res['strike_frame_idx']} of {res['total_frames']} ({res['fps']:.1f} FPS)",
                            use_column_width=True
                        )

                    # Biomechanics Telemetry
                    st.markdown("### 📐 Striker Biomechanics Diagnostics")
                    bio = res['biomechanics']
                    b1, b2, b3, b4 = st.columns(4)
                    with b1:
                        st.metric("Torso Lean Angle", f"{bio['torso_lean_deg']:.1f}°", bio['torso_lean_label'])
                    with b2:
                        st.metric("Pelvis Rotation", f"{bio['pelvis_angle_deg']:.1f}°", bio['pelvis_angle_label'])
                    with b3:
                        st.metric("Peak Ankle Speed", f"{bio['kicking_ankle_peak_speed']:.0f} px/s", "Swing Momentum")
                    with b4:
                        st.metric("Plant Foot Offset", f"{bio['plant_to_ball_dist']:.2f}", "Support Stance")

            finally:
                if os.path.exists(video_path):
                    try:
                        os.unlink(video_path)
                    except OSError:
                        pass

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 10px;">
    <p>KickSense — Bahria University Islamabad | Department of Computer Science</p>
    <p>Ashir Ijaz (01-135231-017) | Ibrahim Tariq (01-135231-037) | Supervisor: Ms. Zainab Iftikhar</p>
</div>
""", unsafe_allow_html=True)