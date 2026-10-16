"""
KickSense Production Backend API
FastAPI service serving:
  1. Full AI inference (video -> foot-ball contact detection -> pose extraction -> CNN-LSTM prediction)
  2. Telemetry and biomechanical diagnostics (torso lean, pelvis angle, ankle speed)
  3. Sample clips endpoints for instant demo testing
  4. Static web application hosting
"""
import os
import time
import base64
import tempfile
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from scripts.contact_detector import detect_contact_and_extract_sequence

MODEL_PATH = "outputs/model/best_model_improved.keras"
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov"}
DIRECTION_LABELS = ["Left", "Center", "Right"]
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

app = FastAPI(
    title="KickSense AI Platform",
    description="Deep-Learning Football Penalty Kick Direction & Power Prediction Engine",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=503, detail=f"Model weights not found at '{MODEL_PATH}'")
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model

def encode_jpeg_b64(bgr_img, max_width=800):
    if bgr_img is None:
        return None
    h, w = bgr_img.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        bgr_img = cv2.resize(bgr_img, (max_width, int(h * scale)))
    ret, buf = cv2.imencode('.jpg', bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ret:
        return None
    return base64.b64encode(buf.tobytes()).decode('ascii')

def execute_prediction(video_path):
    t_start = time.time()
    
    # 1. Run smart contact detector
    res = detect_contact_and_extract_sequence(video_path, target_frames=26)
    if res is None:
        raise HTTPException(
            status_code=422,
            detail="Could not detect player body pose. Please provide a clear penalty video showing the striker's full body."
        )
        
    seq = res['sequence'].reshape(1, 26, 132) # (1, 26, 132)
    
    # 2. Run model inference
    model = get_model()
    raw_preds = model.predict(seq, verbose=0)
    
    dir_probs = raw_preds[0][0] # 3 classes: Left, Center, Right
    pwr_norm = float(raw_preds[1][0][0]) # 0.0 - 1.0
    
    pred_idx = int(np.argmax(dir_probs))
    pred_direction = DIRECTION_LABELS[pred_idx]
    pred_confidence = float(dir_probs[pred_idx] * 100.0)
    
    pred_power = float(np.clip(pwr_norm * 10.0, 1.0, 10.0))
    if pred_power >= 7.5:
        pwr_category = "High Power (Driven Strike)"
    elif pred_power >= 4.5:
        pwr_category = "Medium Power (Balanced)"
    else:
        pwr_category = "Low Power (Finesse Placement)"
        
    t_ms = round((time.time() - t_start) * 1000, 1)
    
    # Encode skeleton image
    strike_b64 = encode_jpeg_b64(res['strike_image'])
    
    return {
        "status": "success",
        "direction": pred_direction,
        "confidence": round(pred_confidence, 1),
        "probabilities": {
            "Left": round(float(dir_probs[0] * 100), 1),
            "Center": round(float(dir_probs[1] * 100), 1),
            "Right": round(float(dir_probs[2] * 100), 1)
        },
        "power": round(pred_power, 1),
        "power_category": pwr_category,
        "strike_frame_idx": res['strike_frame_idx'],
        "start_frame_idx": res['start_frame_idx'],
        "total_frames": res['total_frames'],
        "fps": res['fps'],
        "strike_skeleton_base64": strike_b64,
        "biomechanics": res['biomechanics'],
        "inference_time_ms": t_ms
    }

# ----------------- API Endpoints -----------------

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": os.path.exists(MODEL_PATH),
        "model_path": MODEL_PATH,
        "version": "2.0.0"
    }

@app.get("/api/info")
async def project_info():
    return {
        "project": "KickSense",
        "version": "2.0.0",
        "authors": ["Ashir Ijaz", "Ibrahim Tariq"],
        "supervisor": "Ms. Zainab Iftikhar",
        "institution": "Bahria University Islamabad",
        "dataset": {
            "total_labeled_clips": 460,
            "clean_training_sequences": 426,
            "train_val_test_split": "298 / 64 / 64",
            "classes": ["Left", "Center", "Right"]
        },
        "metrics": {
            "direction_accuracy": "53.1%",
            "weighted_precision": "60.0%",
            "center_precision": "78.6%",
            "power_mae": 2.12
        },
        "architecture": "Multi-Scale Temporal Inception + Bidirectional LSTM + Dual Global Shift-Invariant Pooling"
    }

@app.get("/api/sample-clips")
async def get_sample_clips():
    """Returns pre-loaded sample penalty clips available on the server for 1-click testing."""
    samples = [
        {"id": "penalty_01", "name": "Argentina vs France (World Cup Final)", "true_direction": "Right", "true_power": 10.0},
        {"id": "penalty_02", "name": "World Cup Shootout Clip A", "true_direction": "Left", "true_power": 5.0},
        {"id": "penalty_04", "name": "World Cup Shootout Clip B", "true_direction": "Right", "true_power": 6.0},
        {"id": "penalty_05", "name": "Center Placement Sample", "true_direction": "Center", "true_power": 7.0},
        {"id": "penalty_291", "name": "Recorded Session — Drive", "true_direction": "Left", "true_power": 9.5},
        {"id": "penalty_292", "name": "Recorded Session — Corner", "true_direction": "Right", "true_power": 9.0}
    ]
    # Filter for existing files
    valid = []
    for s in samples:
        path = os.path.join("data", "videos", f"{s['id']}.mp4")
        if os.path.exists(path):
            valid.append(s)
    return {"samples": valid}

@app.post("/api/predict-sample")
async def predict_sample(sample_id: str = Query(...)):
    clean_id = os.path.basename(sample_id).replace(".mp4", "")
    path = os.path.join("data", "videos", f"{clean_id}.mp4")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Sample video '{clean_id}' not found.")
    return JSONResponse(execute_prediction(path))

@app.api_route("/api/sample-video/{sample_id}", methods=["GET", "HEAD"])
async def get_sample_video(sample_id: str):
    clean_id = os.path.basename(sample_id).replace(".mp4", "")
    path = os.path.join("data", "videos", f"{clean_id}.mp4")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Sample video '{clean_id}' not found.")
    return FileResponse(path, media_type="video/mp4")

@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file format '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}")
        
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File too large (Max 100 MB).")
        
    tmp_path = None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp_path = tmp.name
        tmp.write(contents)
        tmp.close()
        
        result = execute_prediction(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
                
    return JSONResponse(result)

# Mount static web app
static_dir = os.path.join(os.path.dirname(__file__) or ".", "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "KickSense API Online. Open /docs for Swagger documentation."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False)
