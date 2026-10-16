# KickSense — Simple Study Guide for Tomorrow

Read this slowly. Every section answers a question your supervisor might ask.

**Who we are:** Ashir Ijaz & Ibrahim Tariq  
**Supervisor:** Ms. Zainab Iftikhar  
**App model file:** `outputs/model/best_model_improved.keras`

---

## What is KickSense? (say this first)

KickSense is a computer-vision project that looks at a short video of a football penalty kick and tries to answer two questions:

1. **Which way will the ball go?** Left, Center, or Right  
2. **How hard was the shot?** A number from 1 to 10  

**Important idea:**  
We do **not** give the raw video pixels to a big image model like ResNet.  
Instead we find the **player’s body joints** (skeleton), turn that motion into numbers, and then a neural network reads those numbers.

---

## Numbers you should remember

| Thing | Number | Meaning in simple words |
|-------|--------|-------------------------|
| Labeled videos | **460** | Master dataset in Excel (259 match clips + 201 newly recorded clips) |
| Used for training & evaluation | **426** | Clean normalized sequences after pose quality filtering |
| Train / Val / Test | **298 / 64 / 64** | Stratified 70/15/15 split across all 3 shot directions |
| Frames per clip (for model) | **26** | Every clip is standardized to 26 time steps |
| Body points per frame | **33** | MediaPipe finds 33 joints (shoulders, hips, knees, etc.) |
| Values per joint | **4** | x, y, z position + visibility confidence |
| Features per frame | **132** | Because 33 × 4 = 132 |
| Model input shape | **(26, 132)** | 26 moments in time, each with 132 numbers |
| Direction accuracy | **53.1%** | (Weighted Precision: **60.0%**, Center Precision: **78.6%** vs 33.3% random guess) |
| Power error (MAE) | **2.12** | On scale 1–10 (improved from >3.5 down to 2.12) |

Video sizes in our data are usually **640×360**, **1280×720**, or **4K (3840×2160)**.

---

## 1. Is our folder structure okay to show?

**Yes — if you show the pipeline, not every random file.**

Think of the project like a factory line:

```
data/videos/      → raw penalty videos (.mp4)
data/frames/      → pictures cut from those videos
data/keypoints/  → body-joint numbers from each clip
data/labels/      → Excel labels (left/center/right, power, etc.)
data/processed/  → final arrays ready for the model
outputs/model/   → trained neural network
outputs/evaluation/ → graphs like confusion matrix
app.py           → the demo website (Streamlit)
```

**What to say:**  
“Our folders follow the steps of the pipeline. Old one-time scripts for cutting videos and updating labels are kept in `scripts/legacy/` so they do not clutter the main project.”

---

## 2. How did we do this work? (story of the project)

Explain it in this order:

1. We collected penalty kick videos (from matches and some recordings).  
2. We labeled each clip in Excel: direction of the shot, power, player, match, etc.  
3. We cut each video into frames (pictures).  
4. We ran **MediaPipe Pose** to find the player’s skeleton on those frames.  
5. We cleaned and normalized the skeleton data, and made every clip the same length (**26 frames**).  
6. We trained a **CNN-LSTM** model to predict direction and power.  
7. We tested it and made graphs.  
8. We built a **Streamlit app** where you upload a video and see the prediction.

### What model are we using?

We use **two parts working together**:

**A) MediaPipe Pose (from Google)**  
- Finds **33 body landmarks** on the player  
- We use this as a tool; we did not train MediaPipe ourselves  

**B) Our own CNN-LSTM network**  
- **CNN (Conv1D):** looks at short patterns in nearby frames (example: hip turning, foot planting)  
- **LSTM:** remembers the longer motion from run-up to kick  
- **Two outputs:**  
  - Direction → Left / Center / Right  
  - Power → 1 to 10  

The best saved model is called **`best_model_improved.keras`**.

There is also an older simpler version (`train_model.py`). The improved one has extra Dropout, BatchNorm, data augmentation, and better training settings. **Use / talk about the improved one.**

### Models we did NOT use — and why

| Model people might ask about | Did we use it? | Simple reason |
|------------------------------|----------------|---------------|
| ResNet / EfficientNet | No | Those work on full images. Our model works on skeleton numbers. |
| Plain / vanilla RNN | No | LSTM is already a better type of RNN. Plain RNN forgets early parts of the run-up. |
| Transformer | No | Needs more data; our set is still small. Future idea. |
| ST-GCN (graph network for skeletons) | No | Very good idea for skeletons, but harder to build. Good next step. |
| 3D CNN on video | No | Too heavy and needs thousands of videos. |

**If she asks “which models did you compare?”**  
Be honest:  
“We compared our **basic CNN-LSTM** with our **improved CNN-LSTM**. We did not train ResNet or ST-GCN yet. Papers suggest ST-GCN could work better later.”

### Why CNN and LSTM together?

- Only CNN → good at short patterns, weak at long time memory  
- Only LSTM → can work, but CNN first helps clean short movements  
- CNN + LSTM → short movements + full kick timeline  

### “Why LSTM, not RNN?”

**LSTM is a type of RNN.**  
A plain RNN has trouble remembering long sequences (vanishing gradient).  
LSTM has special “gates” so it can remember important parts of the run-up and forget noise.  
That is why we chose LSTM.

### Are there losses?

**Yes. Loss = how wrong the model is during training.**

We have **two losses** because we predict two things:

1. **Direction loss:** Categorical Cross-Entropy (CE)  
2. **Power loss:** Mean Squared Error (MSE)  

We care more about direction, so weights are roughly:

`total loss ≈ 1.0 × CE + 0.3 × MSE`

### How good are the results?

Say this honestly:

- Direction accuracy is about **50%**  
- Random guessing for 3 classes is about **33%**  
- So the model is **better than chance**, but **not strong enough for real use yet**  
- **Center** shots are hardest (we have fewer Center examples)  
- Power labels are noisy, so power predictions are approximate  

---

## 3. Dataset questions

### Which dataset?

Our **own custom dataset**.  
It is **not** a famous public dataset like ImageNet or UCF101.  
We made it from public match penalty clips + some local recordings.  
Labels are in: `data/labels/penalty_labels.xlsx`

### What does the model predict?

**Predicts:**
- shot direction (left / center / right)  
- shot power (1–10)  

**Does NOT predict (even if the Excel has it):**
- goal / saved / missed  
- exact goal-zone placement (top-left, etc.) — we prepared that column but did not connect it to the model yet  

### Video size / picture size?

- Videos are mostly **640×360** or **1280×720**  
- We do **not** resize everything to something like 224×224 for a ResNet  
- The model does not “see” a picture size. It sees **26 × 132 numbers** from the skeleton  

### How do we choose frames?

1. Take **all** frames from the video  
2. Run pose detection; if a frame has no body detected, skip it  
3. From the good frames, pick **26 evenly spaced** frames  
4. Flatten each frame’s joints into 132 numbers  

So: we use the whole kick timeline, compressed into 26 steps.

### Do we remove noise / outliers?

**Yes, partly.**

We remove a clip if:
- fewer than **10** frames have a detected pose, or  
- less than **20%** of frames have a pose, or  
- direction / power label is missing or invalid  

That is how **259 → 225**.

**But noise still remains:**
- some clips still have weak pose detection  
- power was estimated by tracking the ball with OpenCV Hough Circles (not perfect)  
- Center class has fewer examples  

### Do we do segmentation?

**No.**

Segmentation means coloring / cutting out every player pixel (like a mask).  
We do **not** do that.

We do **pose estimation**: find 33 body joints and connect them like a stick figure.

**Say this:**  
“We are not doing image segmentation. We are doing pose estimation. Pose gives us the skeleton, which is enough to study body movement for the kick.”

---

## 4. Full pipeline (how everything happens)

```
1. Video (.mp4)
2. extract_frames.py        → many JPEG pictures
3. pose_estimation.py       → skeleton keypoints (.npy files)
4. preprocessing.py         → clean, normalize, make length 26, split data
5. train_model_improved.py  → train CNN-LSTM (here the losses are used)
6. evaluate_model.py        → accuracy, confusion matrix, etc.
7. app.py (Streamlit)       → upload video and get prediction
```

### Hip normalization (simple meaning)

Players can be close or far from the camera.  
So we move coordinates relative to the **hips**, and scale by **torso height**.  
That makes a near player and a far player look more similar to the model.

### Where is the loss used?

Only in **training**.  
Frame extraction, pose, and preprocessing do not use loss.  
The app only does **prediction** (no training loss).

---

## 5. What to show tomorrow (order)

1. Open `README.md` and explain the folders briefly  
2. Open the Excel labels and show direction / power columns  
3. Show one pose result / overlay if available under `outputs/`  
4. Draw or explain: Input → CNN → LSTM → Direction + Power  
5. Show `outputs/evaluation/confusion_matrix.png`  
6. Run the Streamlit app and upload a clear clip  
7. End with limitations and next steps (more data, maybe ST-GCN, better power labels)

### Best demo videos

Use these (clear body, good pose detection):

- **Best:** `data/videos/penalty_266.mp4`  
- Also good: `penalty_247.mp4`, `penalty_281.mp4`  

### Do NOT say

- “Accuracy is 90%+”  
- “We do segmentation”  
- “We already compared ResNet and Transformer experimentally”  

---

## Fast answers if she traps you

| Question | Short answer |
|----------|--------------|
| Why not ResNet? | Because we use skeleton sequences, not raw images. |
| Why LSTM not RNN? | LSTM is a better RNN that remembers longer motion. |
| Is there a loss? | Yes — CE for direction, MSE for power. |
| Segmentation? | No. Pose estimation instead. |
| Why only ~50%? | Small/noisy data, hard task, class imbalance; still above 33% random. |
| What’s next? | More clips, cleaner labels, try ST-GCN. |

---

## One-line summary to end with

“KickSense turns a penalty video into a skeleton sequence and uses a CNN-LSTM to predict shot direction and power. The system works end-to-end with a demo app; accuracy is above random but still needs more data and better labels to improve.”
