"""Generate a simpler Word version of the supervisor study guide."""
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

OUT = Path(__file__).resolve().parent / "KickSense_Simple_Study_Guide.docx"
NAVY = RGBColor(31, 56, 100)


def style_run(run, size=11, bold=False, color=None):
    run.font.size = Pt(size)
    run.font.name = "Calibri"
    run.bold = bold
    if color:
        run.font.color.rgb = color


def para(doc, text, bold=False, size=11):
    p = doc.add_paragraph()
    style_run(p.add_run(text), size=size, bold=bold)
    return p


def bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    style_run(p.add_run(text))
    return p


def add_table(doc, headers, rows):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(10)
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
            for p in cells[i].paragraphs:
                for run in p.runs:
                    run.font.size = Pt(10)
    doc.add_paragraph()


def build():
    doc = Document()

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    style_run(
        title.add_run("KickSense — Simple Study Guide for Tomorrow"),
        size=18,
        bold=True,
        color=NAVY,
    )

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    style_run(
        sub.add_run("Read this slowly. Every section answers a question your supervisor might ask."),
        bold=True,
    )

    for line in (
        "Team: Ashir Ijaz & Ibrahim Tariq  ·  Supervisor: Ms. Zainab Iftikhar",
        "App model file: outputs/model/best_model_improved.keras",
    ):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        style_run(p.add_run(line), size=10)

    doc.add_heading("What is KickSense? (say this first)", level=1)
    para(
        doc,
        "KickSense is a computer-vision project that looks at a short video of a football "
        "penalty kick and tries to answer two questions:",
    )
    bullet(doc, "Which way will the ball go? Left, Center, or Right")
    bullet(doc, "How hard was the shot? A number from 1 to 10")
    para(doc, "Important idea:", bold=True)
    para(
        doc,
        "We do not give the raw video pixels to a big image model like ResNet. "
        "Instead we find the player's body joints (skeleton), turn that motion into numbers, "
        "and then a neural network reads those numbers.",
    )

    doc.add_heading("Numbers you should remember", level=1)
    add_table(
        doc,
        ["Thing", "Number", "Meaning in simple words"],
        [
            ["Labeled videos", "460", "Master dataset (259 match clips + 201 self-recorded)"],
            ["Used for model", "426", "Clean normalized sequences after pose quality filtering"],
            ["Train / Val / Test", "298 / 64 / 64", "Stratified 70/15/15 split across shot directions"],
            ["Frames per clip (for model)", "26", "Every clip is standardized to 26 time steps"],
            ["Body points per frame", "33", "MediaPipe finds 33 joints"],
            ["Values per joint", "4", "x, y, z + confidence"],
            ["Features per frame", "132", "Because 33 × 4 = 132"],
            ["Model input shape", "(26, 132)", "26 moments × 132 numbers"],
            ["Direction accuracy", "53.1%", "Weighted Precision: 60.0% (vs 33.3% random guess)"],
            ["Power error (MAE)", "2.12", "Power error reduced to 2.12 on scale 1–10"],
        ],
    )
    para(doc, "Video sizes in our data are usually 640×360, 1280×720, or 4K (3840×2160).")

    doc.add_heading("1. Is our folder structure okay to show?", level=1)
    para(doc, "Yes — if you show the pipeline, not every random file.", bold=True)
    para(doc, "Think of the project like a factory line:")
    for item in (
        "data/videos/ → raw penalty videos (.mp4)",
        "data/frames/ → pictures cut from those videos",
        "data/keypoints/ → body-joint numbers from each clip",
        "data/labels/ → Excel labels (left/center/right, power, etc.)",
        "data/processed/ → final arrays ready for the model",
        "outputs/model/ → trained neural network",
        "outputs/evaluation/ → graphs like confusion matrix",
        "app.py → the demo website (Streamlit)",
    ):
        bullet(doc, item)
    para(doc, "What to say:", bold=True)
    para(
        doc,
        '"Our folders follow the steps of the pipeline. Old one-time scripts for cutting '
        "videos and updating labels are kept in scripts/legacy/ so they do not clutter "
        'the main project."',
    )

    doc.add_heading("2. How did we do this work? (story of the project)", level=1)
    para(doc, "Explain it in this order:")
    for item in (
        "We collected penalty kick videos (from matches and some recordings).",
        "We labeled each clip in Excel: direction of the shot, power, player, match, etc.",
        "We cut each video into frames (pictures).",
        "We ran MediaPipe Pose to find the player's skeleton on those frames.",
        "We cleaned and normalized the skeleton data, and made every clip the same length (26 frames).",
        "We trained a CNN-LSTM model to predict direction and power.",
        "We tested it and made graphs.",
        "We built a Streamlit app where you upload a video and see the prediction.",
    ):
        bullet(doc, item)

    doc.add_heading("What model are we using?", level=2)
    para(doc, "We use two parts working together:", bold=True)
    para(doc, "A) MediaPipe Pose (from Google)", bold=True)
    bullet(doc, "Finds 33 body landmarks on the player")
    bullet(doc, "We use this as a tool; we did not train MediaPipe ourselves")
    para(doc, "B) Our own CNN-LSTM network", bold=True)
    bullet(doc, "CNN (Conv1D): looks at short patterns in nearby frames")
    bullet(doc, "LSTM: remembers the longer motion from run-up to kick")
    bullet(doc, "Two outputs: Direction (Left/Center/Right) and Power (1 to 10)")
    para(
        doc,
        "The best saved model is best_model_improved.keras. There is also an older simpler "
        "version. Talk about the improved one.",
    )

    doc.add_heading("Models we did NOT use — and why", level=2)
    add_table(
        doc,
        ["Model", "Used?", "Simple reason"],
        [
            ["ResNet / EfficientNet", "No", "Those work on full images. Ours works on skeleton numbers."],
            ["Plain / vanilla RNN", "No", "LSTM is a better RNN that remembers longer."],
            ["Transformer", "No", "Needs more data; our set is still small."],
            ["ST-GCN", "No", "Good for skeletons, harder to build. Good next step."],
            ["3D CNN on video", "No", "Too heavy; needs thousands of videos."],
        ],
    )
    para(doc, "If she asks which models you compared:", bold=True)
    para(
        doc,
        '"We compared our basic CNN-LSTM with our improved CNN-LSTM. We did not train ResNet '
        'or ST-GCN yet. Papers suggest ST-GCN could work better later."',
    )

    doc.add_heading("Why CNN and LSTM together?", level=2)
    bullet(doc, "Only CNN → good at short patterns, weak at long time memory")
    bullet(doc, "Only LSTM → can work, but CNN first helps clean short movements")
    bullet(doc, "CNN + LSTM → short movements + full kick timeline")

    doc.add_heading('Why LSTM, not RNN?', level=2)
    para(
        doc,
        "LSTM is a type of RNN. A plain RNN has trouble remembering long sequences. "
        "LSTM has special gates so it can remember important parts of the run-up and forget noise. "
        "That is why we chose LSTM.",
    )

    doc.add_heading("Are there losses?", level=2)
    para(doc, "Yes. Loss = how wrong the model is during training.", bold=True)
    para(doc, "We have two losses because we predict two things:")
    bullet(doc, "Direction loss: Categorical Cross-Entropy (CE)")
    bullet(doc, "Power loss: Mean Squared Error (MSE)")
    para(doc, "total loss ≈ 1.0 × CE + 0.3 × MSE")

    doc.add_heading("How good are the results?", level=2)
    para(doc, "Say this honestly:")
    bullet(doc, "Direction accuracy is about 50%")
    bullet(doc, "Random guessing for 3 classes is about 33%")
    bullet(doc, "So the model is better than chance, but not strong enough for real use yet")
    bullet(doc, "Center shots are hardest (fewer Center examples)")
    bullet(doc, "Power labels are noisy, so power predictions are approximate")

    doc.add_heading("3. Dataset questions", level=1)
    para(doc, "Which dataset?", bold=True)
    para(
        doc,
        "Our own custom dataset. Not a famous public dataset like ImageNet or UCF101. "
        "Made from public match penalty clips + some local recordings. "
        "Labels are in data/labels/penalty_labels.xlsx.",
    )
    para(doc, "What does the model predict?", bold=True)
    bullet(doc, "Predicts: shot direction and shot power")
    bullet(doc, "Does NOT predict: goal/saved/missed, or exact goal-zone placement (not connected yet)")
    para(doc, "Video size / picture size?", bold=True)
    bullet(doc, "Videos are mostly 640×360 or 1280×720")
    bullet(doc, "The model does not see a picture size. It sees 26 × 132 skeleton numbers")
    para(doc, "How do we choose frames?", bold=True)
    for item in (
        "Take all frames from the video",
        "Run pose detection; if no body is detected, skip that frame",
        "From the good frames, pick 26 evenly spaced frames",
        "Flatten each frame's joints into 132 numbers",
    ):
        bullet(doc, item)
    para(doc, "Do we remove noise / outliers?", bold=True)
    para(doc, "Yes, partly. We remove a clip if:")
    bullet(doc, "fewer than 10 frames have a detected pose, or")
    bullet(doc, "less than 20% of frames have a pose, or")
    bullet(doc, "direction / power label is missing or invalid")
    para(doc, "That is how 259 becomes 225.")
    para(doc, "But noise still remains: weak poses in some clips, noisy ball-tracking power labels, fewer Center examples.")
    para(doc, "Do we do segmentation?", bold=True)
    para(
        doc,
        "No. Segmentation means cutting out every player pixel. We do pose estimation instead: "
        "find 33 body joints like a stick figure.",
    )
    para(
        doc,
        'Say: "We are not doing image segmentation. We are doing pose estimation. '
        'Pose gives us the skeleton, which is enough to study body movement for the kick."',
    )

    doc.add_heading("4. Full pipeline (how everything happens)", level=1)
    for item in (
        "1. Video (.mp4)",
        "2. extract_frames.py → many JPEG pictures",
        "3. pose_estimation.py → skeleton keypoints (.npy files)",
        "4. preprocessing.py → clean, normalize, make length 26, split data",
        "5. train_model_improved.py → train CNN-LSTM (losses are used here)",
        "6. evaluate_model.py → accuracy, confusion matrix, etc.",
        "7. app.py (Streamlit) → upload video and get prediction",
    ):
        bullet(doc, item)
    para(doc, "Hip normalization (simple meaning):", bold=True)
    para(
        doc,
        "Players can be close or far from the camera. We move coordinates relative to the hips "
        "and scale by torso height so near and far players look more similar to the model.",
    )
    para(doc, "Where is the loss used?", bold=True)
    para(
        doc,
        "Only in training. Frame extraction, pose, and preprocessing do not use loss. "
        "The app only does prediction.",
    )

    doc.add_heading("5. What to show tomorrow (order)", level=1)
    for item in (
        "Open README.md and explain the folders briefly",
        "Open the Excel labels and show direction / power columns",
        "Show one pose result / overlay if available under outputs/",
        "Explain: Input → CNN → LSTM → Direction + Power",
        "Show outputs/evaluation/confusion_matrix.png",
        "Run the Streamlit app and upload a clear clip",
        "End with limitations and next steps",
    ):
        bullet(doc, item)
    para(doc, "Best demo videos:", bold=True)
    bullet(doc, "Best: data/videos/penalty_266.mp4")
    bullet(doc, "Also good: penalty_247.mp4, penalty_281.mp4")
    para(doc, "Do NOT say:", bold=True)
    bullet(doc, "Accuracy is 90%+")
    bullet(doc, "We do segmentation")
    bullet(doc, "We already compared ResNet and Transformer experimentally")

    doc.add_heading("Fast answers if she traps you", level=1)
    add_table(
        doc,
        ["Question", "Short answer"],
        [
            ["Why not ResNet?", "Because we use skeleton sequences, not raw images."],
            ["Why LSTM not RNN?", "LSTM is a better RNN that remembers longer motion."],
            ["Is there a loss?", "Yes — CE for direction, MSE for power."],
            ["Segmentation?", "No. Pose estimation instead."],
            ["Why only ~50%?", "Small/noisy data, hard task, imbalance; still above 33% random."],
            ["What's next?", "More clips, cleaner labels, try ST-GCN."],
        ],
    )

    doc.add_heading("One-line summary to end with", level=1)
    para(
        doc,
        "KickSense turns a penalty video into a skeleton sequence and uses a CNN-LSTM to "
        "predict shot direction and power. The system works end-to-end with a demo app; "
        "accuracy is above random but still needs more data and better labels to improve.",
        bold=True,
    )

    doc.save(str(OUT))
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    build()
