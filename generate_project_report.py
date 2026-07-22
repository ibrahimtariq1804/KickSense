"""
Generate KickSense Project Progress Report (Word Document)
"""
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from datetime import datetime
import os

OUTPUT_PATH = "KickSense_Project_Report.docx"


def set_heading_style(paragraph, level=1):
    if level == 1:
        paragraph.runs[0].font.size = Pt(16)
        paragraph.runs[0].font.bold = True
        paragraph.runs[0].font.color.rgb = RGBColor(31, 56, 100)
    elif level == 2:
        paragraph.runs[0].font.size = Pt(13)
        paragraph.runs[0].font.bold = True
        paragraph.runs[0].font.color.rgb = RGBColor(46, 117, 182)


def add_bullet(doc, text, bold_prefix=None):
    p = doc.add_paragraph(style="List Bullet")
    if bold_prefix:
        run = p.add_run(bold_prefix)
        run.bold = True
        p.add_run(text)
    else:
        p.add_run(text)


def add_table(doc, headers, rows):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for paragraph in hdr_cells[i].paragraphs:
            for run in paragraph.runs:
                run.bold = True
    for row in rows:
        row_cells = table.add_row().cells
        for i, val in enumerate(row):
            row_cells[i].text = str(val)
    doc.add_paragraph()


def build_report():
    doc = Document()

    # Title page
    title = doc.add_heading("KickSense", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = doc.add_paragraph("Intelligent Penalty Kick Analysis Using Pose Estimation")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.runs[0].font.size = Pt(14)
    subtitle.runs[0].italic = True

    doc.add_paragraph()
    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    info.add_run("Final Year Project Report — Progress Summary\n").bold = True
    info.add_run("Bahria University Islamabad\n")
    info.add_run("Department of Computer Science\n")
    info.add_run("Session 2026–2027\n\n")
    info.add_run("Team: Ashir Ijaz (01-135231-017) & Ibrahim Tariq (01-135231-037)\n")
    info.add_run("Supervisor: Ms. Zainab Iftikhar\n\n")
    info.add_run(f"Report Generated: {datetime.now().strftime('%B %d, %Y')}")

    doc.add_page_break()

    # 1. Executive Summary
    doc.add_heading("1. Executive Summary", level=1)
    doc.add_paragraph(
        "KickSense is a machine learning–based system that analyzes football penalty kicks "
        "using computer vision and deep learning. The project extracts body pose keypoints "
        "from penalty kick videos using MediaPipe, processes them into temporal sequences, "
        "and trains a CNN-LSTM neural network to predict shot direction (Left/Center/Right) "
        "and shot power (scale 1–10). A Streamlit web application allows users to upload "
        "penalty videos and receive real-time predictions with pose visualization."
    )
    doc.add_paragraph(
        "As of this report, the project has collected and labeled 198 penalty clips from "
        "major tournaments (World Cups, FA Cup, Premier League, etc.), built a complete "
        "data processing pipeline, trained baseline and improved models, evaluated performance, "
        "and deployed a functional demo web app."
    )

    # 2. Project Objectives
    doc.add_heading("2. Project Objectives", level=1)
    objectives = [
        "Collect penalty kick video clips from professional football matches",
        "Extract and annotate labels (shot direction, outcome, player, tournament, etc.)",
        "Use MediaPipe Pose Estimation to extract 33 body keypoints per frame",
        "Preprocess pose sequences into normalized 26-frame feature vectors",
        "Train a CNN-LSTM deep learning model for multi-task prediction",
        "Predict shot direction (classification) and shot power (regression)",
        "Build a user-friendly Streamlit web interface for live video analysis",
        "Evaluate model performance with standard ML metrics and visualizations",
    ]
    for obj in objectives:
        add_bullet(doc, obj)

    # 3. System Architecture
    doc.add_heading("3. System Architecture & Pipeline", level=1)
    doc.add_paragraph(
        "The project follows a standard ML pipeline from raw video to deployed application:"
    )
    pipeline_steps = [
        ("Video Collection", "Penalty clips sourced from YouTube match footage using FFmpeg cutting scripts"),
        ("Frame Extraction", "OpenCV extracts all frames from each clip into data/frames/"),
        ("Pose Estimation", "MediaPipe extracts 33 skeletal keypoints (x, y, z, visibility) per frame"),
        ("Ball Tracking", "Hough Circle Transform detects ball position for placement and power estimation"),
        ("Label Management", "Excel-based label database (penalty_labels.xlsx) with match metadata"),
        ("Preprocessing", "Hip-centered normalization, 26-frame sequencing, train/val/test split"),
        ("Model Training", "CNN-LSTM with dual output heads (direction + power)"),
        ("Evaluation", "Confusion matrix, classification report, regression metrics"),
        ("Web App", "Streamlit app (app.py) for upload, analysis, and visualization"),
    ]
    add_table(doc, ["Stage", "Description"], pipeline_steps)

    # 4. Work Completed
    doc.add_heading("4. Work Completed to Date", level=1)

    # 4.1 Data Collection
    doc.add_heading("4.1 Data Collection & Annotation", level=2)
    doc.add_paragraph(
        "A substantial penalty kick dataset has been assembled from multiple sources:"
    )
    add_table(doc, ["Metric", "Value"], [
        ["Total penalty video clips", "210 videos in data/videos/"],
        ["Processed frame folders", "198 penalty folders in data/frames/"],
        ["Pose keypoint extractions", "198 folders in data/keypoints/"],
        ["Labeled entries", "198 rows in penalty_labels.xlsx"],
        ["Shot direction — Right", "88 clips"],
        ["Shot direction — Left", "71 clips"],
        ["Shot direction — Center", "39 clips"],
        ["Outcomes — Goal", "165"],
        ["Outcomes — Saved", "21"],
        ["Outcomes — Missed", "12"],
    ])
    doc.add_paragraph("Tournaments and matches covered include:")
    tournaments = [
        "FIFA World Cup 2022 (Argentina vs France penalty shootout — initial 8 clips)",
        "Women's World Cup 2023 (Australia vs France)",
        "World Cup 2018, 2014, 2010 (various knockout matches)",
        "Premier League 2023–24",
        "FA Cup, FA Community Shield, EFL Cup finals",
        "Morocco vs Spain, Brazil vs Croatia, and other dramatic shootouts",
    ]
    for t in tournaments:
        add_bullet(doc, t)

    doc.add_paragraph(
        "Label fields: video_id, match, player, shot_direction, outcome, dominant_foot, "
        "tournament, ball_placement, shot_power, ball_detections."
    )

    # 4.2 Video Processing Scripts
    doc.add_heading("4.2 Video Processing Scripts", level=2)
    scripts_video = [
        ("cut_clips.py – cut_clips12.py", "FFmpeg-based scripts to cut individual penalty moments from full match videos with timestamp ranges"),
        ("extract_frames.py", "Extracts every frame from penalty clips using OpenCV into per-clip folders"),
        ("import_new_clips.py", "Bulk imports new clips from download folders with automatic penalty numbering and corruption filtering"),
        ("ball_tracker.py", "Detects football using Hough Circle Transform; estimates ball placement zones and shot power from ball velocity"),
    ]
    add_table(doc, ["Script", "Purpose"], scripts_video)

    # 4.3 Label Management
    doc.add_heading("4.3 Label Management", level=2)
    doc.add_paragraph(
        "Multiple iterative scripts (create_labels.py, update_labels.py through update_labels11.py, "
        "update_labels_new.py) were developed to incrementally add penalty labels as new clips "
        "were collected. The latest script (update_labels_new.py) added 73 new penalties "
        "(penalty_126 through penalty_200) from recent tournament and cup footage."
    )

    # 4.4 Pose Estimation
    doc.add_heading("4.4 Pose Estimation", level=2)
    doc.add_paragraph(
        "pose_estimation.py uses MediaPipe Pose (model_complexity=2, min_detection_confidence=0.5) "
        "to process extracted frames. For each penalty clip it:"
    )
    pose_steps = [
        "Detects 33 body landmarks per frame",
        "Draws skeletal overlay on frames",
        "Saves annotated pose images to outputs/penalty_XX/",
        "Stores keypoint arrays as keypoints.npy (shape: frames × 33 × 4)",
        "Skips already-processed clips for incremental runs",
    ]
    for s in pose_steps:
        add_bullet(doc, s)

    # 4.5 Preprocessing
    doc.add_heading("4.5 Data Preprocessing", level=2)
    doc.add_paragraph("preprocessing.py implements a 7-step pipeline:")
    preprocess_steps = [
        "Coordinate normalization — hip-centered, torso-height scaled",
        "Sequence structuring — resample to exactly 26 frames per clip",
        "Feature flattening — 33 keypoints × 4 values = 132 features per frame",
        "Label encoding — direction (0/1/2), power (1–10), placement (5 classes)",
        "Outlier removal — discard clips with missing/invalid labels",
        "Stratified split — 70% train / 15% validation / 15% test",
        "Save processed .npy arrays to data/processed/",
    ]
    for s in preprocess_steps:
        add_bullet(doc, s)

    # 4.6 Model Development
    doc.add_heading("4.6 Model Development", level=2)
    doc.add_paragraph("Two model versions were developed:")
    doc.add_paragraph("Baseline Model (train_model.py):", style="List Bullet")
    baseline = [
        "Architecture: Conv1D(64) → Conv1D(128) → MaxPool → LSTM(128) → LSTM(64) → Dense(64)",
        "Dual outputs: direction (softmax, 3 classes) + power (sigmoid, regression)",
        "Optimizer: Adam (lr=0.001), 100 epochs, early stopping (patience=15)",
        "Saved as: outputs/model/best_model.keras",
    ]
    for b in baseline:
        add_bullet(doc, b)

    doc.add_paragraph("Improved Model (train_model_improved.py):", style="List Bullet")
    improved = [
        "Enhanced architecture with additional BatchNormalization and Dropout layers",
        "Data augmentation: noise injection, scaling, horizontal mirroring (with label flip), time shifting",
        "Class-weight balancing for imbalanced direction classes",
        "ReduceLROnPlateau callback for adaptive learning rate",
        "150 epochs, early stopping (patience=20)",
        "Saved as: outputs/model/best_model_improved.keras (used in web app)",
    ]
    for i in improved:
        add_bullet(doc, i)

    doc.add_paragraph("Model input shape: (26, 132) — 26 time steps × 132 pose features")

    # 4.7 Evaluation
    doc.add_heading("4.7 Model Evaluation", level=2)
    doc.add_paragraph(
        "evaluate_model.py provides comprehensive evaluation on the held-out test set:"
    )
    eval_items = [
        "Direction classification: Accuracy, Precision, Recall, F1-Score",
        "Per-class metrics for Left, Center, Right",
        "Power regression: MAE, MSE, RMSE on 1–10 scale",
        "Confusion matrix heatmap (outputs/evaluation/confusion_matrix.png)",
        "Actual vs Predicted power scatter plot",
        "Per-class metrics bar chart",
        "Classification report with sklearn",
    ]
    for e in eval_items:
        add_bullet(doc, e)
    doc.add_paragraph(
        "Current reported direction accuracy: approximately 50% (as noted in the Streamlit app sidebar). "
        "This is above the random baseline of 33.3% for 3-class classification but indicates room for improvement."
    )

    # 4.8 EDA & Visualization
    doc.add_heading("4.8 Exploratory Data Analysis & Visualization", level=2)
    viz_items = [
        ("EDA.ipynb", "Jupyter notebook for initial exploratory analysis"),
        ("eda_analysis.py", "Generates shot direction, outcome, direction-vs-outcome, and ball placement charts"),
        ("data_visualization.py", "Comprehensive dataset dashboard with tournament breakdown, power distribution, and placement heatmaps"),
        ("visualize_single_penalty.py", "Visualizes pose keypoints and predictions for individual penalty clips"),
    ]
    add_table(doc, ["File", "Purpose"], viz_items)
    doc.add_paragraph("Output charts saved to outputs/ and outputs/visualizations/")

    # 4.9 Web Application
    doc.add_heading("4.9 Streamlit Web Application", level=2)
    doc.add_paragraph(
        "app.py delivers a polished, production-ready demo interface with:"
    )
    app_features = [
        "Custom CSS styling with gradient header and metric cards",
        "Video upload support (.mp4, .avi, .mov)",
        "Real-time frame extraction (26 evenly spaced frames)",
        "MediaPipe pose estimation with skeletal overlay visualization",
        "CNN-LSTM inference using best_model_improved.keras",
        "Shot direction prediction with confidence percentages",
        "Shot power prediction on 1–10 scale with color-coded labels",
        "Direction probability breakdown with progress bars",
        "Sidebar with project info, model stats, and team credits",
        "Run command: streamlit run app.py",
    ]
    for f in app_features:
        add_bullet(doc, f)

    # 4.10 Project File Inventory
    doc.add_heading("4.10 Project File Inventory", level=2)
    add_table(doc, ["Category", "Files"], [
        ["Core Pipeline", "extract_frames.py, pose_estimation.py, preprocessing.py, ball_tracker.py"],
        ["Model", "train_model.py, train_model_improved.py, evaluate_model.py"],
        ["Data Tools", "create_labels.py, update_labels*.py, import_new_clips.py"],
        ["Video Cutting", "cut_clips.py through cut_clips12.py"],
        ["Analysis", "eda_analysis.py, data_visualization.py, visualize_single_penalty.py, EDA.ipynb"],
        ["Application", "app.py"],
        ["Data", "data/videos/, data/frames/, data/keypoints/, data/labels/, data/processed/"],
        ["Outputs", "outputs/model/, outputs/evaluation/, outputs/visualizations/, pose images"],
    ])

    # 5. Technology Stack
    doc.add_heading("5. Technology Stack", level=1)
    add_table(doc, ["Technology", "Usage"], [
        ["Python 3", "Primary programming language"],
        ["OpenCV (cv2)", "Video I/O, frame extraction, image processing"],
        ["MediaPipe", "33-point human pose estimation"],
        ["TensorFlow / Keras", "CNN-LSTM model building and training"],
        ["NumPy / Pandas", "Numerical data processing and label management"],
        ["scikit-learn", "Train/test split, metrics, class weights"],
        ["Matplotlib / Seaborn", "Charts, confusion matrices, EDA plots"],
        ["Streamlit", "Web application framework"],
        ["FFmpeg", "Video clip cutting from full match recordings"],
        ["Excel (openpyxl)", "Label database storage"],
    ])

    # 6. Current Limitations
    doc.add_heading("6. Current Limitations", level=1)
    limitations = [
        "Model accuracy (~50%) is modest — significantly above random chance but not yet production-grade",
        "Shot power labels derived from ball tracking (Hough circles) are noisy and approximate",
        "Ball placement prediction head was prepared in preprocessing but not integrated into the final model",
        "Pose estimation uses static_image_mode (per-frame) rather than video-optimized tracking",
        "Class imbalance: Center shots (39) are underrepresented vs Right (88) and Left (71)",
        "Many cut_clips and update_labels scripts are versioned iterations rather than a unified pipeline",
        "No requirements.txt or README documentation for reproducibility",
        "No automated testing or CI/CD pipeline",
        "Web app processes uploaded videos frame-by-frame without GPU batch optimization",
        "Player names in newer labels use placeholders (Player1, Player2, etc.)",
    ]
    for lim in limitations:
        add_bullet(doc, lim)

    # 7. Future Upgrades
    doc.add_heading("7. Recommended Future Upgrades", level=1)

    doc.add_heading("7.1 Model & Accuracy Improvements", level=2)
    upgrades_model = [
        "Collect more data — target 500+ penalty clips with balanced class distribution",
        "Add ball placement as a third prediction head (5-class classification)",
        "Predict penalty outcome (goal/saved/missed) as an additional output",
        "Try Transformer-based architectures (e.g., Temporal Fusion Transformer, ST-GCN)",
        "Use MediaPipe video mode or MoveNet for smoother temporal pose tracking",
        "Incorporate goalkeeper pose as additional input features",
        "Apply transfer learning from sports action recognition datasets (e.g., Kinetics, Sports-1M)",
        "Use cross-validation instead of single train/val/test split for more robust evaluation",
        "Hyperparameter tuning with Optuna or Keras Tuner",
        "Ensemble multiple models for improved direction prediction",
    ]
    for u in upgrades_model:
        add_bullet(doc, u)

    doc.add_heading("7.2 Data & Labeling Improvements", level=2)
    upgrades_data = [
        "Replace placeholder player names with actual player identities",
        "Manually verify and correct ball_placement and shot_power labels",
        "Add run-up speed, body angle, and planting foot as engineered features",
        "Include goalkeeper dive direction as a label for comparative analysis",
        "Build a labeling GUI (e.g., Label Studio or custom Streamlit tool)",
        "Standardize video quality and camera angles across clips",
        "Add data from different leagues and playing surfaces for generalization",
    ]
    for u in upgrades_data:
        add_bullet(doc, u)

    doc.add_heading("7.3 Application & Deployment", level=2)
    upgrades_app = [
        "Deploy web app to cloud (Streamlit Cloud, Hugging Face Spaces, or AWS)",
        "Add real-time webcam/video stream analysis",
        "Show frame-by-frame pose animation (not just single best frame)",
        "Add comparison mode: compare two penalty kicks side by side",
        "Export analysis report as PDF for coaches/analysts",
        "Mobile-friendly responsive design",
        "Add user authentication and saved analysis history",
        "Integrate with football analytics APIs (StatsBomb, Wyscout)",
    ]
    for u in upgrades_app:
        add_bullet(doc, u)

    doc.add_heading("7.4 Engineering & Code Quality", level=2)
    upgrades_eng = [
        "Consolidate cut_clips*.py and update_labels*.py into single parameterized scripts",
        "Create requirements.txt with pinned dependency versions",
        "Write a comprehensive README with setup and usage instructions",
        "Add a config.yaml for paths, hyperparameters, and model settings",
        "Implement a unified CLI (e.g., python kicksense.py --stage preprocess)",
        "Add unit tests for preprocessing and normalization functions",
        "Set up Git version control and GitHub repository",
        "Dockerize the application for reproducible deployment",
        "Add logging throughout the pipeline",
        "Create Jupyter notebook walkthrough for thesis documentation",
    ]
    for u in upgrades_eng:
        add_bullet(doc, u)

    doc.add_heading("7.5 Research & Thesis Extensions", level=2)
    upgrades_thesis = [
        "Compare pose-only vs pose+ball features ablation study",
        "Analyze which body joints contribute most to direction prediction (SHAP/LIME)",
        "Study correlation between run-up biomechanics and shot outcome",
        "Benchmark against existing penalty analysis research papers",
        "Publish dataset and model as open-source contribution",
        "Present goalkeeper strategy recommendations based on predicted shot direction",
        "Real-time penalty shootout simulator for training goalkeepers",
    ]
    for u in upgrades_thesis:
        add_bullet(doc, u)

    # 8. Project Timeline Summary
    doc.add_heading("8. Development Timeline", level=1)
    add_table(doc, ["Period", "Milestone"], [
        ["April 2026", "Project setup, initial 8 Argentina vs France clips, label creation, EDA notebook"],
        ["Late April 2026", "Frame extraction, pose estimation pipeline, ball tracking, label updates v2–v4"],
        ["April–May 2026", "Massive data expansion (100+ clips), cut_clips scripts v5–v12, label updates v5–v11"],
        ["Early May 2026", "Preprocessing pipeline, baseline CNN-LSTM training, EDA and visualization scripts"],
        ["Mid May 2026", "Improved model with augmentation, model evaluation, Streamlit web app deployment"],
        ["Late May 2026", "73 new penalty labels (penalty_126–200), import_new_clips, pose estimation updates"],
    ])

    # 9. How to Run
    doc.add_heading("9. How to Run the Project", level=1)
    doc.add_paragraph("Prerequisites: Python 3.x with virtual environment (kicksense_env) activated.")
    run_steps = [
        ("Full pipeline:", [
            "python extract_frames.py          # Extract frames from videos",
            "python pose_estimation.py         # Extract pose keypoints",
            "python preprocessing.py           # Preprocess and split data",
            "python train_model_improved.py    # Train improved model",
            "python evaluate_model.py          # Evaluate on test set",
        ]),
        ("Web application:", [
            "streamlit run app.py              # Launch web interface",
        ]),
        ("Analysis:", [
            "python eda_analysis.py            # Generate EDA charts",
            "python data_visualization.py      # Generate dataset dashboard",
        ]),
    ]
    for section, commands in run_steps:
        p = doc.add_paragraph()
        p.add_run(section).bold = True
        for cmd in commands:
            doc.add_paragraph(cmd, style="List Bullet")

    # 10. Conclusion
    doc.add_heading("10. Conclusion", level=1)
    doc.add_paragraph(
        "KickSense has progressed from a concept to a functional end-to-end system encompassing "
        "data collection (198+ labeled penalties), computer vision (pose estimation and ball tracking), "
        "deep learning (CNN-LSTM multi-task model), evaluation (comprehensive metrics and visualizations), "
        "and deployment (Streamlit web app). The project demonstrates a complete machine learning pipeline "
        "applied to sports analytics."
    )
    doc.add_paragraph(
        "The primary area for improvement is model accuracy, which can be addressed through more balanced "
        "data, better pose tracking, feature engineering, and advanced architectures. Engineering improvements "
        "such as code consolidation, documentation, and cloud deployment would make the project more "
        "professional and presentation-ready for the final year project defense."
    )

    doc.add_paragraph()
    closing = doc.add_paragraph()
    closing.alignment = WD_ALIGN_PARAGRAPH.CENTER
    closing.add_run("— End of Report —").italic = True

    doc.save(OUTPUT_PATH)
    return OUTPUT_PATH


if __name__ == "__main__":
    path = build_report()
    full_path = os.path.abspath(path)
    print(f"Report saved to: {full_path}")
