import pandas as pd

new_data = {
    "video_id": [
        "penalty_113", "penalty_114", "penalty_115", "penalty_116",
        "penalty_117", "penalty_118", "penalty_119", "penalty_120",
        "penalty_121", "penalty_122", "penalty_123", "penalty_124", "penalty_125"
    ],
    "match": ["Manchester United vs Chelsea"] * 13,
    "player": [
        "Tevez", "Ballack", "Carrick", "Belletti",
        "Ronaldo", "Lampard", "Hargreaves", "A.Cole",
        "Nani", "Terry", "Anderson", "Kalou", "Anelka"
    ],
    "shot_direction": [
        "right", "left", "right", "left",
        "center", "right", "right", "left",
        "right", "right", "left", "right", "left"
    ],
    "outcome": [
        "goal", "goal", "goal", "goal",
        "missed", "goal", "goal", "goal",
        "goal", "missed", "goal", "goal", "saved"
    ],
    "dominant_foot": [
        "right", "right", "right", "right",
        "right", "right", "right", "left",
        "left", "right", "right", "right", "right"
    ],
    "tournament": ["Champions League 2008"] * 13
}

# Load existing labels
existing_df = pd.read_excel("data/labels/penalty_labels.xlsx")

# Drop ball tracking columns temporarily
existing_df = existing_df.drop(columns=['ball_placement', 'shot_power', 'ball_detections'], errors='ignore')

# Add new penalties
new_df = pd.DataFrame(new_data)
merged_df = pd.concat([existing_df, new_df], ignore_index=True)

merged_df.to_excel("data/labels/penalty_labels.xlsx", index=False)
print(f"Labels updated! Total penalties: {len(merged_df)}")