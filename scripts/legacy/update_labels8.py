import pandas as pd

new_data = {
    "video_id": [
        "penalty_74", "penalty_75", "penalty_76", "penalty_77",
        "penalty_78", "penalty_79", "penalty_80", "penalty_81", "penalty_82"
    ],
    "match": [
        "Nigeria vs Morocco", "Nigeria vs Morocco", "Nigeria vs Morocco",
        "Nigeria vs Morocco", "Nigeria vs Morocco", "Nigeria vs Morocco",
        "Nigeria vs Morocco", "Nigeria vs Morocco", "Nigeria vs Morocco"
    ],
    "player": [
        "Player1", "Player2", "Player3", "Player4",
        "Player5", "Player6", "Player7", "Player8", "Player9"
    ],
    "shot_direction": [
        "right", "left", "right", "center",
        "left", "right", "left", "right", "center"
    ],
    "outcome": [
        "goal", "goal", "goal", "goal",
        "goal", "goal", "goal", "goal", "goal"
    ],
    "dominant_foot": [
        "right", "right", "right", "right",
        "right", "right", "right", "right", "right"
    ],
    "tournament": [
        "AFCON 2025", "AFCON 2025", "AFCON 2025", "AFCON 2025",
        "AFCON 2025", "AFCON 2025", "AFCON 2025", "AFCON 2025", "AFCON 2025"
    ]
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
print(merged_df[['video_id', 'match', 'outcome']])