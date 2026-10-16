import pandas as pd

new_data = {
    "video_id": [
        "penalty_64", "penalty_65", "penalty_66", "penalty_67",
        "penalty_68", "penalty_69", "penalty_70", "penalty_71",
        "penalty_72", "penalty_73"
    ],
    "match": [
        "Italy vs England", "Italy vs England", "Italy vs England", "Italy vs England",
        "Italy vs England", "Italy vs England", "Italy vs England", "Italy vs England",
        "Italy vs England", "Italy vs England"
    ],
    "player": [
        "Berardi", "Kane", "Belotti", "Maguire",
        "Bonucci", "Rashford", "Bernardeschi", "Sancho",
        "Jorginho", "Saka"
    ],
    "shot_direction": [
        "right", "right", "center", "right",
        "center", "left", "right", "left",
        "center", "right"
    ],
    "outcome": [
        "goal", "goal", "saved", "goal",
        "goal", "missed", "goal", "missed",
        "missed", "saved"
    ],
    "dominant_foot": [
        "right", "right", "right", "right",
        "right", "right", "right", "right",
        "right", "right"
    ],
    "tournament": [
        "Euro 2020", "Euro 2020", "Euro 2020", "Euro 2020",
        "Euro 2020", "Euro 2020", "Euro 2020", "Euro 2020",
        "Euro 2020", "Euro 2020"
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