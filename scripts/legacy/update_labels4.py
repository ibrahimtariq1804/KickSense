import pandas as pd

new_data = {
    "video_id": [
        "penalty_33", "penalty_34", "penalty_35", "penalty_36",
        "penalty_37", "penalty_38", "penalty_39", "penalty_40"
    ],
    "match": [
        "Japan vs Croatia", "Japan vs Croatia", "Japan vs Croatia", "Japan vs Croatia",
        "Japan vs Croatia", "Japan vs Croatia", "Japan vs Croatia", "Japan vs Croatia"
    ],
    "player": [
        "Modric", "Doan", "Livakovic", "Asano",
        "Majer", "Mitoma", "Orsic", "Maeda"
    ],
    "shot_direction": [
        "left", "right", "center", "left",
        "right", "left", "right", "center"
    ],
    "outcome": [
        "goal", "saved", "goal", "goal",
        "goal", "saved", "goal", "missed"
    ],
    "dominant_foot": [
        "left", "right", "right", "right",
        "right", "right", "right", "right"
    ],
    "tournament": [
        "World Cup 2022", "World Cup 2022", "World Cup 2022", "World Cup 2022",
        "World Cup 2022", "World Cup 2022", "World Cup 2022", "World Cup 2022"
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