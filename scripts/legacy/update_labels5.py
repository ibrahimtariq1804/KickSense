import pandas as pd

new_data = {
    "video_id": [
        "penalty_41", "penalty_42", "penalty_43", "penalty_44",
        "penalty_45", "penalty_46", "penalty_47"
    ],
    "match": [
        "Morocco vs Spain", "Morocco vs Spain", "Morocco vs Spain", "Morocco vs Spain",
        "Morocco vs Spain", "Morocco vs Spain", "Morocco vs Spain"
    ],
    "player": [
        "Sarabia", "Hakimi", "Soler", "Ziyech",
        "Busquets", "Benoun", "Montiel"
    ],
    "shot_direction": [
        "center", "right", "left", "right",
        "left", "center", "right"
    ],
    "outcome": [
        "missed", "goal", "saved", "goal",
        "saved", "goal", "goal"
    ],
    "dominant_foot": [
        "right", "right", "right", "right",
        "right", "right", "right"
    ],
    "tournament": [
        "World Cup 2022", "World Cup 2022", "World Cup 2022", "World Cup 2022",
        "World Cup 2022", "World Cup 2022", "World Cup 2022"
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