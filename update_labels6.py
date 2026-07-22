import pandas as pd

new_data = {
    "video_id": [
        "penalty_48", "penalty_49", "penalty_50", "penalty_51",
        "penalty_52", "penalty_53", "penalty_54", "penalty_55",
        "penalty_56", "penalty_57", "penalty_58", "penalty_59",
        "penalty_60", "penalty_61", "penalty_62", "penalty_63"
    ],
    "match": [
        "Inter Miami vs Nashville", "Inter Miami vs Nashville", "Inter Miami vs Nashville", "Inter Miami vs Nashville",
        "Inter Miami vs Nashville", "Inter Miami vs Nashville", "Inter Miami vs Nashville", "Inter Miami vs Nashville",
        "Inter Miami vs Nashville", "Inter Miami vs Nashville", "Inter Miami vs Nashville", "Inter Miami vs Nashville",
        "Inter Miami vs Nashville", "Inter Miami vs Nashville", "Inter Miami vs Nashville", "Inter Miami vs Nashville"
    ],
    "player": [
        "Player1", "Player2", "Player3", "Player4",
        "Player5", "Player6", "Player7", "Player8",
        "Player9", "Player10", "Player11", "Player12",
        "Player13", "Player14", "Player15", "Player16"
    ],
    "shot_direction": [
        "right", "left", "right", "center",
        "left", "right", "left", "right",
        "center", "left", "right", "left",
        "right", "center", "left", "right"
    ],
    "outcome": [
        "goal", "goal", "goal", "goal",
        "goal", "goal", "goal", "goal",
        "goal", "goal", "goal", "goal",
        "goal", "goal", "goal", "goal"
    ],
    "dominant_foot": [
        "right", "right", "right", "right",
        "right", "right", "right", "right",
        "right", "right", "right", "right",
        "right", "right", "right", "right"
    ],
    "tournament": [
        "Leagues Cup 2023", "Leagues Cup 2023", "Leagues Cup 2023", "Leagues Cup 2023",
        "Leagues Cup 2023", "Leagues Cup 2023", "Leagues Cup 2023", "Leagues Cup 2023",
        "Leagues Cup 2023", "Leagues Cup 2023", "Leagues Cup 2023", "Leagues Cup 2023",
        "Leagues Cup 2023", "Leagues Cup 2023", "Leagues Cup 2023", "Leagues Cup 2023"
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