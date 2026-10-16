import pandas as pd

new_data = {
    "video_id": [
        "penalty_92", "penalty_93", "penalty_94", "penalty_95",
        "penalty_96", "penalty_97", "penalty_98", "penalty_99",
        "penalty_100", "penalty_101", "penalty_102", "penalty_103",
        "penalty_104", "penalty_105", "penalty_106", "penalty_107",
        "penalty_108", "penalty_109", "penalty_110", "penalty_111", "penalty_112"
    ],
    "match": ["Australia vs France"] * 21,
    "player": [
        "Player1", "Player2", "Player3", "Player4", "Player5",
        "Player6", "Player7", "Player8", "Player9", "Player10",
        "Player11", "Player12", "Player13", "Player14", "Player15",
        "Player16", "Player17", "Player18", "Player19", "Player20", "Player21"
    ],
    "shot_direction": [
        "right", "left", "right", "center", "left",
        "right", "left", "right", "center", "left",
        "right", "left", "right", "center", "left",
        "right", "left", "right", "center", "left", "right"
    ],
    "outcome": [
        "goal", "goal", "goal", "goal", "goal",
        "goal", "goal", "goal", "goal", "goal",
        "goal", "goal", "goal", "goal", "goal",
        "goal", "goal", "goal", "goal", "goal", "goal"
    ],
    "dominant_foot": ["right"] * 21,
    "tournament": ["Women's World Cup 2023"] * 21
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