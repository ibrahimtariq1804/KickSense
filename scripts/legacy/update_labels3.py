import pandas as pd

# New penalties from England vs Switzerland Euro 2024
new_data = {
    "video_id": [
        "penalty_24", "penalty_25", "penalty_26", "penalty_27",
        "penalty_28", "penalty_29", "penalty_30", "penalty_31", "penalty_32"
    ],
    "match": [
        "England vs Switzerland", "England vs Switzerland", "England vs Switzerland",
        "England vs Switzerland", "England vs Switzerland", "England vs Switzerland",
        "England vs Switzerland", "England vs Switzerland", "England vs Switzerland"
    ],
    "player": [
        "Bellingham", "Xhaka", "Saka", "Akanji",
        "Trent", "Vargas", "Trippier", "Shaqiri", "Pickford"
    ],
    "shot_direction": [
        "right", "left", "right", "left",
        "right", "left", "right", "center", "left"
    ],
    "outcome": [
        "goal", "goal", "goal", "goal",
        "goal", "saved", "goal", "goal", "goal"
    ],
    "dominant_foot": [
        "right", "right", "right", "right",
        "right", "right", "right", "right", "right"
    ],
    "tournament": [
        "Euro 2024", "Euro 2024", "Euro 2024", "Euro 2024",
        "Euro 2024", "Euro 2024", "Euro 2024", "Euro 2024", "Euro 2024"
    ]
}

# Load existing labels
existing_df = pd.read_excel("data/labels/penalty_labels.xlsx")

# Add new penalties
new_df = pd.DataFrame(new_data)
merged_df = pd.concat([existing_df, new_df], ignore_index=True)

merged_df.to_excel("data/labels/penalty_labels.xlsx", index=False)
print(f"Labels updated! Total penalties: {len(merged_df)}")
print(merged_df[['video_id', 'match', 'outcome']])