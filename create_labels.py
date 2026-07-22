import pandas as pd

data = {
    "video_id": [
        "penalty_01", "penalty_02", "penalty_03", "penalty_04",
        "penalty_05", "penalty_06", "penalty_07", "penalty_08"
    ],
    "match": [
        "Argentina vs France", "Argentina vs France", "Argentina vs France", "Argentina vs France",
        "Argentina vs France", "Argentina vs France", "Argentina vs France", "Argentina vs France"
    ],
    "player": [
        "Mbappe", "Messi", "Coman", "Dybala",
        "Tchouameni", "Paredes", "Kolo Muani", "Montiel"
    ],
    "shot_direction": [
        "right", "left", "left", "right",
        "center", "left", "right", "right"
    ],
    "outcome": [
        "goal", "goal", "saved", "goal",
        "missed", "goal", "saved", "goal"
    ],
    "dominant_foot": [
        "right", "left", "right", "right",
        "right", "right", "right", "right"
    ],
    "tournament": [
        "World Cup 2022", "World Cup 2022", "World Cup 2022", "World Cup 2022",
        "World Cup 2022", "World Cup 2022", "World Cup 2022", "World Cup 2022"
    ]
}

df = pd.DataFrame(data)
df.to_excel("data/labels/penalty_labels.xlsx", index=False)
print("Labels Excel created successfully!")
print(df)