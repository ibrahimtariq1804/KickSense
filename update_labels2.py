import pandas as pd

data = {
    "video_id": [
        "penalty_01", "penalty_02", "penalty_03", "penalty_04",
        "penalty_05", "penalty_06", "penalty_07", "penalty_08",
        "penalty_09", "penalty_10", "penalty_11", "penalty_12",
        "penalty_13", "penalty_14", "penalty_15", "penalty_16",
        "penalty_17", "penalty_18", "penalty_19", "penalty_20",
        "penalty_21", "penalty_22", "penalty_23"
    ],
    "match": [
        "Argentina vs France", "Argentina vs France", "Argentina vs France", "Argentina vs France",
        "Argentina vs France", "Argentina vs France", "Argentina vs France", "Argentina vs France",
        "Netherlands vs Argentina", "Netherlands vs Argentina", "Netherlands vs Argentina", "Netherlands vs Argentina",
        "Netherlands vs Argentina", "Netherlands vs Argentina", "Netherlands vs Argentina", "Netherlands vs Argentina",
        "Croatia vs Brazil", "Croatia vs Brazil", "Croatia vs Brazil", "Croatia vs Brazil",
        "Croatia vs Brazil", "Croatia vs Brazil", "Croatia vs Brazil"
    ],
    "player": [
        "Mbappe", "Messi", "Coman", "Dybala",
        "Tchouameni", "Paredes", "Kolo Muani", "Montiel",
        "Virgil van Dijk", "Messi", "Berghuis", "Fernandez",
        "Koopmeiners", "Montiel", "Enzo", "Lautaro",
        "Rodrygo", "Modric", "Casemiro", "Vlasic",
        "Marquinhos", "Majer", "Petkovic"
    ],
    "shot_direction": [
        "right", "left", "left", "right",
        "center", "left", "right", "right",
        "left", "right", "right", "left",
        "left", "right", "center", "right",
        "right", "left", "center", "right",
        "center", "left", "right"
    ],
    "outcome": [
        "goal", "goal", "saved", "goal",
        "missed", "goal", "saved", "goal",
        "saved", "goal", "saved", "goal",
        "saved", "goal", "goal", "goal",
        "saved", "goal", "goal", "goal",
        "missed", "goal", "goal"
    ],
    "dominant_foot": [
        "right", "left", "right", "right",
        "right", "right", "right", "right",
        "right", "left", "right", "right",
        "right", "right", "right", "right",
        "right", "left", "right", "right",
        "right", "right", "right"
    ],
    "tournament": [
        "World Cup 2022", "World Cup 2022", "World Cup 2022", "World Cup 2022",
        "World Cup 2022", "World Cup 2022", "World Cup 2022", "World Cup 2022",
        "World Cup 2022", "World Cup 2022", "World Cup 2022", "World Cup 2022",
        "World Cup 2022", "World Cup 2022", "World Cup 2022", "World Cup 2022",
        "World Cup 2022", "World Cup 2022", "World Cup 2022", "World Cup 2022",
        "World Cup 2022", "World Cup 2022", "World Cup 2022"
    ]
}

df = pd.DataFrame(data)
df.to_excel("data/labels/penalty_labels.xlsx", index=False)
print("Labels updated successfully!")
print(f"Total penalties in dataset: {len(df)}")
print(df)