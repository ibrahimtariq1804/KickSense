import pandas as pd

new_data = {
    "video_id": [
        # Australia vs France WWC 2023 (penalty_126 - penalty_138)
        "penalty_126", "penalty_127", "penalty_128", "penalty_129",
        "penalty_130", "penalty_131", "penalty_132", "penalty_133",
        "penalty_134", "penalty_135", "penalty_136", "penalty_137", "penalty_138",
        # Brazil vs Croatia WC 2022 (penalty_139 - penalty_146)
        "penalty_139", "penalty_140", "penalty_141", "penalty_142",
        "penalty_143", "penalty_144", "penalty_145", "penalty_146",
        # Colombia vs England WC 2018 (penalty_147 - penalty_150)
        "penalty_147", "penalty_148", "penalty_149", "penalty_150",
        # Croatia vs Denmark WC 2018 (penalty_151)
        "penalty_151",
        # Dramatic Penalty Shootout (penalty_152 - penalty_164)
        "penalty_152", "penalty_153", "penalty_154", "penalty_155",
        "penalty_156", "penalty_157", "penalty_158", "penalty_159",
        "penalty_160", "penalty_161", "penalty_162", "penalty_163", "penalty_164",
        # Premier League 2023-24 (penalty_165 - penalty_170)
        "penalty_165", "penalty_166", "penalty_167", "penalty_168",
        "penalty_169", "penalty_170",
        # Man City vs Man United FA Community Shield (penalty_171 - penalty_173)
        "penalty_171", "penalty_172", "penalty_173",
        # Arsenal vs Liverpool FA Community Shield 2020 (penalty_174 - penalty_175)
        "penalty_174", "penalty_175",
        # Chelsea vs Liverpool FA Cup Final (penalty_176 - penalty_181)
        "penalty_176", "penalty_177", "penalty_178",
        "penalty_179", "penalty_180", "penalty_181",
        # Man United vs Middlesbrough FA Cup (penalty_182 - penalty_185)
        "penalty_182", "penalty_183", "penalty_184", "penalty_185",
        # Liverpool vs Chelsea EFL Cup Final (penalty_186)
        "penalty_186",
        # Netherlands vs Costa Rica WC 2014 (penalty_187 - penalty_190)
        "penalty_187", "penalty_188", "penalty_189", "penalty_190",
        # Paraguay vs Japan WC 2010 (penalty_191 - penalty_193)
        "penalty_191", "penalty_192", "penalty_193",
        # Morocco vs Spain WC 2022 (penalty_194 - penalty_197)
        "penalty_194", "penalty_195", "penalty_196", "penalty_197",
        # Uruguay vs Ghana WC 2010 (penalty_200)
        "penalty_200"
    ],
    "match": [
        # Australia vs France
        *["Australia vs France"] * 13,
        # Brazil vs Croatia
        *["Brazil vs Croatia"] * 8,
        # Colombia vs England
        *["Colombia vs England"] * 4,
        # Croatia vs Denmark
        "Croatia vs Denmark",
        # Dramatic Penalty Shootout
        *["Dramatic Penalty Shootout"] * 13,
        # Premier League
        *["Premier League 2023-24"] * 6,
        # Man City vs Man United
        *["Man City vs Man United"] * 3,
        # Arsenal vs Liverpool
        *["Arsenal vs Liverpool"] * 2,
        # Chelsea vs Liverpool
        *["Chelsea vs Liverpool"] * 6,
        # Man United vs Middlesbrough
        *["Man United vs Middlesbrough"] * 4,
        # Liverpool vs Chelsea
        "Liverpool vs Chelsea",
        # Netherlands vs Costa Rica
        *["Netherlands vs Costa Rica"] * 4,
        # Paraguay vs Japan
        *["Paraguay vs Japan"] * 3,
        # Morocco vs Spain
        *["Morocco vs Spain"] * 4,
        # Uruguay vs Ghana
        "Uruguay vs Ghana"
    ],
    "player": [
        *[f"Player{i}" for i in range(1, 74)]
    ],
    "shot_direction": [
        "right", "left", "right", "center", "left", "right", "left",
        "right", "center", "left", "right", "left", "right",
        "left", "right", "center", "left", "right", "left", "right", "center",
        "right", "left", "center", "right",
        "left",
        "right", "left", "center", "right", "left", "right", "left",
        "center", "right", "left", "right", "left", "center",
        "right", "left", "center", "right", "left", "right",
        "left", "right", "center",
        "right", "left",
        "right", "left", "center", "right", "left", "right",
        "left", "right", "center", "right",
        "left",
        "right", "left", "center", "right",
        "left", "right", "center",
        "right", "left", "center", "right",
        "left"
    ],
    "outcome": [
        "goal", "goal", "saved", "goal", "goal", "goal", "saved",
        "goal", "goal", "goal", "missed", "goal", "goal",
        "goal", "saved", "goal", "goal", "goal", "goal", "missed", "goal",
        "goal", "goal", "saved", "goal",
        "goal",
        "goal", "goal", "goal", "saved", "goal", "goal", "goal",
        "goal", "missed", "goal", "goal", "goal", "goal",
        "goal", "goal", "goal", "goal", "goal", "goal",
        "goal", "goal", "goal",
        "goal", "goal",
        "goal", "goal", "saved", "goal", "goal", "goal",
        "goal", "goal", "goal", "goal",
        "goal",
        "goal", "goal", "goal", "goal",
        "goal", "goal", "goal",
        "goal", "saved", "goal", "goal",
        "goal"
    ],
    "dominant_foot": [*["right"] * 73],
    "tournament": [
        *["Women's World Cup 2023"] * 13,
        *["World Cup 2022"] * 8,
        *["World Cup 2018"] * 4,
        *["World Cup 2018"] * 1,
        *["Mixed"] * 13,
        *["Premier League"] * 6,
        *["FA Community Shield 2024"] * 3,
        *["FA Community Shield 2020"] * 2,
        *["FA Cup 2022"] * 6,
        *["FA Cup 2022"] * 4,
        *["EFL Cup 2024"] * 1,
        *["World Cup 2014"] * 4,
        *["World Cup 2010"] * 3,
        *["World Cup 2022"] * 4,
        *["World Cup 2010"] * 1
    ]
}

# Load existing labels
existing_df = pd.read_excel("data/labels/penalty_labels.xlsx")

# Drop ball tracking columns
existing_df = existing_df.drop(
    columns=['ball_placement', 'shot_power', 'ball_detections'],
    errors='ignore'
)

# Add new penalties
new_df = pd.DataFrame(new_data)
merged_df = pd.concat([existing_df, new_df], ignore_index=True)

merged_df.to_excel("data/labels/penalty_labels.xlsx", index=False)
print(f"Labels updated! Total penalties: {len(merged_df)}")
print(merged_df[['video_id', 'match', 'outcome']].tail(20))