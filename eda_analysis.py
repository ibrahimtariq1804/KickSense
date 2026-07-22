import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np

df = pd.read_excel("data/labels/penalty_labels.xlsx")

plt.style.use('seaborn-v0_8-whitegrid')
colors = {'blue': '#378ADD', 'green': '#639922', 'red': '#E24B4A', 
          'orange': '#D85A30', 'gray': '#888780', 'teal': '#1D9E75'}

# Chart 1: Shot Direction
fig, ax = plt.subplots(figsize=(8, 5))
counts = df['shot_direction'].value_counts()
bars = ax.bar(counts.index, counts.values, 
              color=[colors['blue'], colors['teal'], colors['gray']], 
              edgecolor='white', linewidth=0.5, width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_title('Shot Direction Distribution', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Direction', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('outputs/eda_shot_direction.png', dpi=150, bbox_inches='tight')
plt.show()

# Chart 2: Outcome Pie
fig, ax = plt.subplots(figsize=(7, 7))
counts = df['outcome'].value_counts()
wedge_colors = [colors['green'], colors['orange'], colors['red']]
wedges, texts, autotexts = ax.pie(counts.values, labels=counts.index, 
                                   autopct='%1.1f%%', colors=wedge_colors,
                                   startangle=90, pctdistance=0.85,
                                   wedgeprops=dict(edgecolor='white', linewidth=2))
for text in autotexts:
    text.set_fontsize(12)
    text.set_fontweight('bold')
ax.set_title('Penalty Outcome Breakdown', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('outputs/eda_outcome.png', dpi=150, bbox_inches='tight')
plt.show()

# Chart 3: Shot Direction vs Outcome
fig, ax = plt.subplots(figsize=(9, 5))
crosstab = pd.crosstab(df['shot_direction'], df['outcome'])
crosstab.plot(kind='bar', ax=ax, 
              color=[colors['green'], colors['red'], colors['orange']],
              edgecolor='white', linewidth=0.5, width=0.6)
ax.set_title('Shot Direction vs Outcome', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Shot Direction', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.legend(title='Outcome', bbox_to_anchor=(1.02, 1), loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('outputs/eda_direction_vs_outcome.png', dpi=150, bbox_inches='tight')
plt.show()

# Chart 4: Ball Placement
fig, ax = plt.subplots(figsize=(9, 5))
counts = df['ball_placement'].value_counts()
bars = ax.bar(counts.index, counts.values, 
              color=colors['blue'], edgecolor='white', linewidth=0.5, width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('Ball Placement Distribution', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Placement', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('outputs/eda_ball_placement.png', dpi=150, bbox_inches='tight')
plt.show()

# Chart 5: Shot Power Distribution
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df['shot_power'], bins=10, color=colors['teal'], 
        edgecolor='white', linewidth=0.5)
ax.set_title('Shot Power Distribution', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Shot Power (1-10)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('outputs/eda_shot_power.png', dpi=150, bbox_inches='tight')
plt.show()

# Chart 6: Penalties per Tournament
fig, ax = plt.subplots(figsize=(10, 5))
counts = df['tournament'].value_counts()
bars = ax.barh(counts.index, counts.values, 
               color=colors['blue'], edgecolor='white', linewidth=0.5, height=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            str(val), ha='left', va='center', fontsize=11, fontweight='bold')
ax.set_title('Penalties by Tournament', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Count', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('outputs/eda_tournament.png', dpi=150, bbox_inches='tight')
plt.show()

# Chart 7: Dominant Foot
fig, ax = plt.subplots(figsize=(7, 5))
counts = df['dominant_foot'].value_counts()
bars = ax.bar(counts.index, counts.values,
              color=[colors['blue'], colors['teal']],
              edgecolor='white', linewidth=0.5, width=0.4)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_title('Dominant Foot Distribution', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Foot', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('outputs/eda_dominant_foot.png', dpi=150, bbox_inches='tight')
plt.show()

# Chart 8: Data Cleaning Report
print("\n=== DATA CLEANING REPORT ===")
print(f"\nTotal penalties: {len(df)}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")
print(f"\nClass balance - Shot Direction:")
print(df['shot_direction'].value_counts())
print(f"\nClass balance - Outcome:")
print(df['outcome'].value_counts())
print(f"\nClass balance - Ball Placement:")
print(df['ball_placement'].value_counts())
print("\n=== CLEANING COMPLETE ===")

print("\nAll EDA charts saved to outputs/ folder!")