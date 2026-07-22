import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

df = pd.read_excel("data/labels/penalty_labels.xlsx")
os.makedirs("outputs/visualizations", exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
colors = {
    'blue': '#378ADD', 'green': '#639922', 'red': '#E24B4A',
    'orange': '#D85A30', 'gray': '#888780', 'teal': '#1D9E75',
    'purple': '#7F77DD', 'amber': '#BA7517', 'pink': '#D4537E'
}

# ============================================================
# VIZ 1: Complete Dataset Overview Dashboard
# ============================================================
fig = plt.figure(figsize=(18, 10))
fig.suptitle('KickSense Dataset Overview — 125 Penalties', 
             fontsize=20, fontweight='bold', y=1.01)

gs = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.4)

# Shot Direction
ax1 = fig.add_subplot(gs[0, 0])
counts = df['shot_direction'].value_counts()
bars = ax1.bar(counts.index, counts.values,
               color=[colors['blue'], colors['teal'], colors['gray']],
               edgecolor='white', width=0.5)
ax1.set_title('Shot Direction', fontweight='bold')
ax1.set_ylabel('Count')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             str(val), ha='center', fontsize=10, fontweight='bold')

# Outcome
ax2 = fig.add_subplot(gs[0, 1])
counts = df['outcome'].value_counts()
wedges, texts, autotexts = ax2.pie(
    counts.values, labels=counts.index,
    colors=[colors['green'], colors['orange'], colors['red']],
    autopct='%1.0f%%', startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=2))
for a in autotexts:
    a.set_fontsize(10)
    a.set_fontweight('bold')
ax2.set_title('Outcome', fontweight='bold')

# Ball Placement
ax3 = fig.add_subplot(gs[0, 2])
counts = df['ball_placement'].value_counts()
bars = ax3.bar(counts.index, counts.values,
               color=colors['blue'], edgecolor='white', width=0.5)
ax3.set_title('Ball Placement', fontweight='bold')
ax3.set_ylabel('Count')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)
for bar, val in zip(bars, counts.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             str(val), ha='center', fontsize=10, fontweight='bold')

# Shot Power
ax4 = fig.add_subplot(gs[0, 3])
ax4.hist(df['shot_power'], bins=10, color=colors['teal'],
         edgecolor='white', linewidth=0.5)
ax4.set_title('Shot Power', fontweight='bold')
ax4.set_xlabel('Power (1-10)')
ax4.set_ylabel('Frequency')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# Tournament
ax5 = fig.add_subplot(gs[1, 0:2])
counts = df['tournament'].value_counts()
bars = ax5.barh(counts.index, counts.values,
                color=colors['blue'], edgecolor='white', height=0.5)
ax5.set_title('Penalties by Tournament', fontweight='bold')
ax5.set_xlabel('Count')
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
for bar, val in zip(bars, counts.values):
    ax5.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
             str(val), ha='left', va='center', fontsize=10, fontweight='bold')

# Direction vs Outcome Stacked
ax6 = fig.add_subplot(gs[1, 2:4])
crosstab = pd.crosstab(df['shot_direction'], df['outcome'])
crosstab.plot(kind='bar', stacked=True, ax=ax6,
              color=[colors['green'], colors['red'], colors['orange']],
              edgecolor='white', linewidth=0.5, width=0.5)
ax6.set_title('Direction vs Outcome', fontweight='bold')
ax6.set_xlabel('Shot Direction')
ax6.set_ylabel('Count')
ax6.legend(title='Outcome', bbox_to_anchor=(1.02, 1))
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('outputs/visualizations/01_dataset_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 01_dataset_overview.png")

# ============================================================
# VIZ 2: Goal Heatmap on Goal Frame
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))

goal_data = {
    'top_left': 50, 'top_right': 30, 'center': 39,
    'bottom_left': 3, 'bottom_right': 3
}

goal_zones = {
    'top_left':     (0.05, 0.5, 0.3, 0.45),
    'center':       (0.35, 0.1, 0.3, 0.85),
    'top_right':    (0.65, 0.5, 0.3, 0.45),
    'bottom_left':  (0.05, 0.05, 0.3, 0.45),
    'bottom_right': (0.65, 0.05, 0.3, 0.45),
}

max_val = max(goal_data.values())
cmap = plt.cm.Blues

for zone, (x, y, w, h) in goal_zones.items():
    val = goal_data.get(zone, 0)
    intensity = val / max_val
    rect = mpatches.FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.01",
                                    facecolor=cmap(0.2 + intensity * 0.8),
                                    edgecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, f'{val}\n{zone.replace("_", " ")}',
            ha='center', va='center', fontsize=13, fontweight='bold',
            color='white' if intensity > 0.4 else '#1a1a2e')

# Goal frame
goal_frame = mpatches.FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                      boxstyle="round,pad=0.01",
                                      fill=False,
                                      edgecolor='#1a1a2e', linewidth=4)
ax.add_patch(goal_frame)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Ball Placement Heatmap\n(Goalkeeper\'s View)', 
             fontsize=16, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('outputs/visualizations/02_ball_placement_heatmap.png', 
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 02_ball_placement_heatmap.png")

# ============================================================
# VIZ 3: Shot Power vs Outcome
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Shot Power Analysis', fontsize=16, fontweight='bold')

# Box plot
outcome_order = ['goal', 'saved', 'missed']
outcome_colors = [colors['green'], colors['orange'], colors['red']]
data_by_outcome = [df[df['outcome'] == o]['shot_power'].values for o in outcome_order]
bp = axes[0].boxplot(data_by_outcome, labels=outcome_order, patch_artist=True,
                     medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp['boxes'], outcome_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
axes[0].set_title('Shot Power by Outcome', fontweight='bold')
axes[0].set_ylabel('Shot Power')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Average power per direction
avg_power = df.groupby('shot_direction')['shot_power'].mean()
bars = axes[1].bar(avg_power.index, avg_power.values,
                   color=[colors['blue'], colors['teal'], colors['gray']],
                   edgecolor='white', width=0.4)
axes[1].set_title('Average Shot Power by Direction', fontweight='bold')
axes[1].set_ylabel('Average Power')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
for bar, val in zip(bars, avg_power.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/visualizations/03_shot_power_analysis.png', 
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 03_shot_power_analysis.png")

# ============================================================
# VIZ 4: Match-wise Breakdown
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Match-wise Analysis', fontsize=16, fontweight='bold')

# Penalties per match
match_counts = df['match'].value_counts()
bars = axes[0].barh(match_counts.index, match_counts.values,
                    color=colors['blue'], edgecolor='white', height=0.6)
axes[0].set_title('Penalties per Match', fontweight='bold')
axes[0].set_xlabel('Count')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
for bar, val in zip(bars, match_counts.values):
    axes[0].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 str(val), ha='left', va='center', fontsize=10, fontweight='bold')

# Goal rate per match
goal_rate = df.groupby('match').apply(
    lambda x: (x['outcome'] == 'goal').sum() / len(x) * 100
).sort_values(ascending=True)
bars = axes[1].barh(goal_rate.index, goal_rate.values,
                    color=colors['green'], edgecolor='white', height=0.6)
axes[1].set_title('Goal Rate per Match (%)', fontweight='bold')
axes[1].set_xlabel('Goal Rate (%)')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].axvline(x=goal_rate.mean(), color=colors['red'], 
                linestyle='--', linewidth=1.5, label=f'Avg: {goal_rate.mean():.0f}%')
axes[1].legend()
for bar, val in zip(bars, goal_rate.values):
    axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{val:.0f}%', ha='left', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/visualizations/04_match_analysis.png', 
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 04_match_analysis.png")

# ============================================================
# VIZ 5: Correlation Heatmap
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

df_encoded = df.copy()
df_encoded['shot_direction_enc'] = df['shot_direction'].map({'left': 0, 'center': 1, 'right': 2})
df_encoded['outcome_enc'] = df['outcome'].map({'missed': 0, 'saved': 1, 'goal': 2})
df_encoded['foot_enc'] = df['dominant_foot'].map({'left': 0, 'right': 1})
df_encoded['placement_enc'] = df['ball_placement'].map({
    'bottom_left': 0, 'bottom_right': 1, 'center': 2, 'top_left': 3, 'top_right': 4
})

corr_cols = ['shot_direction_enc', 'outcome_enc', 'foot_enc', 
             'shot_power', 'placement_enc', 'ball_detections']
corr_labels = ['Shot Direction', 'Outcome', 'Dominant Foot', 
               'Shot Power', 'Ball Placement', 'Ball Detections']

corr_matrix = df_encoded[corr_cols].corr()
corr_matrix.index = corr_labels
corr_matrix.columns = corr_labels

im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, shrink=0.8)

ax.set_xticks(range(len(corr_labels)))
ax.set_yticks(range(len(corr_labels)))
ax.set_xticklabels(corr_labels, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(corr_labels, fontsize=10)

for i in range(len(corr_labels)):
    for j in range(len(corr_labels)):
        val = corr_matrix.iloc[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=9, color=color, fontweight='bold')

ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('outputs/visualizations/05_correlation_heatmap.png', 
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 05_correlation_heatmap.png")

print("\n=== ALL VISUALIZATIONS SAVED TO outputs/visualizations/ ===")
print(f"Total: 5 visualization files")