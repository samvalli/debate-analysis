import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_dotplot_scores(data, scores, labels):
    
    sns.set(style="whitegrid")

    platforms = data.index.tolist()  # wiki, kialo, cmv
    num_scores = len(scores)
    num_platforms = len(platforms)

    fig, ax = plt.subplots(figsize=(6, 8))

    # Assign fixed colors to platforms
    palette = {"cmv": "red", "kialo": "orange", "wiki": "blue"}

    # Spacing configuration
    score_spacing = 2.5
    platform_offset = 0.4
    y_positions = {}

    # Plot one dot per (score, platform)
    for score_idx, score in enumerate(scores):
        base_y = score_idx * score_spacing
        for i, platform in enumerate(platforms):
            value = data.loc[platform, score]

            y_position = base_y + (i - num_platforms / 2) * platform_offset
            y_positions[score] = base_y

            ax.plot(
                value, y_position,
                "o", color=palette[platform],
                label=platform if score_idx == 0 else "",
                markersize=6
            )

    # Force x-axis to start at 0.0
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(0.0, x_max)

    # Formatting
    ax.set_yticks([y_positions[s] for s in scores])
    ax.set_yticklabels(labels)
    ax.legend(framealpha=0.5)
    plt.show()