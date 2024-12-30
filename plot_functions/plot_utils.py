import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

def plot_violin_graphs_quantiles_old(data, feature):
    # Calculate mean and standard deviation for each quantile
    group_stats = data.groupby("quantile")[feature].agg(['mean', 'std']).reset_index()

    # Create violin plot
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', linewidth=0.5)
    sns.violinplot(data=data, x="quantile", y=feature, palette="muted")

    # Prepare mean values for the text box
    mean_text = "Mean Values by Quantile:\n"
    for idx, row in group_stats.iterrows():
        mean_text += f" {row['quantile']}: {row['mean']:.2f}\n"

    # Get axis limits for dynamic positioning
    x_min, x_max = plt.gca().get_xlim()
    y_min, y_max = plt.gca().get_ylim()

    # Dynamically position the text box in the top-left corner inside the plot
    plt.gca().text(
        x=x_min + 0.02 * (x_max - x_min),  # Slight padding from the left edge
        y=y_max - 0.15 * (y_max - y_min),  # Slight padding from the top edge
        s=mean_text,
        fontsize=8,  # Smaller font size
        color="black",
        bbox=dict(
            facecolor="white", 
            alpha=0.7, 
            edgecolor="black", 
            boxstyle="round,pad=0.3"  # Smaller padding inside the box
        )
    )

    # Customize the plot
    plt.title(f"{feature} Distribution Over Quantiles")
    plt.xlabel("Quantile")
    plt.ylabel(feature)
    plt.show()


    
def scatter_plot_z_scores_side_by_side(data, feature):
    """
    Create two side-by-side scatter plots with the same y-axis values
    but different x-axis values.

    Parameters:
        data (DataFrame): The dataset containing the values.
        feature (str): The column name to use for the y-axis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)  # Share y-axis for both subplots

    # First subplot
    axes[0].scatter(data['z_scores_page_it'], data[feature], color='b', alpha=0.7)
    axes[0].set_title(f"{feature} vs z_scores_page_it")
    axes[0].set_xlabel("z_scores_page_it")
    axes[0].set_ylabel(feature)
    axes[0].grid(True, linestyle='--', linewidth=0.5)

    # Second subplot
    axes[1].scatter(data['z_scores_page_mod'], data[feature], color='g', alpha=0.7)
    axes[1].set_title(f"{feature} vs z_scores_page_mod")
    axes[1].set_xlabel("z_scores_page_mod")
    axes[1].grid(True, linestyle='--', linewidth=0.5)

    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_hist(data,feature,bins):
    plt.figure(figsize=(10, 6))
    plt.hist(data[feature],bins=bins)
    plt.show()


def plot_violin_graphs_quantiles(data,groupby,feature):
    # Calculate mean and standard deviation for each quantile
    group_stats = data.groupby(groupby)[feature].agg(['mean', 'std']).reset_index()

    # Create violin plot
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', linewidth=0.5)
    sns.violinplot(data=data, x=groupby, y=feature, palette="muted")

    # Prepare mean values for the text box
    mean_text = "Mean Values by Quantile:\n"
    for idx, row in group_stats.iterrows():
        mean_text += f" {row[groupby]}: {row['mean']:.2f}\n"

    # Get axis limits for dynamic positioning
    x_min, x_max = plt.gca().get_xlim()
    y_min, y_max = plt.gca().get_ylim()

    # Dynamically position the text box in the top-left corner inside the plot
    plt.gca().text(
        x=x_min + 0.02 * (x_max - x_min),  # Slight padding from the left edge
        y=y_max - 0.15 * (y_max - y_min),  # Slight padding from the top edge
        s=mean_text,
        fontsize=8,  # Smaller font size
        color="black",
        bbox=dict(
            facecolor="white", 
            alpha=0.7, 
            edgecolor="black", 
            boxstyle="round,pad=0.3"  # Smaller padding inside the box
        )
    )

    # Customize the plot
    plt.title(f"{feature} Distribution Over {groupby}")
    plt.xlabel("Quantile")
    plt.ylabel(feature)
    plt.show()


def u_test_platform(data,feature,platform_0,platform_1):

    top_length=data[data['platform']==platform_0][feature].tolist()
    other_length=data[data['platform']==platform_1][feature].tolist()
    stat, p = mannwhitneyu(top_length, other_length, alternative='two-sided')

    print(f"P-value: {p}")
    # Interpret the result
    alpha = 0.05  # significance level
    if p < alpha:
        print("Reject the null hypothesis: The distributions are different.")
    else:
        print("Fail to reject the null hypothesis: No significant difference.")
