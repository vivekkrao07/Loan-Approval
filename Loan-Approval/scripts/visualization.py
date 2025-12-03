import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def plot_top3_tree(clf, feature_names):
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=["Not Approved", "Approved"],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=10,
        ax=ax
    )
    plt.title("Decision Tree (Top 3 Levels)", fontsize=16)
    plt.tight_layout()
    return fig
