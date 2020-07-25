import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_feature_importance(features, model, n):
    """
    plot ordered feature importance (random forest)

    :param features: list of features
    :param model: random forest model
    :param n: number of top n features to plot

    """
    feature_importance = pd.DataFrame({'Features': features, 'Importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(x="Importance", y="Features", data=feature_importance[:n])


