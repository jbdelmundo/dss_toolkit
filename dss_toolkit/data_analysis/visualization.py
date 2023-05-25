import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

guide = """
Common Visualizations for Exploratory Data Analysis

Scatterplot
- Numeric Vs Numeric
- Numeric Vs Numeric Vs Categorical (Multple colors)

Histogram (for frequency, distribution)
- Numeric
- Numeric Vs Categorical (multiple histograms)

Box Plot (for range, quantiles)
- Numeric Vs Categorical

Countplot
- Categorical Vs Categorical

"""
def help():
    print(guide)

def plt_scatter(data, x_col, y_col, category_col=None, figsize=(20, 10),title=None):
    """
    Scatterplot for continuous vs continuous relationship
    
    Parameters
    ----------
    data : Dataframe
    x_col: str
        Data along X-Axis
    y_col: str
        Data along Y-Axis
    category_col : str (default: None)
        If set, plot will be grouped ("colored") using this column
    """

    plt.figure(figsize=figsize)

    if category_col is not None:
        category_values = data[category_col].unique()
        for cat_val in category_values:
            ix = data[category_col] == cat_val
            x_, y_ = data.loc[ix, x_col], data.loc[ix, y_col]
            plt.scatter(x_, y_,label=cat_val)
        plt.legend()
    else: 
        x_, y_ = data.loc[:, x_col], data.loc[:, y_col]
        plt.scatter(x_, y_)

    # plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="both", style="plain")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if title is None:
        plt.title(f"{x_col} vs {y_col}")
    else:
        plt.title(title)
    plt.show()


def plt_hist(data, col, category_col=None,plot_consolidated=False, figsize=(20, 10), bars=100,quantile=(0.0,1.0),title=None):
    plt.figure(figsize=figsize)
    plt.ticklabel_format(axis="both", style="plain")

    s = data[col]
    
    start = s.quantile(quantile[0])
    end = s.quantile(quantile[1])
    step = (end - start) / bars
    
    if title:
        plt.title(title)

    if category_col is not None:
        if plot_consolidated:
            s.hist(bins=np.arange(start, end, step), alpha=0.1, label="All")

        for cat_val in data[category_col].value_counts().index.tolist():
            s_ = data.loc[data[category_col] == cat_val, col]
            s_.hist(bins=np.arange(start, end, step), alpha=0.3, label=cat_val)

        plt.legend()
    else:

        s.hist(bins=np.arange(start, end, step))
    if title is None:
        plt.title(f"Histogram of {col}")
    else:
        plt.title(title)
    plt.xlabel(col)
    plt.show()


def plt_boxplot(data, col, category=None, figsize=(20, 10), bars=100):
    plt.figure(figsize=figsize)
    plt.ticklabel_format(axis="both", style="plain")

    s = data[col]
    start = s.quantile(0.05)
    end = s.quantile(0.95)
    step = (end - start) / bars

    #         s.hist(bins=np.arange(start, end, step), alpha=0.1, label="All")
    items = []
    labels = []
    for cat_val in category.value_counts().index.tolist():
        s_ = data.loc[category == cat_val, col]
        items.append(s_)
        labels.append(cat_val)

    plt.boxplot(items, labels=labels)
    plt.show()


def plt_boxplot2(data, numeric_col, cat_cols, showfliers=False, figsize=(20, 10)):

    ax = data.boxplot(
        column=numeric_col,
        by=cat_cols,
        figsize=figsize,
        showfliers=showfliers,
    )
    #     ax.ticklabel_format(axis="both", style="plain")
    plt.show()


def plt_countplot(data, x, hue=None, figsize=(20, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    if hue is not None:
        sns.countplot(x=x, hue=hue, data=data, ax=ax)
    else:
        sns.countplot(x=x, data=data, ax=ax)
