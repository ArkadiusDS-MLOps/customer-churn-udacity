"""
Helper functions for EDA

Author: Arkadiusz Modzelewski
"""

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_histogram(
        dataframe: pd.DataFrame,
        column: str,
        output_path: str) -> None:
    """
    Plot the histogram

    input:
            dataframe:
            density:

    output:
            None

    """
    plt.figure(figsize=(20, 10))
    sns.histplot(dataframe[column])
    plt.title(column + " Histogram")
    plt.xlabel(column)
    plt.ylabel("Number of observations")
    # Save the histogram
    plt.savefig(output_path + column + "Histogram.png")


def plot_barplot(dataframe: pd.DataFrame,
                 column: str,
                 output_path: str) -> None:
    """
    Plot the histogram

    input:
            dataframe: pandas dataframe
            output_path: path for the output plot

    output:
            None
    """
    x_axis = list(dataframe[column].value_counts('normalize').index)
    y_axis = list(dataframe[column].value_counts('normalize').values)

    plt.figure(figsize=(20, 10))
    sns.barplot(x=x_axis, y=y_axis)
    plt.title(column + " Bar Plot")
    plt.xlabel(column)
    plt.ylabel("Percentage")
    # Save the histogram
    plt.savefig(output_path + column + "BarPlot.png")


def plot_corr_heatmap(dataframe: pd.DataFrame, output_path: str) -> None:
    """
    Plot correlation heatmap

    input:
            dataframe: pandas dataframe
            output_path: path for the output plot

    output:
            None
    """

    plt.figure(figsize=(20, 20))
    sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    # Save the histogram
    plt.savefig(output_path)
