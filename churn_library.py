"""
Prediction of Customer Churn

Author: Arkadiusz Modzelewski
"""

# import libraries
import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth: str) -> pd.DataFrame:
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            dataframe: pandas dataframe
    """
    dataframe = pd.read_csv(pth)
    return dataframe


def plot_histogram(dataframe: pd.DataFrame, density: bool, column: str, title: str,
                   xlabel: str, ylabel: str, output_path: str) -> None:
    """
    Plot the histogram

    :param dataframe:
    :param density:


    """
    plt.figure(figsize=(20, 10))
    if density:
        sns.histplot(dataframe[column], stat='density', kde=True)
    else:
        sns.histplot(dataframe[column])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Save the histogram
    plt.savefig(output_path)


def plot_barplot(x: list, y: list, title: str, xlabel: str, ylabel: str, output_path: str) -> None:
    """
    Plot the histogram


    """
    plt.figure(figsize=(20, 10))
    sns.barplot(x=x, y=y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Save the histogram
    plt.savefig(output_path)


def perform_eda(dataframe: pd.DataFrame, output_path: str) -> None:
    """
    perform eda on df and save figures to images folder
    input:
            dataframe: pandas dataframe

    output:
            None
    """
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    plot_histogram(
        dataframe=dataframe, density=False, column="Churn", title="Churn Histogram", xlabel="Label",
        ylabel="Number of observations", output_path=output_path + "Churn_Histogram.png"
    )
    plot_histogram(
        dataframe=dataframe, density=False, column="Customer_Age", title="Customer_Age Histogram",
        xlabel="Customer Age", ylabel="Number of observations",
        output_path=output_path + "Customer_Age_Histogram.png"
    )
    plot_histogram(
        dataframe=dataframe, density=False, column="Total_Trans_Ct",
        title="Total_Trans_Ct Histogram", xlabel="Total Trans_Ct", ylabel="Density",
        output_path=output_path + "Total_Trans_Ct_Histogram.png"
    )
    plot_barplot(
        x=list(df.Marital_Status.value_counts('normalize').index),
        y=list(df.Marital_Status.value_counts('normalize').values),
        title="Marital_Status Bar Plot", xlabel="Marital Status", ylabel="Percentage",
        output_path=output_path + "Marital_Status_Barplot.png"
    )


def encoder_helper(dataframe, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used
                      for naming variables or index y column]

    output:
            dataframe: pandas dataframe with new columns for
    """
    pass


def perform_feature_engineering(dataframe, response):
    """
    input:
              dataframe: pandas dataframe
              response: string of response name [optional argument that could be used
                        for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    pass


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    pass


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    pass


def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    pass


if __name__ == '__main__':
    df = import_data("data/bank_data.csv")
    perform_eda(dataframe=df, output_path="images/eda/")
