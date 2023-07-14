"""
Prediction of Customer Churn

Author: Arkadiusz Modzelewski
"""

# import libraries
import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

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


def plot_corr_heatmap(dataframe: pd.DataFrame, output_path: str) -> None:
    """
    Plot correlation heatmap

    :param dataframe: pandas dataframe
    :param output_path: path for the output plot

    """
    plt.figure(figsize=(20, 10))
    sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths=2)
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
        dataframe=dataframe, density=True, column="Total_Trans_Ct",
        title="Total_Trans_Ct Histogram", xlabel="Total Trans_Ct", ylabel="Density",
        output_path=output_path + "Total_Trans_Ct_Histogram.png"
    )
    plot_barplot(
        x=list(df.Marital_Status.value_counts('normalize').index),
        y=list(df.Marital_Status.value_counts('normalize').values),
        title="Marital_Status Bar Plot", xlabel="Marital Status", ylabel="Percentage",
        output_path=output_path + "Marital_Status_Barplot.png"
    )
    plot_corr_heatmap(dataframe=dataframe, output_path=output_path + "correlation_heatmap.png")


def encoder_helper(dataframe: pd.DataFrame, category_lst: list) -> pd.DataFrame:
    """
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features

    output:
            dataframe: pandas dataframe with new columns for
    """

    for column in category_lst:

        values_lst = []
        values_groups = dataframe.groupby(column).mean()['Churn']

        for val in dataframe[column]:
            values_lst.append(values_groups.loc[val])

        response_col = column + "_Churn"
        dataframe[response_col] = values_lst

    return dataframe


def perform_feature_engineering(dataframe, keep_cols):
    """
    input:
              dataframe: pandas dataframe

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    X = pd.DataFrame()
    y, X[keep_cols] = dataframe['Churn'], dataframe[keep_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


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

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    df = encoder_helper(dataframe=df, category_lst=cat_columns)

    cols_to_keep = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                    'Total_Relationship_Count', 'Months_Inactive_12_mon',
                    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                    'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                    'Income_Category_Churn', 'Card_Category_Churn']

    perform_feature_engineering(dataframe=df, keep_cols=cols_to_keep)
