"""

"""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV


def hyperparameter_optimization(x_train, y_train, model, param_dict):
    """
    Perform hyperparameter optimization for model
    """

    cv_rfc = GridSearchCV(estimator=model, param_grid=param_dict, cv=5)
    cv_rfc.fit(x_train, y_train)

    return cv_rfc.best_estimator_


def classification_report_image(
        y_train,
        y_test,
        y_train_pred_lr,
        y_train_pred_rf,
        y_test_pred_lr,
        y_test_pred_rf,
        output_path
):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_pred_lr: training predictions from logistic regression
            y_train_pred_rf: training predictions from random forest
            y_test_pred_lr: test predictions from logistic regression
            y_test_pred_rf: test predictions from random forest

    output:
             None
    """
    # Create a single figure with a 2x2 grid
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))

    # Train Classification Report - Logistic Regression
    clf_report_train_lr = classification_report(
        y_train, y_train_pred_lr, output_dict=True)
    sns.heatmap(pd.DataFrame(
        clf_report_train_lr).iloc[:-1, :].T, annot=True, ax=axes[0, 0])
    axes[0, 0].set_title('Train Classification Report - Logistic Regression')

    # Test Classification Report - Logistic Regression
    clf_report_test_lr = classification_report(
        y_test, y_test_pred_lr, output_dict=True)
    sns.heatmap(pd.DataFrame(
        clf_report_test_lr).iloc[:-1, :].T, annot=True, ax=axes[0, 1])
    axes[0, 1].set_title('Test Classification Report - Logistic Regression')

    # Train Classification Report - Random Forest
    clf_report_train_rf = classification_report(
        y_train, y_train_pred_rf, output_dict=True)
    sns.heatmap(pd.DataFrame(
        clf_report_train_rf).iloc[:-1, :].T, annot=True, ax=axes[1, 0])
    axes[1, 0].set_title('Train Classification Report - Random Forest')

    # Test Classification Report - Random Forest
    clf_report_test_rf = classification_report(
        y_test, y_test_pred_rf, output_dict=True)
    sns.heatmap(pd.DataFrame(
        clf_report_test_rf).iloc[:-1, :].T, annot=True, ax=axes[1, 1])
    axes[1, 1].set_title('Test Classification Report - Random Forest')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the classification report image
    plt.savefig(output_path)


def plot_roc_comparison(rfc, lrc, x_test, y_test, output_path: str) -> None:
    """
    Plot roc curves comparison

    input:
            rfc:
            lrc:
            x_test:
            y_test:
            output_path:

    output:
            None

    """
    plt.figure(figsize=(20, 10))
    ax = plt.gca()

    # Plot Random Forest ROC curve
    plot_roc_curve(
        rfc,
        x_test,
        y_test,
        ax=ax,
        alpha=0.8,
        name='Random Forest')
    # Plot Logistic Regression ROC curve
    plot_roc_curve(
        lrc, x_test, y_test, ax=ax, alpha=0.8, name='Logistic Regression'
    )

    plt.legend(loc='lower right')
    plt.savefig(output_path)


def feature_importance_plot(model, x_data, output_path):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    # Calculate feature importance
    importance = model.feature_importances_
    # Sort feature importance in descending order
    indices = np.argsort(importance)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 15))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importance[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    # Save the feature importance plot
    plt.savefig(output_path)
