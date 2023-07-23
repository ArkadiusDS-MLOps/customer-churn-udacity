"""
Includes Churn Class that is used for churn modelling

Author: Arkadiusz Modzelewski
"""
import logging

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils.eda import plot_histogram, plot_barplot, plot_corr_heatmap
from utils.training_and_evaluation import (
    hyperparameter_optimization, classification_report_image,
    plot_roc_comparison, feature_importance_plot
)


class ChurnModel:
    """
    Class for Churn Modelling

    """

    def __init__(self):
        """
        Class init method or constructor
        """

        self.dataframe = None
        self.categorical_features = None
        self.numerical_features = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def import_data(self, data_path: str) -> pd.DataFrame:
        """
        Returns dataframe for the csv found at pth

        Parameters:
            self: The instance of the class
            data_path: path to file where data is stored
        Returns:
            dataframe: pandas dataframe

        """
        try:
            self.dataframe = pd.read_csv(data_path)
            self.dataframe['Churn'] = self.dataframe['Attrition_Flag'].apply(
                lambda val: 0 if val == "Existing Customer" else 1
            )
            del self.dataframe['Attrition_Flag']
            del self.dataframe['Unnamed: 0']

        except FileNotFoundError:
            logging.info("Method import_data: Path to the file does not exist")

    def perform_eda(self, eda_output_path="images/eda/") -> None:
        """
        Perform eda on df and save figures to images folder

        Parameters:
            self: The instance of the class
            eda_output_path: path for eda resulting plots
        Returns:
            None

        """

        # Select categorical features
        self.categorical_features = self.dataframe.select_dtypes(
            include=['object']
        ).columns.tolist()

        # Select numerical features
        self.numerical_features = self.dataframe.select_dtypes(
            include=['int64', 'float64']
        ).columns.tolist()

        for num_feature in self.numerical_features:
            plot_histogram(
                dataframe=self.dataframe,
                column=num_feature,
                output_path=eda_output_path
            )

        for cat_feature in self.categorical_features:
            plot_barplot(
                dataframe=self.dataframe,
                column=cat_feature,
                output_path=eda_output_path
            )

        plot_corr_heatmap(
            dataframe=self.dataframe,
            output_path=eda_output_path
        )

    def encode_cat_features(self, category_list: list) -> pd.DataFrame:
        """
        Turn each categorical column into a new column with
        proportion of churn for each category - associated with cell 15 from the notebook

        Parameters:
            self: The instance of the class
            category_list: list of columns that contain categorical features
        Returns:
            None

        """

        for column in category_list:

            values_lst = []
            values_groups = self.dataframe.groupby(column).mean()['Churn']

            for val in self.dataframe[column]:
                values_lst.append(values_groups.loc[val])

            response_col = column + "_Churn"
            self.dataframe[response_col] = values_lst

    def perform_feature_engineering(self, cols_to_keep: list) -> None:
        """
        Perform feature engineering and split data into train and test data

        Parameters:
            self: The instance of the class
            cols_to_keep: List of columns to keep for training
        Returns:
            None

        """
        y_dependent = self.dataframe['Churn']
        x_features = self.dataframe[cols_to_keep]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x_features, y_dependent, test_size=0.3, random_state=42
        )

    def train_models(self, results_output_path="images/results/", models_path="./models/"):
        """
        Train, store model results: images + scores, and store models

        Parameters:
            self: The instance of the class
            results_output_path: path to directory with results plots
            models_path: path to save resulting models
        Returns:
            None
        """

        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

        lrc.fit(self.x_train, self.y_train)

        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }

        rfc = hyperparameter_optimization(
            x_train=self.x_train, y_train=self.y_train, model=rfc, param_dict=param_grid
        )

        y_train_preds_rf = rfc.predict(self.x_train)
        y_test_preds_rf = rfc.predict(self.x_test)

        y_train_preds_lr = lrc.predict(self.x_train)
        y_test_preds_lr = lrc.predict(self.x_test)

        # scores
        classification_report_image(
            self.y_train,
            self.y_test,
            y_train_preds_lr,
            y_train_preds_rf,
            y_test_preds_lr,
            y_test_preds_rf,
            output_path=results_output_path + "classification_report.png"
        )

        # plot roc curve
        plot_roc_comparison(
            rfc=rfc,
            lrc=lrc,
            x_test=self.x_test,
            y_test=self.y_test,
            output_path=results_output_path + "roc_curves.png")

        feature_importance_plot(
            model=rfc,
            x_data=self.x_train,
            output_path=results_output_path + "feature_importance_plot.png"
        )

        # save best model
        joblib.dump(rfc, models_path + "rfc_model.pkl")
        joblib.dump(lrc, models_path + "logistic_model.pkl")
