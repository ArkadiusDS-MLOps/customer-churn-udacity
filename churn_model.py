"""
Includes Churn Class that is used for churn modelling
"""
import logging

import pandas as pd

from utils.eda import plot_histogram, plot_barplot


class ChurnModel:
    """
    Class for Churn Modelling

    """

    def __init__(self, data_path: str, eda_output_path: str, results_output_path: str):
        """
        Class init method or constructor
        """

        self.data_path = data_path
        self.eda_output_path = eda_output_path
        self.results_output_path = results_output_path
        self.dataframe = None

    def import_data(self) -> pd.DataFrame:
        """
        Returns dataframe for the csv found at pth

        Parameters:
            self: The instance of the class
        Returns:
            dataframe: pandas dataframe

        """
        try:
            self.dataframe = pd.read_csv(self.data_path)
            self.dataframe['Churn'] = self.dataframe['Attrition_Flag'].apply(
                lambda val: 0 if val == "Existing Customer" else 1
            )
            del self.dataframe['Attrition_Flag']
            del self.dataframe['Unnamed: 0']

        except FileNotFoundError:
            logging.info("Method import_data: Path to the file does not exist")

    def perform_eda(self) -> None:
        """
        Perform eda on df and save figures to images folder

        Parameters:
            self: The instance of the class
        Returns:
            None

        """

        # Select categorical features
        categorical_features = self.dataframe.select_dtypes(
            include=['object']
        ).columns.tolist()

        # Select numerical features
        numerical_features = self.dataframe.select_dtypes(
            include=['int64', 'float64']
        ).columns.tolist()

        for num_feature in numerical_features:
            plot_histogram(
                dataframe=self.dataframe,
                column=num_feature,
                output_path=self.eda_output_path
            )

        for cat_feature in categorical_features:
            plot_barplot(
                dataframe=self.dataframe,
                column=cat_feature,
                output_path=self.eda_output_path
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
