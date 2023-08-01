"""
Tests and logging for churn modelling methods and functions

Author: Arkadiusz Modzelewski
"""
import logging

import numpy as np
import pandas as pd
import pytest
import churn_model as cls
from unittest.mock import patch, MagicMock

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


# Fixture for importing data
@pytest.fixture
def imported_data():
    # Code to import data using the import_data method
    try:
        churn = cls.ChurnModel()
        churn.import_data("./data/bank_data.csv")
        logging.info("Using import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Using import_eda: The file wasn't found")
        raise err
    return churn.dataframe


def test_import_check_shape(imported_data):
    """
    Test data import - this example is completed for you to assist with the other test functions
    """

    try:
        assert imported_data.shape[0] > 0
        assert imported_data.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_import_check_target(imported_data):
    """
    Test data import - this example is completed for you to assist with the other test functions
    """

    try:
        assert 'Churn' in list(imported_data.columns)
    except AssertionError as err:
        logging.error("Testing import_data:Target variable was not created")
        raise err


def test_import_check_dropped_cols(imported_data):
    """
    Test data import - this example is completed for you to assist with the other test functions
    """

    try:
        assert 'Attrition_Flag' not in list(imported_data.columns)
        assert 'Unnamed: 0' not in list(imported_data.columns)
    except AssertionError as err:
        logging.error(
            "Testing import_data:'Unnamed: 0' or 'Attrition_Flag' columns was not deleted"
        )
        raise err


@pytest.fixture
def churn_model_instance():
    # Create an instance of ChurnModel
    churn_model_instance = cls.ChurnModel()

    # Sample data for testing
    data = {
        'Category1': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B'],
        'Category2': ['X', 'Y', 'X', 'Z', 'X', 'Z', 'X', 'Y'],
        'Churn': [1, 0, 0, 1, 1, 0, 0, 1]
    }
    churn_model_instance.dataframe = pd.DataFrame(data)

    return churn_model_instance


# Mocked plot functions
@patch("churn_model.plot_histogram")
@patch("churn_model.plot_barplot")
@patch("churn_model.plot_corr_heatmap")
def test_perform_eda(
        mock_corr_heatmap, mock_barplot, mock_histogram, churn_model_instance
):
    # Call the perform_eda method with a custom output path
    eda_output_path = "test_output/"
    churn_model_instance.perform_eda(eda_output_path)

    # Assert that the functions were called with the expected arguments
    for num_feature in churn_model_instance.numerical_features:
        mock_histogram.assert_any_call(
            dataframe=churn_model_instance.dataframe,
            column=num_feature,
            output_path=eda_output_path
        )

    for cat_feature in churn_model_instance.categorical_features:
        mock_barplot.assert_any_call(
            dataframe=churn_model_instance.dataframe,
            column=cat_feature,
            output_path=eda_output_path
        )

    mock_corr_heatmap.assert_called_once_with(
        dataframe=churn_model_instance.dataframe,
        output_path=eda_output_path
    )


# def test_eda():
#     """
#     Test perform eda function
#     """
#
#
def test_col_presence_encoder_helper(churn_model_instance):
    """test encoder helper if creates columns and save it to dataframe"""
    # Call the encode_cat_features method with the given category_list
    category_list = ['Category1', 'Category2']
    churn_model_instance.encode_cat_features(category_list)

    # Check if new columns have been added
    for column in category_list:
        new_column = column + '_Churn'
        assert new_column in churn_model_instance.dataframe.columns


def test_col_data_encoder_helper(churn_model_instance):
    """test encoder helper if data in new columns is correct"""
    # Call the encode_cat_features method with the given category_list
    category_list = ['Category1', 'Category2']
    churn_model_instance.encode_cat_features(category_list)

    # Check the correctness of the encoding
    expected_result = {
        'Category1_Churn': [0.33333, 0.66667, 0.33333, 0.5, 0.66667, 0.5, 0.33333, 0.66667],
        'Category2_Churn': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    }
    for col, expected_values in expected_result.items():
        np.testing.assert_almost_equal(
            actual=list(churn_model_instance.dataframe[col]),
            desired=expected_values,
            decimal=5
        )


# def test_perform_feature_engineering(perform_feature_engineering):
#     """test perform_feature_engineering"""
#
#
# def test_train_models(train_models):
#     """test train_models"""
#
#
# if __name__ == "__main__":
#     pass
