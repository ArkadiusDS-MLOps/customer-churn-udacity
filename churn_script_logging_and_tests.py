"""
Tests and logging for churn modelling methods and functions

Author: Arkadiusz Modzelewski
"""
import logging
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

import churn_model as cls
from sklearn.datasets import load_iris, make_classification

from constants import CAT_COLUMNS, COLUMNS_TO_KEEP

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


# Fixture for importing data
@pytest.fixture
def imported_data():
    """
    Code to import data using the import_data method
    """
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
def churn_model_instance(imported_data):
    """
    Create an instance of ChurnModel with dataframe
    """

    churn_model_instance = cls.ChurnModel()
    churn_model_instance.dataframe = imported_data

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


def test_col_presence_encoder_helper(churn_model_instance):
    """test encoder helper if creates columns and save it to dataframe"""
    # Call the encode_cat_features method with the given category_list
    churn_model_instance.encode_cat_features(CAT_COLUMNS)
    for col in COLUMNS_TO_KEEP:
        assert col in churn_model_instance.dataframe.columns


def test_perform_feature_engineering(churn_model_instance):
    """test perform_feature_engineering"""

    churn_model_instance.encode_cat_features(CAT_COLUMNS)
    # Call the function you want to test
    churn_model_instance.perform_feature_engineering(COLUMNS_TO_KEEP)

    # Check if the attributes are correctly assigned
    assert churn_model_instance.x_train is not None
    assert churn_model_instance.x_test is not None
    assert churn_model_instance.y_train is not None
    assert churn_model_instance.y_test is not None

    # Check if the shapes of train and test data match
    assert churn_model_instance.x_train.shape[0] + churn_model_instance.x_test.shape[0] == \
           len(churn_model_instance.dataframe['Churn'])

    # Check if the length of y_train and y_test matches the data length
    assert len(churn_model_instance.y_train) + len(churn_model_instance.y_test) == \
           len(churn_model_instance.dataframe['Churn'])


def test_train_models(churn_model_instance):
    """test train_models"""

    # Encoding categorical features
    churn_model_instance.encode_cat_features(category_list=CAT_COLUMNS)
    # Feature engineering and train test split
    churn_model_instance.perform_feature_engineering(cols_to_keep=COLUMNS_TO_KEEP)

    # Call the function you want to test
    churn_model_instance.train_models()

    assert os.path.exists("images/results/classification_report.png")
    assert os.path.exists("images/results/roc_curves.png")
    assert os.path.exists("images/results/feature_importance_plot.png")
    assert os.path.exists("models/rfc_model.pkl")
    assert os.path.exists("models/logistic_model.pkl")
