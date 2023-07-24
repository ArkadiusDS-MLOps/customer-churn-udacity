"""
Tests and logging for churn modelling methods and functions

Author: Arkadiusz Modzelewski
"""
import logging
import pytest
import churn_model as cls

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


# def test_eda():
#     """
#     Test perform eda function
#     """
#
#
# def test_encoder_helper(encoder_helper):
#     """test encoder helper"""
#
#
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
