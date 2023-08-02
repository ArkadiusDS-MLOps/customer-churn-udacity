"""
Tests and logging for churn modelling methods and functions

Author: Arkadiusz Modzelewski
"""
import logging
import os
from unittest.mock import patch
import pytest
import churn_model as cls
from constants import CAT_COLUMNS, COLUMNS_TO_KEEP

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


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
        logging.error("ERROR Using import_eda: The file wasn't found")
        raise err
    return churn.dataframe


def test_import_check_shape(imported_data):
    """
    Test data import - this example is completed for you to assist with the other test functions
    """

    try:
        assert imported_data.shape[0] > 0
        assert imported_data.shape[1] > 0
        logging.info("SUCCESS: Test import_check_shape")
    except AssertionError as err:
        logging.error("ERROR: Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_import_check_target(imported_data):
    """
    Test data import - this example is completed for you to assist with the other test functions
    """

    try:
        assert 'Churn' in list(imported_data.columns)
        logging.info("SUCCESS: Test import_check_target")
    except AssertionError as err:
        logging.error("ERROR: Testing import_data:Target variable was not created")
        raise err


def test_import_check_dropped_cols(imported_data):
    """
    Test data import - this example is completed for you to assist with the other test functions
    """

    try:
        assert 'Attrition_Flag' not in list(imported_data.columns)
        assert 'Unnamed: 0' not in list(imported_data.columns)
        logging.info("SUCCESS: Test import_check_dropped_cols")
    except AssertionError as err:
        logging.error(
            "ERROR Testing import_data:'Unnamed: 0' or 'Attrition_Flag' columns was not deleted"
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


def test_perform_eda(churn_model_instance):
    """
    Testing perform eda
    """

    try:
        churn_model_instance.perform_eda()
        logging.info("SUCCESS: Testing perform_eda")
    except AssertionError as err:
        logging.error(
            "ERROR: Testing perform_eda: calling function"
        )
        raise err


# Mocked plot functions
@patch("churn_model.plot_histogram")
@patch("churn_model.plot_barplot")
@patch("churn_model.plot_corr_heatmap")
def test_perform_eda_plot_functions(
        mock_corr_heatmap, mock_barplot, mock_histogram, churn_model_instance
):
    """
    Testing perform eda plot functions
    """
    # Call the perform_eda method with a custom output path
    eda_output_path = "test_output/"
    churn_model_instance.perform_eda(eda_output_path)

    # Assert that the functions were called with the expected arguments
    try:
        for num_feature in churn_model_instance.numerical_features:
            mock_histogram.assert_any_call(
                dataframe=churn_model_instance.dataframe,
                column=num_feature,
                output_path=eda_output_path
            )
        logging.info("SUCCESS: Testing plot_histogram: calling function")
    except AssertionError as err:
        logging.error(
            "ERROR: Testing plot_histogram: calling function"
        )
        raise err
    try:
        for cat_feature in churn_model_instance.categorical_features:
            mock_barplot.assert_any_call(
                dataframe=churn_model_instance.dataframe,
                column=cat_feature,
                output_path=eda_output_path
            )
        logging.info("SUCCESS: Testing plot_barplot: calling function")
    except AssertionError as err:
        logging.error(
            "ERROR: Testing plot_barplot: calling function"
        )
        raise err
    try:

        mock_corr_heatmap.assert_called_once_with(
            dataframe=churn_model_instance.dataframe,
            output_path=eda_output_path
        )
        logging.info("SUCCESS: Testing plot_corr_heatmap: calling function")
    except AssertionError as err:
        logging.error(
            "ERROR: Testing plot_corr_heatmap: calling function"
        )
        raise err


def test_col_presence_encoder_helper(churn_model_instance):
    """
    Test encoder helper if creates columns and save it to dataframe
    """
    # Call the encode_cat_features method with the given category_list
    churn_model_instance.encode_cat_features(CAT_COLUMNS)
    for col in COLUMNS_TO_KEEP:
        try:
            assert col in churn_model_instance.dataframe.columns
            logging.info(
                f"SUCCESS: Testing col_presence_encoder_helper: {col} available".format(col)
            )
        except AssertionError as err:
            logging.error(
                "ERROR: Testing col_presence_encoder_helper: calling function"
            )
            raise err


def test_perform_feature_engineering(churn_model_instance):
    """
    Test perform_feature_engineering
    """

    churn_model_instance.encode_cat_features(CAT_COLUMNS)
    # Call the function you want to test
    churn_model_instance.perform_feature_engineering(COLUMNS_TO_KEEP)
    try:
        # Check if the attributes are correctly assigned
        assert churn_model_instance.x_train is not None
        assert churn_model_instance.x_test is not None
        assert churn_model_instance.y_train is not None
        assert churn_model_instance.y_test is not None
        logging.info(
            f"SUCCESS: Testing perform_feature_engineering: "
            f"x_train, x_test, y_train, y_test available"
        )
    except AssertionError as err:
        logging.error(
            "ERROR: Testing perform_feature_engineering: x_train or x_test or y_train or y_test "
            "not available in class instance"
        )
        raise err

    # Check if the shapes of train and test data match
    try:
        assert churn_model_instance.x_train.shape[0] + churn_model_instance.x_test.shape[0] == \
               len(churn_model_instance.dataframe['Churn'])

        # Check if the length of y_train and y_test matches the data length
        assert len(churn_model_instance.y_train) + len(churn_model_instance.y_test) == \
               len(churn_model_instance.dataframe['Churn'])
        logging.info(
            "SUCCESS: Testing perform_feature_engineering: train and test data match "
        )
    except AssertionError as err:
        logging.error(
            "ERROR: Testing perform_feature_engineering: train and test data does not match "
        )
        raise err


def test_train_models(churn_model_instance):
    """
    Test train_models
    """

    # Encoding categorical features
    churn_model_instance.encode_cat_features(category_list=CAT_COLUMNS)
    # Feature engineering and train test split
    churn_model_instance.perform_feature_engineering(cols_to_keep=COLUMNS_TO_KEEP)

    # Call the function you want to test
    churn_model_instance.train_models()
    try:
        assert os.path.exists("images/results/classification_report.png")
        assert os.path.exists("images/results/roc_curves.png")
        assert os.path.exists("images/results/feature_importance_plot.png")
        assert os.path.exists("models/rfc_model.pkl")
        assert os.path.exists("models/logistic_model.pkl")
        logging.info(
            "SUCCESS: Testing train_models"
        )
    except AssertionError as err:
        logging.error(
            "ERROR: Testing train_models"
        )
        raise err


if __name__ == "__main__":

    churn = cls.ChurnModel()
    churn.import_data("./data/bank_data.csv")

    test_import_check_shape(churn.dataframe)
    test_import_check_target(churn.dataframe)
    test_import_check_dropped_cols(churn.dataframe)
    test_perform_eda(churn)
    test_col_presence_encoder_helper(churn)
    test_perform_feature_engineering(churn)
    test_train_models(churn)
