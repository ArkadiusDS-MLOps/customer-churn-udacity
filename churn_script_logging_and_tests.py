import os
import logging
import churn_model as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    """
    Test data import - this example is completed for you to assist with the other test functions
    """

    try:
        churn = cls.ChurnModel()
        churn.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert churn.dataframe.shape[0] > 0
        assert churn.dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


# def test_eda():
#     """
#     Test perform eda function
#     """
#
#
# def test_encoder_helper(encoder_helper):
#     '''
# 	test encoder helper
# 	'''
#
#
# def test_perform_feature_engineering(perform_feature_engineering):
#     '''
# 	test perform_feature_engineering
# 	'''
#
#
# def test_train_models(train_models):
#     '''
# 	test train_models
# 	'''
#
#
# if __name__ == "__main__":
#     pass
