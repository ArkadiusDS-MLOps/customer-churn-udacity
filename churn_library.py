"""
Prediction of Customer Churn

Author: Arkadiusz Modzelewski
"""

# import libraries
import os
from churn_model import ChurnModel
from constants import CAT_COLUMNS, COLUMNS_TO_KEEP
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


if __name__ == '__main__':

    churn = ChurnModel(data_path="data/bank_data.csv")
    churn.import_data()

    churn.perform_eda()

    churn.encode_cat_features(category_list=CAT_COLUMNS)

    churn.perform_feature_engineering(cols_to_keep=COLUMNS_TO_KEEP)

    churn.train_models()
