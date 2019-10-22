from getdata import GetHousingData
from preparedata import DataPreparator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

TARGET_COL = 'median_house_value'
TARGET_TYPE = 'num'
STRATIFY_COL = 'median_income'

#### fetch data
housingDatafetcher = GetHousingData()
housing = housingDatafetcher.load_housing_data()

#### split data
from splitdata import SplitData as sd

#housing_r_train_set, housing_r_test_set = sd.get_random_split(housing)
housing_s_train_set, housing_s_test_set, params = sd.getStratifiedSplit(housing, STRATIFY_COL)
train_df = housing_s_train_set.copy()
test_df = housing_s_test_set.copy()

#### Explore

from exploredata import DataExplorer

data_info = DataExplorer.get_base_info(train_df, target_col=TARGET_COL, target_type=TARGET_TYPE)
# large_corr_info = DataExplorer.get_large_corr_pairs(train_df)
# large_corr_info = DataExplorer.get_large_corr_with_target(train_df)

#### preparation

x, y, x_transformer, y_transformer = DataPreparator.pipeline_fit_transform(train_df, data_info)
x_test, y_test = DataPreparator.pipeline_transform(test_df, data_info, x_transformer, y_transformer)

#### train model

from buildmodels import Modeler

lin_reg_model, lin_cv_score = Modeler.lin_reg(x, y)
tree_reg_model, tree_cv_score = Modeler.tree_reg(x, y)
rf_reg_model, rf_cv_score = Modeler.rf_reg(x, y)

#### test set
from sklearn.metrics import mean_squared_error
y_test_lin_prediction = lin_reg_model.predict(x_test)
rmse_lin = np.sqrt(mean_squared_error(y_test_lin_prediction, y_test))

y_test_tree_prediction = tree_reg_model.predict(x_test)
rmse_tree = np.sqrt(mean_squared_error(y_test_tree_prediction, y_test))

y_test_rf_prediction = rf_reg_model.predict(x_test)
rmse_rf = np.sqrt(mean_squared_error(y_test_rf_prediction, y_test))

#### tune

rf_reg_model_tuned, rf_reg_tuner = Modeler.tune(Modeler.rf_reg_model, x, y)
rf_reg_tuner.cv_results_