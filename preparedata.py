# prepare the data for predictive modeling

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class DataPreparator:

    @staticmethod
    def get_x_transformer(df: pd.DataFrame, data_info: object) -> pd.DataFrame:
        '''output is np.ndarray'''

        num_pipeline = Pipeline([
            ('median_imputer', SimpleImputer(strategy='median')),
            ('std_scaler', StandardScaler())
        ])
        cat_pipeline = Pipeline([
            ('const_impute', SimpleImputer(strategy='constant', fill_value='missing')),
            ('one_hot', OneHotEncoder())
        ])
        column_transformer = ColumnTransformer([
            ('num', num_pipeline, data_info.num_cols),
            ('cat', cat_pipeline, data_info.cat_cols),
        ])
        return column_transformer

    @staticmethod
    def get_y_transformer(df: pd.DataFrame, data_info: object) -> pd.DataFrame:
        '''output is np.ndarray'''

        if data_info.target_type == 'num':
            y_pipeline = Pipeline([
                ('median_imputer', SimpleImputer(strategy='median')),
                # ('std_scaler', StandardScaler())
            ])
        elif data_info.target_type == 'cat':
            y_pipeline = Pipeline([
                ('const_impute', SimpleImputer(strategy='constant', fill_value='missing')),
                ('one_hot', OneHotEncoder())
            ])
        else:
            print('unknown target type')
            y_pipeline = None

        return y_pipeline

    @staticmethod
    def pipeline_fit_transform(df, data_info):
        x_transformer = DataPreparator.get_x_transformer(df, data_info)
        y_transformer = DataPreparator.get_y_transformer(df, data_info)
        df_x, df_y = DataPreparator.separate_target(df, data_info.target_col)
        x = x_transformer.fit_transform(df_x)
        y = y_transformer.fit_transform(df_y)
        return x, y, x_transformer, y_transformer

    @staticmethod
    def pipeline_transform(df, data_info, x_transformer, y_transformer):
        df_x, df_y = DataPreparator.separate_target(df, data_info.target_col)
        x_test = x_transformer.transform(df_x)
        y_test = y_transformer.transform(df_y)
        return x_test, y_test

    @staticmethod
    def separate_target(df, target_col):
        df_y = df[target_col].to_frame(name=target_col)
        df_x = df.drop(target_col, axis=1)
        return df_x, df_y

    @staticmethod
    def impute(df_in, data_info):
        median_imputer = SimpleImputer(strategy='median')
        df_num = df_in[data_info.num_cols].coppy()
        median_imputer.fit(df_num)
        np_out = median_imputer.transform(df_num)
        df_num_out = pd.DataFrame(np_out, columns=df_num.columns, index=df_num.index)

        cat_impute_str = 'NA'
        df_cat = df_in[data_info.cat_cols].copy()












