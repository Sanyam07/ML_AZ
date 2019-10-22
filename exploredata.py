# explore data
import numpy as np
import pandas as pd
import collections
from itertools import chain
from util import Util
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


class DataExplorer:
    data = pd.DataFrame(np.zeros((10, 10)))
    DataInfo = collections.namedtuple('DataInfo', ('columns',
                                                   'num_cols',
                                                   'cat_cols',
                                                   'n_num_cols',
                                                   'n_cat_cols',
                                                   'target_col',
                                                   'target_type')
                                      )
    # data_info = DataInfo(columns=data.columns,
    #                      num_cols=data.select_dtypes(include='number').columns,
    #                      cat_cols=data.select_dtypes(include='object').columns,
    #                      n_num_cols=len(data.select_dtypes(include='number').columns),
    #                      n_cat_cols=len(data.select_dtypes(include='number').columns)
    #                      )

    def __init__(self, df):
        self.data = df
        self.data_info = DataExplorer.get_base_info(df)

    @staticmethod
    def get_base_info(df, target_col=None, target_type='num'):
        df_h = df.head().copy()
        if target_col is not None:
            df_h.drop(target_col, axis=1, inplace=True)

        data_info = DataExplorer.DataInfo(columns=df_h.columns,
                                          num_cols=df_h.select_dtypes(include='number').columns,
                                          cat_cols=df_h.select_dtypes(include='object').columns,
                                          n_num_cols=len(df_h.select_dtypes(include='number').columns),
                                          n_cat_cols=len(df_h.select_dtypes(include='number').columns),
                                          target_col=target_col,
                                          target_type=target_type
                                          )
        return data_info

    @staticmethod
    def corr_analysis(df):

        return

    @staticmethod
    def get_large_corr_pairs(df, n=3):

        corr_mat = df.corr()
        #corr_mat_abs_np = abs(corr_mat).to_numpy() - np.eye(len(corr_mat))
        corr_mat_abs_up_np = np.triu(abs(corr_mat),1)
        large_corr_pairs = Util.largest_indices(corr_mat_abs_up_np, n)
        var_pairs = [(corr_mat.index[large_corr_pairs[0][s]], corr_mat.columns[large_corr_pairs[1][s]])  for s in list(range(3))]
        var_set = list(set(chain.from_iterable(var_pairs)))
        #scatter_matrix(df[var_set])
        large_corr_info = {'n' : n,
                           'idx':large_corr_pairs,
                           'corr':corr_mat_abs_up_np[large_corr_pairs],
                           'var_pairs': var_pairs,
                           'var_set': var_set,
                           'corr_mat': corr_mat}
        DataExplorer.corr_plotter(df, large_corr_info)
        return large_corr_info

    @staticmethod
    def get_large_corr_with_target(df, target_col = 'median_house_value', n=3):
        large_corr_info = DataExplorer.get_large_corr_pairs(df, n)
        corr_vec = large_corr_info['corr_mat'][target_col].sort_values(ascending = False).drop(target_col)
        large_corr_with_target = corr_vec[:n]
        var_pairs_with_target = [(large_corr_with_target.index[s], target_col) for s in
                     list(range(3))]
        var_set_with_target = list(large_corr_with_target.index)
        large_corr_info.update({'large_corr_with_target': large_corr_with_target,
                                'var_pairs_with_target': var_pairs_with_target,
                                'var_set_with_target': var_set_with_target,
                                'target_col': target_col})
        DataExplorer.corr_with_target_plotter(df, large_corr_info)
        return large_corr_info

    @staticmethod
    def corr_with_target_plotter(df, info):
        n=info['n']
        fig = plt.figure(2, figsize=(12, 5))
        plt.title('large corr with target')
        for idx in list(range(n)):
            plt.subplot(3,3,idx+1)
            x_col, y_col = info['var_pairs_with_target'][idx][0], info['target_col']
            plt.scatter(df[x_col], df[y_col], alpha=0.3)
            plt.title(x_col + ' / ' + y_col)
            plt.xlabel(x_col)
            plt.ylabel(y_col)


    @staticmethod
    def corr_plotter(df, info):
        n=info['n']
        fig = plt.figure(1, figsize=(12, 5))
        for idx in list(range(n)):
            plt.subplot(3,3,idx+1)
            x_col, y_col = info['var_pairs'][idx][0], info['var_pairs'][idx][1]
            plt.scatter(df[x_col], df[y_col], alpha=0.3)
            plt.title(x_col + ' / ' + y_col)
            plt.xlabel(x_col)
            plt.ylabel(y_col)

    @staticmethod
    def geo_plotter(df, xCol = 'longitude', yCol = 'latitude', areaCol = 'population', colorCol = 'median_house_value'):
        fig = plt.figure(11, figsize=(7, 5))
        df.plot(kind='scatter', x=xCol, y=yCol, alpha=0.4, s=df[areaCol] / 100,
                     c=df[colorCol], cmap='jet')
        #plt.colorbar()


    # list(housing.columns)
    # housing.head()
    # housing.describe()
    # housing.info()
    #
    # housing['ocean_proximity'].value_counts()
    #
    # housing.hist(bins=50, figsize=(20, 15))
    #
    # test_set.describe()
    # test_set.info()
    # test_set['housing_median_age'].describe()
    #
    # housing['median_income'].describe()
    # housing['income_cat'] = pd.cut(housing['median_income'], bins=[0., 1.5, 3.0, 4.5, 6, np.inf],
    #                                labels=[1, 2, 3, 4, 5])
    # housing['income_cat'].hist()
