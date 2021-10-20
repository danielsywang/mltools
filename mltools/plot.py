import itertools
import math

import pandas as pd
import numpy as np
from sklearn.preprocessing import (MinMaxScaler, PowerTransformer,
                                   QuantileTransformer, RobustScaler,
                                   StandardScaler)
import matplotlib.pyplot as plt
import seaborn as sns

# 2021.09.14 Created by Daniel SY wang


def get_colpairlist(col_list):
    """
    生成列对的列表，为探究多变量间分布关系作准备.
    colpairlist的结构为[(x1,y1),(x2,y2),...,(xp1,yp1),(xp2,yp2),...,...]

    Parameters
    ----------
    col_list : list
        such as ['col1','col2','col3','col4']

    Returns
    ------
    colpairlist : list of tuple
        column pair list for plot,such as [('col1','col2'),('col1','col3'),('col3','col4')] 

    """

#    col_list=list(data.columns)[:5]
    colpairlist = list(itertools.combinations(col_list, 2))

    return colpairlist


def pair_plot(data, colpairlist, ncol=5, figsize_x=20, scale_y=1, plot_type='scatter'):
    """
    生成多个关系图的列表，探究多变量间分布关系。
    绘图方法包括：散点图(scatter)、散点回归图(reg)、回归残差图(resid)、散点分布图(joint)、核密度估计图(kde)

    Parameters
    ----------
    data : df
        pandas Dataframe
    colpairslist : list of tuple
        column pair list, such as [('col1','col2'),('col1','col3'),('1','col4'),('col2','col3'),('col2','col4')]
    ncol : int
        subfigure column numbers in a row
    figsize_x : float
        sub-figure width
    scale_y : float
        scale height
    plot_type : string
        'scatter'(default), 'reg','resid','joint','kde'

    Returns
    -------
    return : None
        plot multiple figure matrix

    """
    nrow = math.ceil(len(colpairlist)/ncol)

    if plot_type in ['scatter', 'reg', 'resid', 'kde']:
        figsize_y = nrow*figsize_x/ncol*scale_y
        fig, axes = plt.subplots(
            nrows=nrow, ncols=ncol, figsize=(figsize_x, figsize_y))

        for colpair, ax in zip(colpairlist, axes.flat):

            if plot_type == 'scatter':
                sns.scatterplot(data=data, x=colpair[0], y=colpair[1], ax=ax)
            elif plot_type == 'reg':
                sns.regplot(data=data, x=colpair[0], y=colpair[1], ax=ax)
            elif plot_type == 'resid':
                sns.residplot(
                    data=data, x=colpair[0], y=colpair[1], ax=ax, lowess=True)
            elif plot_type == 'kde':
                sns.kdeplot(
                    data=data, x=colpair[0], y=colpair[1], shade=True, ax=ax)

    if plot_type == 'joint':
        for colpair in colpairlist:
            sns .jointplot(data=data, x=colpair[0], y=colpair[1])

    return


def hist_plot(data, ncol=5, n_bin=100, figsize_x=20, scale_y=1):
    """
    plot histograms of DataFrame

    Parameters
    ----------
    data : df
        pandas Dataframe
    ncol : int
        subfigure column numbers in a row
    figsize_x : float
        sub-figure width
    scale_y : float
        scale height

    Returns
    -------
    return : None
        plot multiple figures

    """
    nrow = math.ceil(len(list(data.columns))/ncol)

    figsize_y = nrow*figsize_x/ncol*scale_y
    data.hist(layout=(-1, ncol), figsize=(figsize_x, figsize_y), bins=n_bin)


def trans_plot(data, trans_method_list='all', ncol=5, n_bin=100, figsize_x=20, scale_y=1):
    """
    plot histograms of each column and its transformed of DataFrame \n

    Parameters
    ----------
    data : df
        pandas Dataframe
    trans_method_list : list
        a list of 'sqrt','cbrt','log1p','reciprocal',
        'square','cube','expm1','minmax','zscore',
        'robust','quant','box','yeo'.
        user can choice some transfomer method as a list.
        such as: list of 'minmax','zscore','robust','box','yeo'.
        default includes all method.    
    ncol : int
        subfigure column numbers in a row
    figsize_x : float
        sub-figure width
    scale_y : float
        scale height

    Returns
    -------
    return : None
        plot multiple figures

    """
    if trans_method_list == 'all':
        trans_method_list = ['sqrt', 'cbrt', 'log1p', 'reciprocal', 'square',
                             'cube', 'expm1', 'minmax', 'zscore', 'robust', 'quant', 'box', 'yeo']

    for col in data.columns:

        data_col = np.array(data[col]).reshape(-1, 1)
        df = pd.DataFrame(data=data_col, columns=[col])
        for trans_method in trans_method_list:
            try:
                if trans_method == 'sqrt':
                    temp_data = np.sqrt(data_col)
                elif trans_method == 'cbrt':
                    temp_data = np.cbrt(data_col)
                elif trans_method == 'log1p':
                    temp_data = np.log1p(data_col)
                elif trans_method == 'reciprocal':
                    temp_data = np.reciprocal(data_col)
                elif trans_method == 'square':
                    temp_data = np.square(data_col)
                elif trans_method == 'cube':
                    temp_data = np.power(data_col, 3)
                elif trans_method == 'expm1':
                    temp_data = np.expm1(data_col)
                elif trans_method == 'minmax':
                    trans = MinMaxScaler()
                elif trans_method == 'zscore':
                    trans = StandardScaler()
                elif trans_method == 'robust':
                    trans = RobustScaler()
                elif trans_method == 'quant':
                    trans = QuantileTransformer(
                        n_quantiles=500, output_distribution='normal', random_state=100)
                elif trans_method == 'box':
                    # The Box-Cox transformation can only be applied to strictly positive data
                    trans = PowerTransformer(method='box-cox')
                elif trans_method == 'yeo':
                    trans = PowerTransformer(method='yeo-johnson')

                if trans_method in ['minmax', 'zscore', 'robust', 'quant', 'box', 'yeo']:
                    temp_data = trans.fit_transform(data_col)

                col_name = col+'__'+trans_method
                df[col_name] = temp_data
            except:
                pass

        hist_plot(df, ncol=ncol, n_bin=n_bin,
                  figsize_x=figsize_x, scale_y=scale_y)
    return 0
